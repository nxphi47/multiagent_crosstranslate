import os
import math
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import apex
import random

from functools import partial

from .optim import get_optimizer
from .utils import to_cuda, concat_batches, find_modules
from .utils import parse_lambda_config, update_lambdas
from .model.memory import HashingMemory
from .model.transformer import TransformerFFN
from .model import build_model
import copy

from .evaluation.online_bleu import SentBLEUV2
from . import noising

from .trainer import EncDecTrainer, convert_to_text

logger = getLogger()


class EncDecMACDmbeamOnlineTrainer(EncDecTrainer):
    def __init__(self, encoder, decoder, encoder1, decoder1, encoder2, decoder2, data, params):
        self.MODEL_NAMES = ['encoder', 'decoder', 'encoder1', 'decoder1', 'encoder2', 'decoder2']
        self.MODEL_NOUSE_NAMES = ['encoder1', 'decoder1', 'encoder2', 'decoder2']
        self.encoder = encoder
        self.decoder = decoder
        self.encoder1 = encoder1
        self.decoder1 = decoder1
        self.encoder2 = encoder2
        self.decoder2 = decoder2
        self.data = data
        self.params = params
        self.onl_bleu_scorer = SentBLEUV2(self.params.pad_index, self.params.eos_index, self.params.unk_index)
        if getattr(params, 'bt_sync', -1) > 1:
            raise NotImplementedError
        else:
            self.encoder_clone, self.decoder_clone = None, None

        super(EncDecTrainer, self).__init__(data, params)
        # probability of masking out / randomize / not modify words to predict
        # params.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

    def save_checkpoint(self, name, include_optimizers=True):
        """
                Save the model / checkpoints.
                """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            if name in self.MODEL_NOUSE_NAMES:
                logger.warning("Skip saving {} parameters ...".format(name))
                continue
            logger.warning("Saving {} parameters ...".format(name))
            data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            data['amp'] = apex.amp.state_dict()
            for name in self.optimizers.keys():
                logger.warning("Saving {} optimizer ...".format(name))
                data['{}_optimizer'.format(name)] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning("Reloading checkpoint from {} ...".format(checkpoint_path))
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for name in self.MODEL_NAMES:
            if name in self.MODEL_NOUSE_NAMES:
                logger.warning("Skip loading {} parameters ...".format(name))
                continue
            if "_clone" in name or "bertscore_model" in name:
                # fixme: quick fix
                continue
            try:
                getattr(self, name).load_state_dict(data[name])
            except Exception as e:
                print('Reload error: name={}, try removing "module."'.format(name))
                reload = data[name]
                if all([k.startswith('module.') for k in reload.keys()]):
                    reload = {k[len('module.'):]: v for k, v in reload.items()}
                try:
                    getattr(self, name).load_state_dict(reload)
                except Exception as ee:
                    print('Reload error again: name={}, try adding "module."'.format(name))
                    reload = data[name]
                    if not any([k.startswith('module.') for k in reload.keys()]):
                        reload = {'module.{}'.format(k): v for k, v in reload.items()}
                    getattr(self, name).load_state_dict(reload)
                # raise e

        # reload optimizers
        if self.params.amp > 0 and "amp" in data:
            assert "amp" in data, 'amp must be in data for amp={}'.format(self.params.amp)
            logger.info('Load AMP parameters...')
            apex.amp.load_state_dict(data['amp'])
        else:
            logger.info('!!! amp not in data, so optimizer parameters will not be loaded')

        for name in self.optimizers.keys():

            if "amp" in data:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                logger.warning("Reloading checkpoint optimizer {} ...".format(name))
                self.optimizers[name].load_state_dict(data['{}_optimizer'.format(name)])
            else:  # instead, we only reload current iterations / learning rates
                logger.warning("Not reloading checkpoint optimizer {}.".format(name))
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning("No 'num_updates' for optimizer {}.".format(name))
                        continue
                    logger.warning("Reloading 'num_updates' and 'lr' for optimizer {}.".format(name))
                    param_group['num_updates'] = data['{}_optimizer'.format(name)]['param_groups'][group_id][
                        'num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(
            "Checkpoint reloaded. Resuming at epoch {} / iteration {} ...".format(self.epoch, self.n_total_iter))

    def mask_word(self, w):
        _w_real = w
        # _w_rand = np.random.randint(self.params.n_words, size=w.shape)
        # _w_mask = np.full(w.shape, self.params.mask_index)
        _w_rand = torch.randint(self.params.n_words, size=w.size(), device=w.device)
        _w_mask = w.new(w.size()).fill_(self.params.mask_index)
        probs = torch.multinomial(self.params.pred_probs, len(_w_real), replacement=True).to(w.device)
        # probs = probs.cpu()
        # _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
        # _w = _w_mask * (probs == 0) + _w_real * (probs == 1) + _w_rand * (probs == 2)
        _w = _w_mask * (probs == 0).long() + _w_real * (probs == 1).long() + _w_rand * (probs == 2).long()
        return _w

    def unfold_segments(self, segs):
        """Unfold the random mask segments, for example:
           The shuffle segment is [2, 0, 0, 2, 0],
           so the masked segment is like:
           [1, 1, 0, 0, 1, 1, 0]
           [1, 2, 3, 4, 5, 6, 7] (positions)
           (1 means this token will be masked, otherwise not)
           We return the position of the masked tokens like:
           [1, 2, 5, 6]
        """
        pos = []
        curr = 1  # We do not mask the start token
        for l in segs:
            if l >= 1:
                pos.extend([curr + i for i in range(l)])
                curr += l
            else:
                curr += 1
        return np.array(pos)

    def shuffle_segments(self, segs, unmasked_tokens):
        """
        We control 20% mask segment is at the start of sentences
                   20% mask segment is at the end   of sentences
                   60% mask segment is at random positions,
        """

        p = np.random.random()
        if p >= 0.8:
            shuf_segs = segs[1:] + unmasked_tokens
        elif p >= 0.6:
            shuf_segs = segs[:-1] + unmasked_tokens
        else:
            shuf_segs = segs + unmasked_tokens

        random.shuffle(shuf_segs)

        if p >= 0.8:
            shuf_segs = segs[0:1] + shuf_segs
        elif p >= 0.6:
            shuf_segs = shuf_segs + segs[-1:]
        return shuf_segs

    def get_segments(self, mask_len, span_len):
        segs = []
        while mask_len >= span_len:
            segs.append(span_len)
            mask_len -= span_len
        if mask_len != 0:
            segs.append(mask_len)
        return segs

    def restricted_mask_sent(self, x, l, span_len=100000):
        """ Restricted mask sents
            if span_len is equal to 1, it can be viewed as
            discrete mask;
            if span_len -> inf, it can be viewed as
            pure sentence mask
        """
        if span_len <= 0:
            span_len = 1
        max_len = 0
        positions, inputs, targets, outputs, = [], [], [], []
        # mask_len = round(len(x[:, 0]) * self.params.word_mass)
        mask_len = math.floor(len(x[:, 0]) * self.params.word_mass)
        len2 = [mask_len for i in range(l.size(0))]

        unmasked_tokens = [0 for i in range(l[0] - mask_len - 1)]
        segs = self.get_segments(mask_len, span_len)

        for i in range(l.size(0)):
            words = np.array(x[:l[i], i].tolist())
            shuf_segs = self.shuffle_segments(segs, unmasked_tokens)
            pos_i = self.unfold_segments(shuf_segs)
            output_i = words[pos_i].copy()
            target_i = words[pos_i - 1].copy()
            words[pos_i] = self.mask_word(words[pos_i])
            inputs.append(words)
            targets.append(target_i)
            outputs.append(output_i)
            positions.append(pos_i - 1)

        x1 = torch.LongTensor(max(l), l.size(0)).fill_(self.params.pad_index)
        x2 = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        y = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        pos = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        l1 = l.clone()
        l2 = torch.LongTensor(len2)
        for i in range(l.size(0)):
            x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
            x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
            y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
            pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))

        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)
        return x1, l1, x2, l2, y, pred_mask, pos

    def restricted_mask_sent_v3(self, x, l, span_len=100000):
        if span_len <= 0:
            span_len = 1
        max_len = 0
        bsz = l.size(0)
        # positions, inputs, targets, outputs, = [], [], [], []
        lens = l - 2  # remove eos
        device = x.device
        mask_len = (lens.float() * self.params.word_mass).long()
        mask_len_max = mask_len.max()
        # FIXME: only one segment is available
        # inputs = x.new(x.size(0), x.size(1)).fill_(self.params.pad_index)
        inputs = x.clone()
        targets = x.new(mask_len_max, x.size(1)).fill_(self.params.pad_index)
        outputs = x.new(mask_len_max, x.size(1)).fill_(self.params.pad_index)
        positions = x.new(mask_len_max, x.size(1)).fill_(0)

        # def random_mask_start(mlen_, len_):
        #     p = np.random.random()
        #     if p >= 0.8:
        #         return 0
        #     elif p >= 0.6:
        #         return len_ - mlen_
        #     else:
        #         return np.random.randint(0, len_ - mlen_)

        # mask_pos_rand = [random_mask_start(x, y) for x, y in zip(mask_len.cpu().numpy(), lens.cpu().numpy())]
        prob = torch.rand(lens.size(), device=device)
        mask_pos_rand = (torch.rand(lens.size(), device=device) * (lens - mask_len).float()).long()
        mask_pos_rand[prob >= 0.8] = 0
        mask_pos_rand[prob >= 0.6] = (lens - mask_len)[prob >= 0.6]
        include_indices = []
        for i in range(bsz):
            if mask_len[i] <= 0:
                continue
            include_indices.append(i)
            mask_idx = torch.arange(
                1 + mask_pos_rand[i], 1 + mask_pos_rand[i] + mask_len[i], dtype=torch.long, device=device)
            targets[:mask_idx.size(0), i] = x[mask_idx - 1, i]
            outputs[:mask_idx.size(0), i] = x[mask_idx, i]
            inputs[mask_idx, i] = self.mask_word(inputs[mask_idx, i])
            positions[:mask_idx.size(0), i] = mask_idx - 1
        x1 = inputs[:, include_indices]
        l1 = l[include_indices]
        x2 = targets[:, include_indices]
        l2 = mask_len[include_indices]
        positions = positions[:, include_indices]
        y = outputs[:, include_indices]
        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)
        # return inputs, l, targets, l2, y, pred_mask, positions
        # logger.info('out: {},{},{},{},{}'.format(x1.size(), l1, x2.size(), l2, positions.size()))
        return x1, l1, x2, l2, y, pred_mask, positions

    def _filter_valid_pairs(self, x1, len1, lang1_id, x2, len2, lang2_id):
        params = self.params
        max_len = params.max_len
        filter_bleu = getattr(params, 'filter_bleu', 0)
        bsz = len1.size(0)
        indices = torch.arange(bsz, dtype=torch.long, device=x1.device)
        fil_idx = indices[(len1 <= max_len) & (len2 <= max_len) & (len1 > 2) & (len2 > 2)]
        if fil_idx.numel() == 0:
            return None, None, None, None, None, None
        _len1 = len1[fil_idx]
        _len2 = len2[fil_idx]

        fil_idx1 = fil_idx.unsqueeze(0).expand(x1.size(0), fil_idx.size(0))
        fil_idx2 = fil_idx.unsqueeze(0).expand(x2.size(0), fil_idx.size(0))
        _x1 = x1.gather(dim=1, index=fil_idx1)[:_len1.max()]
        _x2 = x2.gather(dim=1, index=fil_idx2)[:_len2.max()]
        _langs1 = _x1.clone().fill_(lang1_id)
        _langs2 = _x2.clone().fill_(lang2_id)
        new_bsz = _x1.size(1)

        if filter_bleu > 0:
            # filter sentences that BLEU is less then filter_bleu
            _x1_l, _len1_l, _x2_l, _len2_l = [], [], [], []
            for i in range(new_bsz):
                xx = _x1[:_len1[i], i]
                yy = _x2[:_len1[i], i]

                bleu = self.onl_bleu_scorer.get_bleu(xx, yy)
                if bleu < filter_bleu:
                    _x1_l.append(_x1[:, i:i + 1])
                    _x2_l.append(_x2[:, i:i + 1])
                    _len1_l.append(_len1[i:i + 1])
                    _len2_l.append(_len2[i:i + 1])
            if len(_x1_l) > 0:
                self.onl_bleu_scorer.add_qualified(len(_x1_l))
                _len1 = torch.cat(_len1_l, 0)
                _len2 = torch.cat(_len2_l, 0)
                _x1 = torch.cat(_x1_l, 1)[:_len1.max()]
                _x2 = torch.cat(_x2_l, 1)[:_len2.max()]

                _langs1 = _x1.clone().fill_(lang1_id)
                _langs2 = _x2.clone().fill_(lang2_id)
            else:
                return None, None, None, None, None, None
        return _x1, _len1, _langs1, _x2, _len2, _langs2

    def _filter_long_sents(self, x1, len1):
        params = self.params
        max_len = params.max_len
        bsz = len1.size(0)
        indices = torch.arange(bsz, dtype=torch.long, device=x1.device)
        fil_idx = indices[len1 <= max_len]
        if fil_idx.numel() == 0:
            return None, None
        _len1 = len1[fil_idx]
        fil_idx1 = fil_idx.unsqueeze(0).expand(x1.size(0), fil_idx.size(0))
        _x1 = x1.gather(dim=1, index=fil_idx1)[:_len1.max()]
        # _langs1 = _x1.clone().fill_(lang1_id)
        return _x1, _len1

    def mass_step_xz(self, lang, lambda_coeff, x_, len_):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        min_len = getattr(params, 'min_len', 5)
        self.encoder.train()
        self.decoder.train()
        if len_.min() - 2 < min_len:
            return

        lang1_id = params.lang2id[lang]
        lang2_id = params.lang2id[lang]
        # if x_ is None:
        #     x_, len_ = self.get_batch('mass', lang)

        (x1, len1, x2, len2, y, pred_mask, positions) = self.restricted_mask_sent_v3(x_, len_, int(params.lambda_span))
        if x1.size(1) == 0:
            return

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)

        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        enc_mask = x1.ne(params.mask_index)
        enc_mask = enc_mask.transpose(0, 1)

        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True,
                            src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)

        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('BTMA-%s' % lang)].append(loss.item())

        # self.optimize(loss, ['encoder', 'decoder'])
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def bt_step_mbeam_macd(
            self, lang1, lang2, lang3, lambda_coeff, direction1, enforce_default=False, first_sec=False, wrote=False
    ):
        """
        Back-translation step for machine translation.
        # direction1==True  -> lang1->lang2 and lang3->lang2
        # direction1==False -> lang2->lang1 and lang2->lang3
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        nbest = params.nbest
        tokens_per_batch = params.tokens_per_batch
        beam_size = params.beam_size
        select_opt = getattr(params, 'select_opt', 0)
        assert beam_size > 1
        assert nbest is not None and 1 <= nbest <= beam_size

        if params.multi_gpu:
            infer_encoder2 = self.encoder2.module
            infer_decoder2 = self.decoder2.module
            infer_encoder1 = self.encoder1.module
            infer_decoder1 = self.decoder1.module
        else:
            infer_encoder2 = self.encoder2
            infer_decoder2 = self.decoder2
            infer_encoder1 = self.encoder1
            infer_decoder1 = self.decoder1
        self.encoder2.eval()
        self.decoder2.eval()
        self.encoder1.eval()
        self.decoder1.eval()

        if first_sec:
            first_encoder = infer_encoder2
            first_decoder = infer_decoder2
            sec_encoder = infer_encoder1
            sec_decoder = infer_decoder1
        else:
            first_encoder = infer_encoder1
            first_decoder = infer_decoder1
            sec_encoder = infer_encoder2
            sec_decoder = infer_decoder2

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)
        bsz = x1.size(1)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # todo: primary training
        with torch.no_grad():
            xx2, llen2, llangs2 = self._generate_pair_mbeamct(
                first_encoder, first_decoder, x1, len1, langs1, lang2_id, nbest)
            _len1, _bsz = x1.size()
            xx1 = x1.unsqueeze(1).expand(_len1, nbest, _bsz).reshape(_len1, nbest * _bsz)
            llen1 = len1.unsqueeze(0).expand(nbest, _bsz).reshape(nbest * _bsz)
            llangs1 = xx1.clone().fill_(lang1_id)
            # del x1, langs1
            del x1
            del langs1
            # torch.cuda.empty_cache()

            if direction1:
                # lang1->lang2
                d1_x1, d1_len1, d1_langs1 = xx1, llen1, llangs1
                d1_x2, d1_len2, d1_langs2 = xx2, llen2, llangs2
                d1_lang1_id = lang1_id
                d1_lang2_id = lang2_id
            else:
                # lang2->lang1
                d1_x1, d1_len1, d1_langs1 = xx2, llen2, llangs2
                d1_x2, d1_len2, d1_langs2 = xx1, llen1, llangs1
                d1_lang1_id = lang2_id
                d1_lang2_id = lang1_id
            # loss1 = self._train_step_bt(xx2, llen2, llangs2, xx1, llen1, llangs1)
            d1_x1, d1_len1, d1_langs1, d1_x2, d1_len2, d1_langs2 = self._filter_valid_pairs(
                d1_x1, d1_len1, d1_lang1_id, d1_x2, d1_len2, d1_lang2_id)
            del xx1, llen1, llangs1

        if d1_x1 is not None:
            try:
                loss1 = self._train_step_bt(d1_x1, d1_len1, d1_langs1, d1_x2, d1_len2, d1_langs2)
                self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss1.item())
                loss1 = lambda_coeff * loss1
                # optimize
                self.optimize(loss1)
                del loss1
            except RuntimeError as e:
                print('OOM Occur, skip train step loss-1, local_rank: params.local_rank'.format(params.local_rank))
                # logger.info('OOM Occur, skip train step loss-1, local_rank: params.local_rank'.format(params.local_rank))
            # del xx1, llen1, llangs1
            # del d1_x1, d1_x1, d1_len1, d1_len2, d1_langs1, d1_langs2

        # torch.cuda.empty_cache()

        # todo: secondary training
        # select best sample for secondary training -> x2_in, len2_in
        with torch.no_grad():
            _len2, _nbsz2 = xx2.size()
            _xx2 = xx2.view(_len2, nbest, _bsz)
            _llen2 = llen2.view(nbest, _bsz)

            x2_in, len2_in = self._select_sec_default(_xx2, _llen2)
            x2_in, len2_in = self._filter_long_sents(x2_in, len2_in)
            if x2_in is None:
                self.n_sentences += params.batch_size
                self.stats['processed_s'] += len1.size(0)
                self.stats['processed_w'] += (len1 - 1).sum().item()
                return
            del xx2, llen2, llangs2, _xx2, _llen2

            # ----
            langs2_in = x2_in.clone().fill_(lang2_id)
            xx3, llen3, llangs3 = self._generate_pair_mbeamct(
                sec_encoder, sec_decoder, x2_in, len2_in, langs2_in, lang1_id, nbest)
            del langs2_in
            _len2, _bsz2 = x2_in.size()
            xx2_in = x2_in.unsqueeze(1).expand(_len2, nbest, _bsz2).reshape(_len2, nbest * _bsz2)
            llen2_in = len2_in.unsqueeze(0).expand(nbest, _bsz2).reshape(nbest * _bsz2)
            del x2_in, len2_in
            llangs2_in = xx2_in.clone().fill_(lang2_id)
            # torch.cuda.empty_cache()

            if direction1:
                # lang3->lang2
                d2_x1, d2_len1, d2_langs1 = xx3, llen3, llangs3
                d2_x2, d2_len2, d2_langs2 = xx2_in, llen2_in, llangs2_in
                d2_lang1_id = lang1_id
                d2_lang2_id = lang2_id
            else:
                # lang2->lang3
                d2_x1, d2_len1, d2_langs1 = xx2_in, llen2_in, llangs2_in
                d2_x2, d2_len2, d2_langs2 = xx3, llen3, llangs3
                d2_lang1_id = lang2_id
                d2_lang2_id = lang1_id

            d2_x1, d2_len1, d2_langs1, d2_x2, d2_len2, d2_langs2 = self._filter_valid_pairs(
                d2_x1, d2_len1, d2_lang1_id, d2_x2, d2_len2, d2_lang2_id)
            del xx3, llen3, llangs3, xx2_in, llen2_in, llangs2_in

        if d2_x1 is not None:
            # loss2 = self._train_step_bt(xx2_in, llen2_in, llangs2_in, xx3, llen3, llangs3)
            try:
                loss2 = self._train_step_bt(d2_x1, d2_len1, d2_langs1, d2_x2, d2_len2, d2_langs2)
                suffix = 'CT'
                self.stats[('BT%s-%s-%s-%s' % (suffix, lang1, lang2, lang3))].append(loss2.item())
                loss2 = lambda_coeff * loss2
                # optimize
                self.optimize(loss2)
                del loss2
            except RuntimeError as e:
                print('OOM Occur, skip train step loss-2, local_rank: params.local_rank'.format(params.local_rank))

            # del d2_x1, d2_x1, d2_len1, d2_len2, d2_langs1, d2_langs2

        # torch.cuda.empty_cache()

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def bt_step_macd(
            self, lang1, lang2, lang3, lambda_coeff, direction1, enforce_default=False, first_sec=False, wrote=False,
            and_rev=False, with_ae_xz=False, ae_xz_coeff=0, with_mass_xz=False, mass_xz_coeff=0, infer_drop=False
    ):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        mass_xz_coeff = float(mass_xz_coeff)
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        nbest = params.nbest
        assert nbest is None or nbest <= 1
        tokens_per_batch = params.tokens_per_batch
        beam_size = params.beam_size
        mbeam_size = getattr(params, 'mbeam_size', -1)
        select_opt = getattr(params, 'select_opt', 0)
        # assert beam_size > 1
        # assert nbest is not None and 1 <= nbest <= beam_size

        if params.multi_gpu:
            infer_encoder1 = self.encoder1.module
            infer_decoder1 = self.decoder1.module
            infer_encoder2 = self.encoder2.module
            infer_decoder2 = self.decoder2.module
        else:
            infer_encoder1 = self.encoder1
            infer_decoder1 = self.decoder1
            infer_encoder2 = self.encoder2
            infer_decoder2 = self.decoder2

        if infer_drop:
            self.encoder2.train()
            self.decoder2.train()
            self.encoder1.train()
            self.decoder1.train()
            try:
                infer_encoder1.train()
                infer_encoder2.train()
                infer_decoder1.train()
                infer_decoder2.train()
            except Exception as e:
                pass
        else:
            self.encoder2.eval()
            self.decoder2.eval()
            self.encoder1.eval()
            self.decoder1.eval()
            try:
                infer_encoder1.eval()
                infer_encoder2.eval()
                infer_decoder1.eval()
                infer_decoder2.eval()
            except Exception as e:
                pass

        if first_sec:
            first_encoder = infer_encoder2
            first_decoder = infer_decoder2
            sec_encoder = infer_encoder1
            sec_decoder = infer_decoder1
        else:
            first_encoder = infer_encoder1
            first_decoder = infer_decoder1
            sec_encoder = infer_encoder2
            sec_decoder = infer_decoder2

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)
        bsz = x1.size(1)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        if with_ae_xz and ae_xz_coeff > 0:
            nx1, nxlen1 = self.add_noise(x1, len1)
            nxlangs1 = nx1.clone().fill_(lang1_id)
            nx1, nxlen1, nxlangs1 = to_cuda(nx1, nxlen1, nxlangs1)
            loss_ae1 = self._train_step_bt(nx1, nxlen1, nxlangs1, x1, len1, langs1)
            # self.stats[('AE-%s' % lang1)].append(loss_ae1.item())
            loss_ae1 = ae_xz_coeff * loss_ae1
            self.optimize(loss_ae1)
            del nx1, nxlen1, nxlangs1
            del loss_ae1

        if with_mass_xz and mass_xz_coeff > 0:
            self.mass_step_xz(lang1, mass_xz_coeff, x1, len1)

        if direction1:
            d1_lang1_id = lang1_id
            d1_lang2_id = lang2_id
            d2_lang1_id = lang1_id
            d2_lang2_id = lang2_id
        else:
            d1_lang1_id = lang2_id
            d1_lang2_id = lang1_id
            d2_lang1_id = lang2_id
            d2_lang2_id = lang1_id

        # todo: primary training
        with torch.no_grad():
            x2, len2, langs2 = self._generate_pair_mbeamct(
                first_encoder, first_decoder, x1, len1, langs1, lang2_id, mbeam_size=mbeam_size, infer_drop=infer_drop)
            if direction1:
                # lang1->lang2
                d1_x1, d1_len1, d1_langs1 = x1, len1, langs1
                d1_x2, d1_len2, d1_langs2 = x2, len2, langs2
            else:
                # lang2->lang1
                d1_x1, d1_len1, d1_langs1 = x2, len2, langs2
                d1_x2, d1_len2, d1_langs2 = x1, len1, langs1
            d1_x1, d1_len1, d1_langs1, d1_x2, d1_len2, d1_langs2 = self._filter_valid_pairs(
                d1_x1, d1_len1, d1_lang1_id, d1_x2, d1_len2, d1_lang2_id)

        if d1_x1 is not None:
            # try:
            loss1 = self._train_step_bt(d1_x1, d1_len1, d1_langs1, d1_x2, d1_len2, d1_langs2)
            self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss1.item())
            loss1 = lambda_coeff * loss1
            self.optimize(loss1)
            del loss1
            if and_rev:
                loss1_rev = self._train_step_bt(d1_x2, d1_len2, d1_langs2, d1_x1, d1_len1, d1_langs1)
                loss1_rev = lambda_coeff * loss1_rev
                self.optimize(loss1_rev)
                del loss1_rev
            # except RuntimeError as e:
            #     print('OOM Occur, skip train step loss-1, local_rank: params.local_rank'.format(params.local_rank))
        else:
            print('d1_x1 is NOne')

        # todo: secondary training
        with torch.no_grad():
            x3, len3, langs3 = self._generate_pair_mbeamct(
                sec_encoder, sec_decoder, x2, len2, langs2, lang1_id, mbeam_size=mbeam_size, infer_drop=infer_drop)

            if direction1:
                # lang3->lang2
                d2_x1, d2_len1, d2_langs1 = x3, len3, langs3
                d2_x2, d2_len2, d2_langs2 = x2, len2, langs2
            else:
                # lang2->lang3
                d2_x1, d2_len1, d2_langs1 = x2, len2, langs2
                d2_x2, d2_len2, d2_langs2 = x3, len3, langs3
            d2_x1, d2_len1, d2_langs1, d2_x2, d2_len2, d2_langs2 = self._filter_valid_pairs(
                d2_x1, d2_len1, d2_lang1_id, d2_x2, d2_len2, d2_lang2_id)

        if with_ae_xz and ae_xz_coeff > 0:
            nx3, nxlen3 = self.add_noise(x3, len3)
            nxlangs3 = nx3.clone().fill_(lang1_id)
            nx3, nxlen3, nxlangs3 = to_cuda(nx3, nxlen3, nxlangs3)
            loss_ae3 = self._train_step_bt(nx3, nxlen3, nxlangs3, x3, len3, langs3)
            self.stats[('BTAE-%s' % lang1)].append(loss_ae3.item())
            loss_ae3 = ae_xz_coeff * loss_ae3
            self.optimize(loss_ae3)
            del nx3, nxlen3, nxlangs3
            del loss_ae3
        if with_mass_xz and mass_xz_coeff > 0:
            self.mass_step_xz(lang1, mass_xz_coeff, x3, len3)

        if d2_x1 is not None:

            # try:
            loss2 = self._train_step_bt(d2_x1, d2_len1, d2_langs1, d2_x2, d2_len2, d2_langs2)
            suffix = 'CT'
            self.stats[('BT%s-%s-%s-%s' % (suffix, lang1, lang2, lang3))].append(loss2.item())
            loss2 = lambda_coeff * loss2
            self.optimize(loss2)
            del loss2
            if and_rev:
                loss2_rev = self._train_step_bt(d2_x2, d2_len2, d2_langs2, d2_x1, d2_len1, d2_langs1)
                loss2_rev = lambda_coeff * loss2_rev
                self.optimize(loss2_rev)
                del loss2_rev
            # except RuntimeError as e:
            #     print('OOM Occur, skip train step loss-2, local_rank: params.local_rank'.format(params.local_rank))
        else:
            print('d1_x2 is NOne')

        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def _generate_pair_mbeamct(self, enc, dec, x1, len1, langs1, lang2_id, nbest=None, mbeam_size=1, infer_drop=False):
        params = self.params
        # nbest = params.nbest if nbest is None else None
        # beam_size = params.beam_size
        # assert beam_size > 1
        # mbeam_size
        assert nbest is None or 1 <= nbest <= mbeam_size
        if infer_drop:
            self.encoder2.train()
            self.decoder2.train()
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder2.eval()
            self.decoder2.eval()
            self.encoder.eval()
            self.decoder.eval()
        with torch.no_grad():

            # primary
            enc1 = enc('fwd', x=x1, lengths=len1, langs=langs1, causal=False).transpose(0, 1)

            if mbeam_size > 1:
                max_len = int(1.5 * len1.max().item() + 10)
                raise ValueError('Temporary disable mbeam here {}'.format(mbeam_size))
                x2, len2 = dec.generate_beam_efficient(
                    src_enc=enc1, src_len=len1, tgt_lang_id=lang2_id,
                    beam_size=mbeam_size, length_penalty=params.length_penalty, early_stopping=params.early_stopping,
                    max_len=max_len, nbest=nbest
                )
                del enc1
                if nbest is not None:
                    _len2, _nb_, bsz = x2.size()
                    x2 = x2.view(_len2, _nb_ * bsz).detach()
                    len2 = len2.transpose(0, 1).reshape(_nb_ * bsz).detach()
            else:
                x2, len2 = dec.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

        return x2, len2, langs2

    def _generate_pair(self, enc, dec, x1, len1, langs1, lang2_id, nbest=None, mbeam_size=1):
        params = self.params
        # nbest = params.nbest if nbest is None else None
        beam_size = params.beam_size
        assert beam_size > 1
        # mbeam_size
        assert nbest is None or 1 <= nbest <= mbeam_size
        self.encoder2.eval()
        self.decoder2.eval()
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            # primary
            enc1 = enc('fwd', x=x1, lengths=len1, langs=langs1, causal=False).transpose(0, 1)
            max_len = int(1.5 * len1.max().item() + 10)
            x2, len2 = dec.generate_beam_efficient(
                src_enc=enc1, src_len=len1, tgt_lang_id=lang2_id,
                beam_size=beam_size, length_penalty=params.length_penalty, early_stopping=params.early_stopping,
                max_len=max_len, nbest=nbest
            )
            del enc1
            x2, len2 = dec.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            # if nbest is not None:
            #     _len2, _nb_, bsz = x2.size()
            #     x2 = x2.view(_len2, _nb_ * bsz).detach()
            #     len2 = len2.transpose(0, 1).reshape(_nb_ * bsz).detach()
            langs2 = x2.clone().fill_(lang2_id)

        return x2, len2, langs2

    def bt_step_macd_v2(
            self, lang1, lang2, lang3, lambda_coeff, enforce_default=False, first_sec=False, wrote=False):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        nbest = params.nbest
        assert nbest is None or nbest <= 1
        tokens_per_batch = params.tokens_per_batch
        beam_size = params.beam_size
        mbeam_size = getattr(params, 'mbeam_size', -1)
        select_opt = getattr(params, 'select_opt', 0)
        # assert beam_size > 1
        # assert nbest is not None and 1 <= nbest <= beam_size

        if params.multi_gpu:
            infer_encoder1 = self.encoder1.module
            infer_decoder1 = self.decoder1.module
            infer_encoder2 = self.encoder2.module
            infer_decoder2 = self.decoder2.module
        else:
            infer_encoder1 = self.encoder1
            infer_decoder1 = self.decoder1
            infer_encoder2 = self.encoder2
            infer_decoder2 = self.decoder2
        self.encoder2.eval()
        self.decoder2.eval()
        self.encoder1.eval()
        self.decoder1.eval()

        if first_sec:
            first_encoder = infer_encoder2
            first_decoder = infer_decoder2
            sec_encoder = infer_encoder1
            sec_decoder = infer_decoder1
        else:
            first_encoder = infer_encoder1
            first_decoder = infer_decoder1
            sec_encoder = infer_encoder2
            sec_decoder = infer_decoder2

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)
        bsz = x1.size(1)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        pri_srclang_id = lang2_id
        pri_tgtlang_id = lang1_id
        sec_srclang_id = lang2_id
        sec_tgtlang_id = lang1_id

        # todo: primary training
        with torch.no_grad():
            x2, len2, langs2 = self._generate_pair_mbeamct(
                first_encoder, first_decoder, x1, len1, langs1, lang2_id, mbeam_size=mbeam_size)
            # todo: lang2->lang1
            pri_src, pri_srclen, pri_src_langs = x2, len2, langs2
            pri_tgt, pri_tgtlen, pri_tgt_langs = x1, len1, langs1
            pri_src, pri_srclen, pri_src_langs, pri_tgt, pri_tgtlen, pri_tgt_langs = self._filter_valid_pairs(
                pri_src, pri_srclen, pri_srclang_id, pri_tgt, pri_tgtlen, pri_tgtlang_id)

        if pri_src is not None:
            # try:
            loss1 = self._train_step_bt(pri_src, pri_srclen, pri_src_langs, pri_tgt, pri_tgtlen, pri_tgt_langs)
            self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss1.item())
            loss1 = lambda_coeff * loss1
            self.optimize(loss1)
            del loss1
        else:
            print('pri_src is NOne')
            return

        # todo: secondary training
        with torch.no_grad():
            # todo: lang2-lang1
            x3, len3, langs3 = self._generate_pair_mbeamct(
                sec_encoder, sec_decoder, pri_src, pri_srclen, pri_src_langs, lang1_id, mbeam_size=mbeam_size)
            # lang2->lang3
            sec_src, sec_srclen, sec_src_langs = pri_src, pri_srclen, pri_src_langs
            sec_tgt, sec_tgtlen, sec_tgt_langs = x3, len3, langs3
            sec_src, sec_srclen, sec_src_langs, sec_tgt, sec_tgtlen, sec_tgt_langs = self._filter_valid_pairs(
                sec_src, sec_srclen, sec_srclang_id, sec_tgt, sec_tgtlen, sec_tgtlang_id)

        if sec_src is not None:
            # try:
            loss2 = self._train_step_bt(sec_src, sec_srclen, sec_src_langs, sec_tgt, sec_tgtlen, sec_tgt_langs)
            suffix = 'CT'
            self.stats[('BT%s-%s-%s-%s' % (suffix, lang1, lang2, lang3))].append(loss2.item())
            loss2 = lambda_coeff * loss2
            self.optimize(loss2)
            del loss2
        else:
            print('sec_src is NOne')

        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def _select_sec_default(self, x2, len2):
        # x2: [len, nb * bsz], len2: [nb * bsz]
        _len2, _nb, _bsz = x2.size()
        len2_in = len2[0].clone()
        x2_in = x2[:len2_in.max(), 0].clone()
        return x2_in, len2_in

    def _train_step_bt(self, x1, len1, langs1, x2, len2, langs2):
        self.encoder.train()
        self.decoder.train()
        enc = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc = enc.transpose(0, 1)
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        ppred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(ppred_mask[:-1])
        dec3 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc, src_len=len1)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=ppred_mask, y=y, get_scores=False)
        return loss
