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

# from .evaluation.online_bleu import SentBLEUV2
# from . import noising

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
        # self.onl_bleu_scorer = SentBLEUV2(self.params.pad_index, self.params.eos_index, self.params.unk_index)
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

    def _filter_valid_pairs(self, x1, len1, lang1_id, x2, len2, lang2_id):
        params = self.params
        max_len = params.max_len
        # filter_bleu = getattr(params, 'filter_bleu', 0)
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

    def bt_step_macd(
            self, lang1, lang2, lang3, lambda_coeff, direction1, enforce_default=False, first_sec=False, wrote=False,
            and_rev=False, with_ae_xz=False, ae_xz_coeff=0, with_mass_xz=False, mass_xz_coeff=0, infer_drop=True
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
            x2, len2, langs2 = self._generate_pair(
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
        else:
            print('d1_x1 is NOne')

        # todo: secondary training
        with torch.no_grad():
            x3, len3, langs3 = self._generate_pair(
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

        if d2_x1 is not None:
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
        else:
            print('d1_x2 is NOne')

        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def _generate_pair(self, enc, dec, x1, len1, langs1, lang2_id, nbest=None, mbeam_size=1, infer_drop=False):
        params = self.params
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
            x2, len2 = dec.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
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
            x2, len2, langs2 = self._generate_pair(
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
            x3, len3, langs3 = self._generate_pair(
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
