# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import to_cuda, restore_segmentation, concat_batches
from ..model.memory import HashingMemory
from ..model.transformer import BeamHypotheses

BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


def eval_memory_usage(scores, name, mem_att, mem_size):
    """
    Evaluate memory usage (HashingMemory / FFN).
    """
    # memory slot scores
    assert mem_size > 0
    mem_scores_w = np.zeros(mem_size, dtype=np.float32)  # weighted scores
    mem_scores_u = np.zeros(mem_size, dtype=np.float32)  # unweighted scores

    # sum each slot usage
    for indices, weights in mem_att:
        np.add.at(mem_scores_w, indices, weights)
        np.add.at(mem_scores_u, indices, 1)

    # compute the KL distance to the uniform distribution
    mem_scores_w = mem_scores_w / mem_scores_w.sum()
    mem_scores_u = mem_scores_u / mem_scores_u.sum()

    # store stats
    scores['%s_mem_used' % name] = float(100 * (mem_scores_w != 0).sum() / len(mem_scores_w))

    scores['%s_mem_kl_w' % name] = float(kl_score(mem_scores_w))
    scores['%s_mem_kl_u' % name] = float(kl_score(mem_scores_u))

    scores['%s_mem_gini_w' % name] = float(gini_score(mem_scores_w))
    scores['%s_mem_gini_u' % name] = float(gini_score(mem_scores_u))

    top50, top90, top99 = tops(mem_scores_w)
    scores['%s_mem_top50_w' % name] = float(top50)
    scores['%s_mem_top90_w' % name] = float(top90)
    scores['%s_mem_top99_w' % name] = float(top99)

    top50, top90, top99 = tops(mem_scores_u)
    scores['%s_mem_top50_u' % name] = float(top50)
    scores['%s_mem_top90_u' % name] = float(top90)
    scores['%s_mem_top99_u' % name] = float(top99)


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.memory_list = trainer.memory_list

        # create directory to store hypotheses, and reference files for BLEU evaluation
        params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
        if self.params.is_master:
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def get_iterator(self, data_set, lang1, lang2=None, stream=False, allow_train_data=False, descending=False):
        """
        Create a new iterator for a dataset.
        """
        assert (data_set in ['valid', 'test']) or allow_train_data
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(self.params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(self.params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1

        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                    descending=descending,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences,
                descending=descending,
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}
        measure_datasets = ['valid', 'test']
        # if params.infer_train:
        #     measure_datasets = ['train'] + measure_datasets

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in measure_datasets:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))
                logger.info("| Load Ref File: {}".format(lang1_path))
                logger.info("| Load Ref File: {}".format(lang2_path))

                if data_set == 'train' and (os.path.exists(lang1_path) and os.path.exists(lang2_path)):
                    print('Skip train dataset creating reference because they exist')
                    continue

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(
                        data_set, lang1, lang2, allow_train_data=params.infer_train):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def evaluate_clm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else "{}-{}".format(lang1, lang2)

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

        # log
        logger.info("Found %i words in %s. %i were predicted correctly." % (n_words, data_set, n_valid))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_clm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else "{}_{}".format(lang1, lang2)

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_mlm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder

    def infer_train(self, trainer):
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            data_set = 'train'
            for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                eval_bleu = params.eval_bleu and params.is_master
                # self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu, infer_train=True)
                self.infer_to_pair_files(scores, data_set, lang1, lang2, eval_bleu, infer_train=True)
        return scores

    def infer_to_pair_files(
            self, scores, data_set, lang1, lang2, eval_bleu, resolve_segmentation=None, infer_train=False):
        params = self.params
        assert (data_set in ['valid', 'test']) or infer_train
        assert lang1 in params.langs
        assert lang2 in params.langs
        descending = getattr(params, 'order_descending', False)
        logger.info('Order descending: {}'.format(descending))
        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []
            source = []
        beam = self.params.beam_size
        lenpen = self.params.length_penalty
        hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
        src_name = 'src{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
        hyp_name = 'infer{0}.{1}-{2}.{3}.b{4}.lp{5}.{6}'.format(
            scores['epoch'], lang1, lang2, data_set, beam, lenpen, lang2)
        src_name = 'infer{0}.{1}-{2}.{3}.b{4}.lp{5}.{6}'.format(
            scores['epoch'], lang1, lang2, data_set, beam, lenpen, lang1)

        hyp_path = os.path.join(params.hyp_path, hyp_name)
        src_path = os.path.join(params.hyp_path, src_name)
        logger.info('Write hypo to: {}'.format(hyp_path))
        logger.info('Write src  to: {}'.format(src_path))
        # ref_path = params.ref_paths[(lang1, lang2, data_set)]

        # export sentences to hypothesis file / restore BPE segmentation
        # with open(hyp_path, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(hypothesis) + '\n')
        # restore_segmentation(hyp_path)

        with open(hyp_path, 'w', encoding='utf-8') as fh:
            with open(src_path, 'w', encoding='utf-8') as fs:

                for index, batch in enumerate(
                        self.get_iterator(data_set, lang1, lang2, allow_train_data=infer_train, descending=descending)):

                    if index % 200 == 0:
                        logger.info('| Generate Index {}'.format(index))

                    # generate batch
                    # (x1, len1), (x2, len2) = batch
                    (x1, len1), (_, __) = batch
                    langs1 = x1.clone().fill_(lang1_id)
                    # langs2 = x2.clone().fill_(lang2_id)

                    # target words to predict
                    # alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
                    # pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
                    # y = x2[1:].masked_select(pred_mask[:-1])
                    # assert len(y) == (len2 - 1).sum().item()

                    # cuda
                    # x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)
                    x1, len1, langs1 = to_cuda(x1, len1, langs1)

                    # encode source sentence
                    enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
                    enc1 = enc1.transpose(0, 1)
                    enc1 = enc1.half() if params.fp16 else enc1

                    # decode target sentence
                    # dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

                    # loss
                    # word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

                    # update stats
                    # n_words += y.size(0)
                    # xe_loss += loss.item() * len(y)
                    # n_valid += (word_scores.max(1)[1] == y).sum().item()
                    if eval_memory:
                        for k, v in self.memory_list:
                            all_mem_att[k].append((v.last_indices, v.last_scores))

                    # generate translation - translate / convert to text
                    # if eval_bleu:
                    max_len = int(1.5 * len1.max().item() + 10)
                    if params.beam_size == 1:
                        generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                    else:
                        generated, lengths = decoder.generate_beam(
                            enc1, len1, lang2_id, beam_size=params.beam_size,
                            length_penalty=params.length_penalty,
                            early_stopping=params.early_stopping,
                            max_len=max_len
                        )
                    hyp_texts = convert_to_text(generated, lengths, self.dico, params)
                    src_texts = convert_to_text(x1, len1, self.dico, params)
                    hypothesis.extend(hyp_texts)
                    source.extend(src_texts)

                    for h, s in zip(hyp_texts, src_texts):
                        fh.write('{}\n'.format(h))
                        fs.write('{}\n'.format(s))

        restore_segmentation(hyp_path, True)
        restore_segmentation(src_path, True)

        logger.info("FINISH GENERATION")

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu, resolve_segmentation=None, infer_train=False):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert (data_set in ['valid', 'test']) or infer_train
        assert lang1 in params.langs
        assert lang2 in params.langs
        descending = getattr(params, 'order_descending', False)
        logger.info('Order descending: {}'.format(descending))
        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for index, batch in enumerate(self.get_iterator(
                data_set, lang1, lang2, allow_train_data=infer_train, descending=descending)):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s-%s_%s' % (data_set, lang1, lang2, mem_name), mem_att, params.mem_size)

        # compute BLEU
        if eval_bleu:

            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu


class BinaryEnsembleDecoder(nn.Module):

    def __init__(self, decoder1, decoder2):
        super().__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.pad_index = self.decoder1.pad_index
        self.eos_index = self.decoder1.eos_index
        self.dim = self.decoder1.dim
        self.n_words = self.decoder1.n_words

    # def forward(self, *input):
        # 'fwd',
        # x = generated[:cur_len],
        # lengths = gen_len,
        # positions = positions[:cur_len],
        # langs = langs[:cur_len],
        # causal = True,
        # src_enc = src_enc,
        # src_len = src_len,
        # cache = cache
    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            # return self.predict(**kwargs)
            raise NotImplementedError('predict not impl')
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc1=None, src_enc2=None, src_len=None, positions=None, langs=None, cache1=None, cache2=None):
        assert src_enc1.size() == src_enc2.size(), '{} != {}'.format(src_enc1.size(), src_enc2.size())
        tensor1 = self.decoder1.fwd(x, lengths, causal, src_enc1, src_len, positions, langs, cache1)
        tensor2 = self.decoder2.fwd(x, lengths, causal, src_enc2, src_len, positions, langs, cache2)
        return tensor1, tensor2

    def generate(self, src_enc1, src_enc2, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc1.size(0) == bs
        assert src_enc2.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # language IDs
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache1 = {'slen': 0}
        cache2 = {'slen': 0}

        while cur_len < max_len:

            # compute word scores
            tensor1, tensor2 = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc1=src_enc1,
                src_enc2=src_enc2,
                src_len=src_len,
                cache1=cache1,
                cache2=cache2
            )
            assert tensor1.size() == (1, bs, self.dim), (cur_len, max_len, src_enc1.size(), tensor1.size(), (1, bs, self.dim))
            assert tensor2.size() == (1, bs, self.dim), (cur_len, max_len, src_enc2.size(), tensor2.size(), (1, bs, self.dim))
            tensor1 = tensor1.data[-1, :, :].type_as(src_enc1)  # (bs, dim)
            tensor2 = tensor2.data[-1, :, :].type_as(src_enc2)  # (bs, dim)
            scores1 = self.decoder1.pred_layer.get_scores(tensor1)      # (bs, n_words)
            scores2 = self.decoder2.pred_layer.get_scores(tensor2)      # (bs, n_words)
            scores = torch.cat((scores1.unsqueeze(-1), scores2.unsqueeze(-1)), -1).mean(-1)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            try:

                generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
            except Exception as e:
                # generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
                generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc1, src_enc2, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        # check inputs
        assert src_enc1.size(0) == src_len.size(0)
        assert src_enc1.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc1 = src_enc1.unsqueeze(1).expand((bs, beam_size) + src_enc1.shape[1:]).contiguous().view(
            (bs * beam_size,) + src_enc1.shape[1:])
        src_enc2 = src_enc2.unsqueeze(1).expand((bs, beam_size) + src_enc2.shape[1:]).contiguous().view(
            (bs * beam_size,) + src_enc2.shape[1:])

        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        langs = positions.clone().fill_(tgt_lang_id)

        # scores for each sentence in the beam
        beam_scores = src_enc1.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = torch.tensor(-1e9).type_as(beam_scores)
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache1 = {'slen': 0}
        cache2 = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor1, tensor2 = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc1=src_enc1,
                src_enc2=src_enc2,
                src_len=src_len,
                cache1=cache1,
                cache2=cache2,
            )
            assert tensor1.size() == (1, bs * beam_size, self.dim)
            assert tensor2.size() == (1, bs * beam_size, self.dim)
            tensor1 = tensor1.data[-1, :, :]  # (bs * beam_size, dim)
            tensor2 = tensor2.data[-1, :, :]  # (bs * beam_size, dim)
            scores1 = self.decoder1.pred_layer.get_scores(tensor1)  # (bs * beam_size, n_words)
            scores2 = self.decoder2.pred_layer.get_scores(tensor2)  # (bs * beam_size, n_words)
            scores = torch.cat((scores1.unsqueeze(-1), scores2.unsqueeze(-1)), -1).mean(-1)

            scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(),
                                                    value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache1.keys():
                if k != 'slen':
                    cache1[k] = (cache1[k][0][beam_idx], cache1[k][1][beam_idx])
            for k in cache2.keys():
                if k != 'slen':
                    cache2[k] = (cache2[k][0][beam_idx], cache2[k][1][beam_idx])


            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len


class EncDecParaPretrainedEvaluator(EncDecEvaluator):

    def __init__(self, trainer, data, params):
        super().__init__(trainer, data, params)
        self.encoder_para = trainer.encoder_para
        self.decoder_para = trainer.decoder_para

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            # for data_set in ['valid', 'test']:
            # fixme: quick fix: run only test
            for data_set in ['test']:

                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt_ensemble_x2(scores, data_set, lang1, lang2, eval_bleu)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def infer_train(self, trainer):
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            data_set = 'train'
            for lang1, lang2, lang3 in set(params.bt_steps):
                logger.info('Perform 2BT: {} -> {} => {}'.format(lang1, lang2, lang3))
                eval_bleu = params.eval_bleu and params.is_master
                # self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu, infer_train=True)
                self.infer_to_triple_files(scores, data_set, lang1, lang2, lang3, eval_bleu, infer_train=True)
        return scores

    def get_iterator(
            self, data_set, lang1, lang2=None, stream=False, allow_train_data=False, descending=False, force_mono=False):
        """
        Create a new iterator for a dataset.
        """
        assert (data_set in ['valid', 'test']) or allow_train_data
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(self.params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(self.params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1

        # FIXME: forcefully get monolingual data despite mt steps

        if lang2 is None or force_mono:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                    descending=descending,
                    infer_train=True
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences,
                descending=descending
            )

        for batch in iterator:
            yield batch if force_mono or lang2 is None or lang1 < lang2 else batch[::-1]

    def infer_to_triple_files(
            self, scores, data_set, lang1, lang2, lang3, eval_bleu, resolve_segmentation=None, infer_train=False):
        params = self.params
        infer_name = params.infer_name
        descending = getattr(params, 'order_descending', False)
        logger.info('Order descending: {}'.format(descending))
        assert infer_name != ''
        assert (data_set in ['valid', 'test']) or infer_train
        assert lang1 in params.langs
        assert lang2 in params.langs
        assert lang3 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        self.encoder_para.eval()
        self.decoder_para.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder
        try:
            encoder_para = self.encoder_para.module if params.multi_gpu else self.encoder_para
            decoder_para = self.decoder_para.module if params.multi_gpu else self.decoder_para
        except Exception as e:
            encoder_para = self.encoder_para
            decoder_para = self.decoder_para

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]

        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}
            all_mem_att2 = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        # if eval_bleu:
        hypothesis2 = []
        hypothesis3 = []
        source = []
        beam = self.params.beam_size
        lenpen = self.params.length_penalty
        # src_name = f"infer{scores['epoch']}.{lang1}-{lang2}-{lang3}.{data_set}.b{beam}.lp{lenpen}.src.{lang1}"
        # hyp2_name = f"infer{scores['epoch']}.{lang1}-{lang2}-{lang3}.{data_set}.b{beam}.lp{lenpen}.hyp.{lang2}"
        # hyp3_name = f"infer{scores['epoch']}.{lang1}-{lang2}-{lang3}.{data_set}.b{beam}.lp{lenpen}.hyp.{lang3}"
        if params.local_rank > -1:
            infer_name = "{}.rank{}".format(infer_name, params.local_rank)
        src_name = "{}.infer{}.{}-{}-{}.{}.b{}.lp{}.src.{}".format(
            infer_name, scores['epoch'], lang1, lang2, lang3, data_set, beam, lenpen, lang1
        )
        hyp2_name = "{}.infer{}.{}-{}-{}.{}.b{}.lp{}.hyp.{}".format(
            infer_name, scores['epoch'], lang1, lang2, lang3, data_set, beam, lenpen, lang2
        )
        hyp3_name = "{}.infer{}.{}-{}-{}.{}.b{}.lp{}.hyp.{}".format(
            infer_name, scores['epoch'], lang1, lang2, lang3, data_set, beam, lenpen, lang3
        )

        src_path = os.path.join(params.hyp_path, src_name)
        hyp2_path = os.path.join(params.hyp_path, hyp2_name)
        hyp3_path = os.path.join(params.hyp_path, hyp3_name)
        logger.info('Write src to  : {}'.format(src_path))
        logger.info('Write hypo2 to: {}'.format(hyp2_path))
        logger.info('Write hypo3 to: {}'.format(hyp3_path))

        with open(hyp2_path, 'w', encoding='utf-8') as fh2:
            with open(hyp3_path, 'w', encoding='utf-8') as fh3:
                with open(src_path, 'w', encoding='utf-8') as fs:

                    for index, batch in enumerate(
                            self.get_iterator(
                                data_set, lang1, lang2, allow_train_data=infer_train, descending=descending,
                                force_mono=True
                            )):

                        if index % 200 == 0:
                            logger.info('| Generate Index {}'.format(index))

                        # todo: 1st stage: lang1 -> lang2 --------------------
                        # (x1, len1), (_, __) = batch
                        x1, len1 = batch
                        # print("x1={}, len1={}".format(x1.size(), len1.size()))
                        langs1 = x1.clone().fill_(lang1_id)
                        x1, len1, langs1 = to_cuda(x1, len1, langs1)
                        # encode source sentence
                        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
                        enc1 = enc1.transpose(0, 1)
                        enc1 = enc1.half() if params.fp16 else enc1

                        if eval_memory:
                            for k, v in self.memory_list:
                                all_mem_att[k].append((v.last_indices, v.last_scores))

                        max_len = int(1.5 * len1.max().item() + 10)
                        if params.beam_size == 1:
                            generated2, lengths2 = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                        else:
                            generated2, lengths2 = decoder.generate_beam(
                                enc1, len1, lang2_id, beam_size=params.beam_size,
                                length_penalty=params.length_penalty,
                                early_stopping=params.early_stopping,
                                max_len=max_len
                            )
                        hyp2_texts = convert_to_text(generated2, lengths2, self.dico, params)
                        src_texts = convert_to_text(x1, len1, self.dico, params)
                        hypothesis2.extend(hyp2_texts)
                        source.extend(src_texts)

                        # todo: 2st stage: lang2 -> lang3 --------------------
                        x2 = generated2
                        len2 = lengths2
                        langs2 = x2.clone().fill_(lang2_id)
                        # encoder
                        enc2 = encoder_para('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
                        enc2 = enc2.transpose(0, 1)
                        enc2 = enc2.half() if params.fp16 else enc2
                        if eval_memory:
                            for k, v in self.memory_list:
                                all_mem_att2[k].append((v.last_indices, v.last_scores))
                        max_len2 = int(1.5 * len2.max().item() + 10)
                        if params.beam_size == 1:
                            generated3, lengths3 = decoder_para.generate(enc2, len2, lang3_id, max_len=max_len2)
                        else:
                            generated3, lengths3 = decoder_para.generate_beam(
                                enc2, len2, lang3_id, beam_size=params.beam_size,
                                length_penalty=params.length_penalty,
                                early_stopping=params.early_stopping,
                                max_len=max_len
                            )
                        hyp3_texts = convert_to_text(generated3, lengths3, self.dico, params)
                        hypothesis3.extend(hyp3_texts)

                        for h2, h3, s in zip(hyp2_texts, hyp3_texts, src_texts):
                            fh2.write('{}\n'.format(h2))
                            fh3.write('{}\n'.format(h3))
                            fs.write('{}\n'.format(s))

        restore_segmentation(hyp2_path, True)
        restore_segmentation(hyp3_path, True)
        restore_segmentation(src_path, True)

        logger.info("FINISH GENERATION")

    def evaluate_mt_ensemble_x2(self, scores, data_set, lang1, lang2, eval_bleu, resolve_segmentation=None, infer_train=False):
        params = self.params
        infer_name = params.infer_name
        descending = getattr(params, 'order_descending', False)
        logger.info('eval_mt_ensemble 2x: Order descending: {}'.format(descending))
        assert infer_name != ''
        assert (data_set in ['valid', 'test']) or infer_train
        assert lang1 in params.langs
        assert lang2 in params.langs
        # assert lang3 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        self.encoder_para.eval()
        self.decoder_para.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder
        try:
            encoder_para = self.encoder_para.module if params.multi_gpu else self.encoder_para
            decoder_para = self.decoder_para.module if params.multi_gpu else self.decoder_para
        except Exception as e:
            encoder_para = self.encoder_para
            decoder_para = self.decoder_para

        decoder_wrapper = BinaryEnsembleDecoder(decoder, decoder_para)

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}
            all_mem_att2 = {k: [] for k, _ in self.memory_list}

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for index, batch in enumerate(self.get_iterator(
                data_set, lang1, lang2, allow_train_data=infer_train, descending=descending)):
            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentences
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1
            enc2 = encoder_para('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc2 = enc2.transpose(0, 1)
            enc2 = enc2.half() if params.fp16 else enc2

            # # decode target sentence
            # dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = decoder_wrapper.generate(enc1, enc2, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder_wrapper.generate_beam(
                        enc1, enc2, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                    # raise NotImplementedError
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu




def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


# def writecsv(_src, _same, _diff, file):
#     with open(file, 'w') as f:
#         f.write(f'index,grade,sentence\n')
#         for i, (sr, sa, di) in enumerate(zip(_src, _same, _diff)):
#             if sa == di:
#                 print(f'same sa==di')
#                 continue
#             f.write(f'{i},x,{sr}\n')
#             f.write(f'{i},,{sa}\n')
#             f.write(f'{i},,{di}\n')

"""
compare.src.same.diff.1m5.100.en.check.csv
>>> diff = [l for i, l in enumerate(lines) if i % 3 == 2]
>>> sa = [l.split(',')[1] for l in same if l.split(',')[1] == '1']
>>> len(sa)
31
>>> di = [l.split(',')[1] for l in diff if l.split(',')[1] == '1']
>>> len(di)
41

random check:
diff same
37  34
"""




