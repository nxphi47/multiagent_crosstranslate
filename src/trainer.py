# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

logger = getLogger()


class Trainer(object):
    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # list memory components
        self.memory_list = []
        self.ffn_list = []
        for name in self.MODEL_NAMES:
            find_modules(getattr(self, name), 'self.{}'.format(name), HashingMemory, self.memory_list)
            find_modules(getattr(self, name), 'self.{}'.format(name), TransformerFFN, self.ffn_list)
        logger.info("Found %i memories." % len(self.memory_list))
        logger.info("Found %i FFN." % len(self.ffn_list))

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for name in self.MODEL_NAMES:
                setattr(self, name,
                        nn.parallel.DistributedDataParallel(getattr(self, name), device_ids=[params.local_rank],
                                                            output_device=params.local_rank, broadcast_buffers=True))

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for name in self.MODEL_NAMES:
                    setattr(self, name,
                            apex.parallel.DistributedDataParallel(getattr(self, name), delay_allreduce=True))

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # probability of masking out / randomize / not modify words to predict
        params.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

        # probabilty to predict a word
        counts = np.array(list(self.data['dico'].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -params.sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index
        params.mask_scores[counts == 0] = 0  # do not predict special tokens

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('CLM-%s' % l, []) for l in params.langs] +
            [('CLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            [('CLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('MLM-%s' % l, []) for l in params.langs] +
            [('MLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            [('MLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('EDMLM-%s' % l, []) for l in params.langs] +
            [('EDMLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            [('EDMLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('AEmbeam-%s' % l, []) for l in params.langs] +
            [('AEmbeam-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            [('AEmbeam-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
            [('mbeamAE-%s' % lang, []) for lang in params.ae_steps] +
            [('AE-%s' % lang, []) for lang in params.ae_steps] +
            [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
            [('2TBT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
            [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
            [('BTCT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
            [('BTMA-%s' % l1, []) for l1, l2, l3 in params.bt_steps] +
            [('BTAE-%s' % l1, []) for l1, l2, l3 in params.bt_steps] +
            # [('BTMA-%s' % lang, []) for lang in params.mass_steps] +
            [('BTSEC-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = []
        for name in self.MODEL_NAMES:
            named_params.extend([(k, p) for k, p in getattr(self, name).named_parameters() if p.requires_grad])

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params if not k.endswith(HashingMemory.MEM_VALUES_PARAMS)]

        # memory values
        if params.use_memory:
            self.parameters['memory'] = [p for k, p in named_params if k.endswith(HashingMemory.MEM_VALUES_PARAMS)]
            assert len(self.parameters['memory']) == len(params.mem_enc_positions) + len(params.mem_dec_positions)

        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # memory values optimizer
        if params.use_memory:
            self.optimizers['memory'] = get_optimizer(self.parameters['memory'], params.mem_values_optimizer)

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        opt_names = self.optimizers.keys()
        models = [getattr(self, name) for name in self.MODEL_NAMES]
        models, optimizers = apex.amp.initialize(
            models,
            [self.optimizers[k] for k in opt_names],
            opt_level=('O%i' % params.amp)
        )
        for name, model in zip(self.MODEL_NAMES, models):
            setattr(self, name, model)
        self.optimizers = {
            opt_name: optimizer
            for opt_name, optimizer in zip(opt_names, optimizers)
        }

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    # print(name, norm_check_a, norm_check_b)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                try:
                    with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                        scaled_loss.backward()
                except Exception as e:
                    for opt in optimizers:
                        with apex.amp.scale_loss(loss, opt) as scaled_loss:
                            scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        # print(name, norm_check_a, norm_check_b)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        if self.n_total_iter % self.params.log_per_iter == 0:
            logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name, lang1, lang2, stream, descending=False):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join(
            [str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        assert stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:
                iterator = self.data['mono'][lang1]['train'].get_iterator(
                    shuffle=True,
                    group_by_size=self.params.group_by_size,
                    n_sentences=-1,
                    descending=descending,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
                descending=descending,
            )

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def word_shuffle(self, x, lengths):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, lengths

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol
        # l_cpu = l.cpu()
        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(lengths.size(0)):
            # generate a random permutation
            # try:
                # print(lengths[i] - 1)
                # arange = np.arange(lengths[i].item() - 1)
                # noise_ = noise[:lengths[i] - 1, i]
                # print(noise_)
                # print(arange)
                # scores = arange + noise_
            scores = np.arange(lengths[i].item() - 1) + noise[:lengths[i] - 1, i]
            # except Exception as e:
            #     print('ERROR: ')
            #     raise e
            permutation = scores.argsort()
            # shuffle words
            x2[:lengths[i] - 1, i].copy_(x2[:lengths[i] - 1, i][torch.from_numpy(permutation)])
        return x2, lengths

    def word_dropout(self, x, lengths):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, lengths
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == lengths.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        bs = lengths.size(0)
        sentences = []
        lengths_ = []
        for i in range(bs):
            assert x[lengths[i] - 1, i] == eos
            words = x[:lengths[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]

            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1 and len(words) > 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert (len(new_s) >= 3 or len(words) == 1) and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths_.append(len(new_s))

        # re-construct input
        l2 = torch.LongTensor(lengths_)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, lengths):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, lengths
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == lengths.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(lengths.size(0)):
            assert x[lengths[i] - 1, i] == eos
            words = x[:lengths[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == lengths[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(self.params.pad_index)
        for i in range(lengths.size(0)):
            x2[:lengths[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, lengths

    def word_shuffle_dropout_blank(self, x, lengths):
        if self.params.word_shuffle == 0 and self.params.word_dropout == 0 and self.params.word_blank == 0:
            return x, lengths

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8)).bool()
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8).bool()
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:
            (x1, len1) = self.get_batch(name, lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

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
            logger.warning("Saving {} parameters ...".format(name))
            data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning("Saving {} optimizer ...".format(name))
                data['{}_optimizer'.format(name)] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
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
            if "_clone" in name:
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
        for name in self.optimizers.keys():
            if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
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

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch, include_optimizers=False)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric, include_optimizers=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
                self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint', include_optimizers=True)
        self.epoch += 1

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def clm_step(self, lang1, lang2, lambda_coeff):
        """
        Next word prediction step (causal prediction).
        CLM objective.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'decoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'causal')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        if params.context_size > 0:  # do not predict without context
            pred_mask[:params.context_size] = 0
        y = x[1:].masked_select(pred_mask[:-1])
        assert pred_mask.sum().item() == y.size(0)

        # cuda
        x, lengths, langs, pred_mask, y = to_cuda(x, lengths, langs, pred_mask, y)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, langs=langs, causal=True)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('CLM-%s' % lang1) if lang2 is None else ('CLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()

    def mlm_step(self, lang1, lang2, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'pred')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()

    def pc_step(self, lang1, lang2, lambda_coeff):
        """
        Parallel classification step. Predict if pairs of sentences are mutual translations of each other.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # sample parallel sentences
        (x1, len1), (x2, len2) = self.get_batch('align', lang1, lang2)
        bs = len1.size(0)
        if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
            self.n_sentences += params.batch_size
            return

        # associate lang1 sentences with their translations, and random lang2 sentences
        y = torch.LongTensor(bs).random_(2)
        idx_pos = torch.arange(bs)
        idx_neg = ((idx_pos + torch.LongTensor(bs).random_(1, bs)) % bs)
        idx = (y == 1).long() * idx_pos + (y == 0).long() * idx_neg
        x2, len2 = x2[:, idx], len2[idx]

        # generate batch / cuda
        x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                      params.eos_index, reset_positions=False)
        x, lengths, positions, langs, new_idx = self.round_batch(x, lengths, positions, langs)
        if new_idx is not None:
            y = y[new_idx]
        x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

        # get sentence embeddings
        h = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)[0]

        # parallel classification loss
        CLF_ID1, CLF_ID2 = 8, 9  # very hacky, use embeddings to make weights for the classifier
        emb = (model.module if params.multi_gpu else model).embeddings.weight
        pred = F.linear(h, emb[CLF_ID1].unsqueeze(0), emb[CLF_ID2, 0])
        loss = F.binary_cross_entropy_with_logits(pred.view(-1), y.to(pred.device).type_as(pred))
        self.stats['PC-%s-%s' % (lang1, lang2)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += lengths.sum().item()


class SingleTrainer(Trainer):

    def __init__(self, model, data, params):
        self.MODEL_NAMES = ['model']

        # model / data / params
        self.model = model
        self.data = data
        self.params = params

        super().__init__(data, params)


class EncDecTrainer(Trainer):

    def __init__(self, encoder, decoder, data, params):

        self.MODEL_NAMES = ['encoder', 'decoder']

        # model / data / params
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.params = params

        if getattr(params, 'bt_sync', -1) > 1:
            # self.encoder_clone, self.decoder_clone = build_model(params, data['dico'])
            self.encoder_clone, self.decoder_clone = copy.deepcopy(self.encoder), copy.deepcopy(self.decoder)
            self.MODEL_NAMES = ['encoder', 'decoder', 'encoder_clone', 'decoder_clone']
        else:
            self.encoder_clone, self.decoder_clone = None, None
        super().__init__(data, params)
        self.generate_mbeam_func = None

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

    def update_syn_model(self):
        logger.info('Update Cloned Model (iter={})'.format(self.n_iter))
        try:
            self.encoder_clone.load_state_dict(copy.deepcopy(self.encoder.state_dict()))
        except Exception as e:
            self.encoder_clone.module.load_state_dict(copy.deepcopy(self.encoder.module.state_dict()))
        try:
            self.decoder_clone.load_state_dict(copy.deepcopy(self.decoder.state_dict()))
        except Exception as e:
            self.decoder_clone.module.load_state_dict(copy.deepcopy(self.decoder.module.state_dict()))


    def mt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        if lang1 == lang2:
            (x1, len1) = self.get_batch('ae', lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            (x1, len1), (x2, len2) = self.get_batch('mt', lang1, lang2)
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
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

        # loss
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('AE-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def bt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        sampling_topp = getattr(params, 'sampling_topp', -1)
        sampling_topk = getattr(params, 'sampling_topk', -1)
        topk_ample_temperature = getattr(params, 'topk_ample_temperature', None)

        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate source batch
        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # generate a translation
        with torch.no_grad():
            # evaluation mode
            self.encoder.eval()
            self.decoder.eval()

            # encode source sentence and translate it
            enc1 = _encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            if sampling_topp > 0:
                x2, len2 = _decoder.generate_nucleus_sampling(
                    enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5), sampling_topp=sampling_topp)
            elif sampling_topk > 0:
                x2, len2 = _decoder.generate_topk_sampling(
                    enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5), sampling_topk=sampling_topk,
                    sample_temperature=topk_ample_temperature
                )
            else:
                x2, len2 = _decoder.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

            # training mode
            self.encoder.train()
            self.decoder.train()

        # encode generate sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=len2)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def bt_sync_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        sampling_topp = getattr(params, 'sampling_topp', -1)
        sampling_topk = getattr(params, 'sampling_topk', -1)
        topk_ample_temperature = getattr(params, 'topk_ample_temperature', None)
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder
        assert self.encoder_clone is not None
        assert self.decoder_clone is not None
        try:
            _encoder_clone = self.encoder_clone.module if params.multi_gpu else self.encoder_clone
        except AttributeError as e:
            _encoder_clone = self.encoder_clone
        try:
            _decoder_clone = self.decoder_clone.module if params.multi_gpu else self.decoder_clone
        except AttributeError as e:
            _decoder_clone = self.decoder_clone

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate source batch
        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # generate a translation
        with torch.no_grad():

            # evaluation mode
            self.encoder_clone.eval()
            self.decoder_clone.eval()

            # encode source sentence and translate it
            enc1 = _encoder_clone('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            if sampling_topp > 0:
                x2, len2 = _decoder_clone.generate_nucleus_sampling(
                    enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5), sampling_topp=sampling_topp)
            elif sampling_topk > 0:
                x2, len2 = _decoder.generate_topk_sampling(
                    enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5), sampling_topk=sampling_topk,
                    sample_temperature=topk_ample_temperature
                )
            else:
                x2, len2 = _decoder_clone.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

        # encode generate sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=len2)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()


class EncDecParaPretrainedTrainer(EncDecTrainer):

    # def __init__(self, encoder, decoder, encoder_para, decoder_para, data, params):
    #     self.encoder_para = encoder_para
    #     self.decoder_para = decoder_para
    #     super().__init__(encoder, decoder, data, params)

    def __init__(self, encoder, decoder, encoder_para, decoder_para, data, params):

        self.MODEL_NAMES = ['encoder', 'decoder', 'encoder_para', 'decoder_para']

        # model / data / params
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_para = encoder_para
        self.decoder_para = decoder_para
        self.data = data
        self.params = params

        if getattr(params, 'bt_sync', -1) > 1:
            # self.encoder_clone, self.decoder_clone = build_model(params, data['dico'])
            self.encoder_clone, self.decoder_clone = copy.deepcopy(self.encoder), copy.deepcopy(self.decoder)
            self.MODEL_NAMES = ['encoder', 'decoder', 'encoder_clone', 'decoder_clone']
        else:
            self.encoder_clone, self.decoder_clone = None, None
        super(EncDecTrainer, self).__init__(data, params)

    def para_bt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
                Back-translation step for machine translation.
                """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder
        self.encoder.train()
        self.decoder.train()

        self.encoder_para.eval()
        self.decoder_para.eval()
        _encoder_para = self.encoder_para.module if params.multi_gpu else self.encoder_para
        _decoder_para = self.decoder_para.module if params.multi_gpu else self.decoder_para

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate source batch
        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # generate a translation
        with torch.no_grad():
            # evaluation mode
            # self.encoder.eval()
            # self.decoder.eval()
            # self.encoder_para.eval()
            # self.decoder_para.eval()

            # encode source sentence and translate it
            enc1 = _encoder_para('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            x2, len2 = _decoder_para.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

            # training mode
            # self.encoder.train()
            # self.decoder.train()

        # encode generate sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=len2)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        self.stats[('2TBT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def _dual_prob(self, encoders, decoders, x1, len1, langs1, x2, len2, langs2):
        # alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        # pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word

        # y = x2[1:].masked_select(pred_mask[:-1])
        # assert len(y) == (len2 - 1).sum().item()

        encs1 = [enc('fwd', x=x1, lengths=len1, langs=langs1, causal=False) for enc in encoders]
        encs1 = [enc.transpose(0, 1) for enc in encs1]

        decs2 = [dec('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)
                 for enc1, dec in zip(encs1, decoders)]

        # desc [len, b, dim)
        # scores_ = [F.softmax(dec.pred_layer.proj(mt[:-1]), -1) for dec, mt in zip(decs2, decoders)]
        scores_ = [dec('predict', tensor=mt[:-1], pred_mask=None, y=None, get_scores=False, softmax=True)
                   for mt, dec in zip(decs2, decoders)]
        probs = [sc.gather(-1, x2[1:].unsqueeze(-1)) for sc in scores_]
        # probs [len, b, 1]
        prob = torch.cat(probs, -1).mean(-1)

        # masked_t = [tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim) for tensor in decs2]
        #
        # scores_ = [F.softmax(dec.proj(mt).view(-1, dec.n_words), -1) for dec, mt in zip(masked_t, decoders)]
        # probs = [sc.gather(-1, y.unsqueeze(-1)).unsqueeze(-1) for sc in scores_]
        # prob = torch.cat(probs, -1).mean(-1)


        return prob

    def _dual_loss(self, encoders, decoders, x1, len1, langs1, x2, len2, langs2):
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        prob = self._dual_prob(encoders, decoders, x1, len1, langs1, x2, len2, langs2)
        loss = -prob.log().masked_fill_(pred_mask[:-1], 0)

        # masked_t = [tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim) for tensor in decs2]
        # scores_ = [F.softmax(dec.proj(mt).view(-1, dec.n_words), -1) for dec, mt in zip(masked_t, decoders)]
        # probs = [sc.gather(-1, y.unsqueeze(-1)).unsqueeze(-1) for sc in scores_]
        # prob = torch.cat(probs, -1).mean(-1)
        # projs = [dec.proj(mt).view(-1, dec.n_words) for dec, mt in zip(masked_t, decoders)]
        # log(1/2(p1 + p2))
        # log(1/2(exp(h1)/sumexp(h1) + exp(h2)/sumexp(h2)))
        return loss

    def _dual_single_prob(self, encoder, decoder, x1, len1, langs1, x2, len2, langs2):
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word

        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)
        dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

        # desc [len, b, dim)
        scores_ = decoder('predict', tensor=dec2[:-1], pred_mask=None, y=None, get_scores=False, softmax=True)
        probs = scores_.gather(-1, x2[1:].unsqueeze(-1)).masked_fill_(~pred_mask[:-1].unsqueeze(-1), 1)
        # probs = probs.prod(0, keepdim=True)
        # probs [1, b, 1]
        return probs

    def para_bt_dual_step(self, lang1, lang2, lang3, lambda_coeff):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder
        self.encoder.train()
        self.decoder.train()

        self.encoder_para.eval()
        self.decoder_para.eval()
        _encoder_para = self.encoder_para.module if params.multi_gpu else self.encoder_para
        _decoder_para = self.decoder_para.module if params.multi_gpu else self.decoder_para

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # x2, len2 = self.get_batch('bt', lang2)
        # langs2 = x2.clone().fill_(lang2_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)
        # x2, len2, langs2 = to_cuda(x2, len2, langs2)

        # encoders = [_encoder, _encoder_para]
        # decoders = [_decoder, _decoder_para]
        encoders = [self.encoder, self.encoder_para]
        decoders = [self.decoder, self.decoder_para]

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            self.encoder_para.eval()
            self.decoder_para.eval()

            enc1 = _encoder_para('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            y1, y_len1 = _decoder_para.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            y_langs1 = y1.clone().fill_(lang2_id)

            # enc2 = _encoder_para('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
            # enc2 = enc2.transpose(0, 1)
            # y2, y_len2 = _decoder_para.generate(enc2, len2, lang1_id, max_len=int(1.3 * len2.max().item() + 5))
            # y_langs2 = y2.clone().fill_(lang1_id)

            # free CUDA memory
            del enc1
            # del enc2

            # sigma_p = self._dual_prob(encoders, decoders, x1, len1, langs1, y1, y_len1, y_langs1)

            p_m = self._dual_single_prob(self.encoder, self.decoder, x1, len1, langs1, y1, y_len1, y_langs1)
            p_mp = self._dual_single_prob(self.encoder_para, self.decoder_para, x1, len1, langs1, y1, y_len1, y_langs1)
            # [len, b, 1]
            p_mmp = torch.cat([p_m, p_mp], -1).mean(-1).detach()
            p_mp_ = p_mp.squeeze(-1).detach()
            sigma_p = (p_mmp / p_mp_).prod(0, keepdim=True).detach()

            o_mp = self._dual_single_prob(self.encoder_para, self.decoder_para, y1, y_len1, y_langs1, x1, len1, langs1)
            o_mp = o_mp.detach()

        # sigma(x, y') = (p(y'|x; [m, mp]) / p(y'|x; [mp]) log p(x|y'; [m, mp])
        # phi  (x', y) = (p(x'|y; [m, mp]) / p(x'|y; [mp]) log p(y|x'; [m, mp])

        self.encoder.train()
        self.decoder.train()
        # self.encoder_para.eval()
        # self.decoder_para.eval()

        o_m = self._dual_single_prob(self.encoder, self.decoder, y1, y_len1, y_langs1, x1, len1, langs1)

        loss = -torch.cat([o_m, o_mp], -1).mean(-1).log()

        # loss = self._dual_loss(encoders, decoders, y1, y_len1, y_langs1, x1, len1, langs1)

        # logger.info('loss sigma_p: {} {}'.format(loss.size(), sigma_p.size()))
        loss = loss * sigma_p
        loss = loss.sum() / len1.sum()

        self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()




def convert_to_text(batch, lengths, dico, params, nbest=None):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert nbest is not None or (lengths.max() == slen and lengths.shape[0] == bs)
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            try:
                words.append(dico[batch[k, j]])
            except Exception as e:
                logger.info('dico size: {}'.format(dico))
                raise e
        sentences.append(" ".join(words))
    return sentences




