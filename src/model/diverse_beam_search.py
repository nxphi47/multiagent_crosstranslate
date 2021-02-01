import torch
from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .beam_search import *


def generate_diverse_beam(
        self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping,
        num_groups, diversity_strength, max_len=200, nbest=None, sample_temperature=None
):
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

    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1
    assert beam_size % num_groups == 0

    # batch size / number of words
    bs = len(src_len)
    n_words = self.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
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
    beam_scores = src_enc.new(bs, beam_size).fill_(0)
    beam_scores[:, 1:] = torch.tensor(-1e9).type_as(beam_scores)
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 1

    # cache compute states
    cache = {'slen': 0}

    # done sentences
    done = [False for _ in range(bs)]

    while cur_len < max_len:

        # compute word scores
        tensor = self.forward(
            'fwd',
            x=generated[:cur_len],
            lengths=src_len.new(bs * beam_size).fill_(cur_len),
            positions=positions[:cur_len],
            langs=langs[:cur_len],
            causal=True,
            src_enc=src_enc,
            src_len=src_len,
            cache=cache
        )
        assert tensor.size() == (1, bs * beam_size, self.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)

        scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
        assert scores.size() == (bs * beam_size, n_words)

        assert sample_temperature is None or sample_temperature == 1.0, 'sample_temperature={} not support'.format(
            sample_temperature)

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
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    return compute_final_decoded(generated_hyps, bs, src_len, self.pad_index, self.eos_index, beam_size, nbest)
