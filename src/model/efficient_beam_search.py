"""
    *** CREDIT TO PHI for this!: nxphi47@gmail.com / nxphi47@github.com ****
"""

import torch
from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .beam_search import BeamHypotheses, compute_final_decoded

logger = getLogger()
# import  nltk.translate.bleu_score as bleu


def beam_efficient_final_decoded(
        hyps_t, bs, src_len, pad_index, eos_index, beam_size, nbest=None):
    """
        *** CREDIT TO PHI for this!: nxphi47@gmail.com / nxphi47@github.com ****
    """
    if nbest is None or nbest <= 0:

        hyp_best = hyps_t['hyps'][:, :, 0]
        output = hyp_best.clone()
        # output = hyp_best

        max_len, _bs = hyp_best.size()
        device = hyp_best.device
        assert _bs == bs
        assert (hyp_best[-1] == pad_index).all()
        assert (hyp_best[0] == eos_index).all()
        hyp_len = ((hyp_best != pad_index) * (hyp_best != eos_index)).int().sum(0) + 2
        # hyp_len = ((hyp_best != pad_index) * (hyp_best != eos_index)).int().cumsum(0).max(0)[0] + 2

        # FIXME: fix this with out the loop!
        eos_mask = torch.arange(max_len, device=device).unsqueeze_(-1).expand_as(output) == (hyp_len - 1).unsqueeze_(0)
        # for i in range(bs):
        #     output[hyp_len[i] - 1, i] = eos_index
        output = output.masked_fill_(eos_mask, eos_index)

        output = output[:hyp_len.max()]
        # assert (output == eos_index).sum() == 2 * bs
        sanity_check = (output == eos_index).sum() == 2 * bs
        if not sanity_check:
            for i in range(bs):
                if (output[:, i] == eos_index).sum() != 2:
                    out = output[:, i]
                    hyp = hyp_best[:, i]
                    out_l = ((out != pad_index) * (out != eos_index)).int().sum() + 2
                    hyp_l = hyp_len[i]
                    logger.info('[b={}][o={}/h={}] out=\n{}\nhyp\n{}'.format(
                        i, out_l, hyp_l, out, hyp
                    ))
            raise ValueError('sanity check fails')
        # assert sanity_check
        return output, hyp_len
    else:
        assert isinstance(nbest, int) and 1 <= nbest <= beam_size
        # [max_len, nbest, bs]
        hyp_best = hyps_t['hyps'][:, :, :nbest]
        device = hyp_best.device

        output = hyp_best.clone()
        max_len, _bs, _nb = output.size()
        assert _bs == bs
        assert _nb == nbest
        assert (output[-1] == pad_index).all()
        assert (output[0] == eos_index).all()
        hyp_len = ((output != pad_index) * (output != eos_index)).int().sum(0) + 2
        # hyp_len: [bs, nb]

        # for i in range(bs):
        #     for j in range(nbest):
        #         output[hyp_len[i, j] - 1, i, j] = eos_index
        #         output[hyp_len[i, j]:, i, j] = pad_index

        eos_mask = torch.arange(max_len, device=device).unsqueeze_(-1).unsqueeze_(
            -1).expand_as(output) == (hyp_len - 1).unsqueeze_(0)
        output = output.masked_fill_(eos_mask, eos_index)

        output = output[:hyp_len.max()]
        # sanity check
        output = output.transpose(1, 2).contiguous()
        # FIXME: change it to nbest: [max_len, nbest, bs]

        sanity_check = (output == eos_index).sum() == 2 * bs * nbest
        # FIXME: problem is in more nbest!
        if not sanity_check:
            for i in range(bs):
                for j in range(nbest):
                    if (output[:, i, j] == eos_index).sum() != 2:
                        out = output[:, i, j]
                        hyp = hyp_best[:, i, j]
                        out_l = ((out != pad_index) * (out != eos_index)).int().sum() + 2
                        hyp_l = hyp_len[i, j]
                        logger.info('[b={},n={}][o={}/h={}] out=\n{}\nhyp\n{}'.format(
                            i, j, out_l, hyp_l, out, hyp
                        ))
            raise ValueError('sanity check fails')
        # assert sanity_check

        return output, hyp_len


def generate_beam_gpu(
        model, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping,
        max_len=200, nbest=None,
        **kwargs
):
    """
        ** Any further upgrade to beam search will based on this default version
    :param model:
    :param src_enc:
    :param src_len:
    :param tgt_lang_id:
    :param beam_size:
    :param length_penalty:
    :param early_stopping:
    :param max_len:
    :param nbest:
    :param sample_temperature:
    :param sample_replacement:
    :param kwargs:
    :return:
    """
    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1
    pad_index = model.pad_index

    # batch size / number of words
    bs = len(src_len)
    n_words = model.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(model.eos_index)  # we use <EOS> for <BOS> everywhere

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
    # generated hypotheses
    device = src_enc.device

    done_t = torch.zeros(bs, device=device).bool()
    # hyps_size = beam_size
    # hyps_size = hyp_size_multiple * beam_size
    hyps_size = beam_size
    hyps_t = {
        "len": torch.zeros(bs, device=device).long(),
        "hyps": generated.new(max_len, bs, hyps_size).fill_(pad_index),
        "worst": src_enc.new(bs).float().fill_(1e9),
        "score": src_enc.new(bs, hyps_size).float().fill_(-1e9),
        "hyp_len": torch.zeros(bs, hyps_size, device=device).long(),
    }
    # testing comapare
    while cur_len < max_len:

        # compute word scores
        tensor = model.forward(
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
        assert tensor.size() == (1, bs * beam_size, model.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = model.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)

        scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
        assert scores.size() == (bs * beam_size, n_words)
        # select next words with scores
        _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
        _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)
        next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        # todo: get next words loop tensors, replacing for sent_id in range(bs) loop
        worst_better = (hyps_t['worst'] >= (next_scores.max(dim=-1)[0].detach() / ((max_len - 1) ** length_penalty)))
        len_less_beam = hyps_t['len'] < beam_size
        new_is_done = worst_better.masked_fill_(len_less_beam, torch.tensor(0).bool()).bool()
        done_t = done_t | new_is_done
        not_done = ~done_t
        # todo: next_words for sentences
        beam_ids_t = next_words // n_words
        word_ids_t = next_words % n_words
        sent_beam_idx_t = beam_ids_t + torch.arange(bs, device=device).unsqueeze_(-1) * beam_size

        is_eos = (word_ids_t == model.eos_index) | word_ids_t.new(bs, 2 * beam_size).fill_(
            int(cur_len + 1 == max_len)).bool()
        # next_sent_beam = [[] for i in range(bs)]
        n_sent_b_t_val = src_enc.new(bs, 2 * beam_size).float().fill_(-1e9)
        n_sent_b_t_widx = generated.new(bs, 2 * beam_size).fill_(pad_index)
        n_sent_b_t_pidx = generated.new(bs, 2 * beam_size).fill_(0)
        # todo: future work: how to GPU this loop!
        for i in range(2 * beam_size):
            value = next_scores[:, i]
            is_step_eos = is_eos[:, i]
            beam_ids = beam_ids_t[:, i]
            word_ids = word_ids_t[:, i]
            sent_beam_idx = sent_beam_idx_t[:, i]
            any_step_eos = torch.any(is_step_eos)

            add_supposed_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
            batch_bucket_not_full = (n_sent_b_t_widx != pad_index).int().sum(-1) < beam_size
            allow_to_add = not_done & batch_bucket_not_full

            n_bucket_full = (n_sent_b_t_widx[add_supposed_sent_idx] != pad_index).int().sum(-1) >= beam_size

            if any_step_eos:
                bscore = value.detach() / (cur_len ** length_penalty)  # FIXME: check cur_len=len(hyp) in Hypo.add()
                len_less_beamsize = hyps_t['len'] < beam_size
                bscore_more_worst = bscore > hyps_t['worst']

                lenlessbeam_bscoremoreworst = len_less_beamsize | bscore_more_worst
                # add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & not_done
                add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & allow_to_add
                add_batch_idx = torch.arange(bs, device=device).masked_select(add_or_not).long()
                if add_batch_idx.numel() > 0:

                    hyp = generated[:cur_len].view(cur_len, bs, beam_size)[:, add_batch_idx].clone()
                    hyp = hyp.gather(2, beam_ids[add_batch_idx].unsqueeze(0).unsqueeze(-1).expand(
                        cur_len, add_batch_idx.size(0), 1))
                    hyp = hyp.squeeze(-1).detach()
                    bscore_batch = bscore[add_batch_idx].detach()

                    hyps_t['hyps'][:cur_len, add_batch_idx, -1] = hyp
                    hyps_t['score'][add_batch_idx, -1] = bscore_batch

                    hyps_t['len'][add_batch_idx] += 1

                    # resort hyps and score
                    # next_hyp_score, next_hyp_idx = torch.topk(
                    #     hyps_t['score'][add_batch_idx], hyps_size, dim=1, largest=True, sorted=True)
                    next_hyp_score, next_hyp_idx = torch.sort(
                        hyps_t['score'][add_batch_idx], dim=1, descending=True)

                    hyps_t['score'][add_batch_idx] = next_hyp_score
                    hyps_t['hyps'][:cur_len, add_batch_idx] = hyps_t['hyps'][:cur_len, add_batch_idx].gather(
                        2, next_hyp_idx.unsqueeze(0).expand_as(hyps_t['hyps'][:cur_len, add_batch_idx]))

                    # assigning worst
                    min_worst_bscore = torch.min(hyps_t['worst'][add_batch_idx], bscore_batch)
                    min_worst_sort = next_hyp_score[:, -1]

                    more_than_beam = (hyps_t['len'][add_batch_idx] > beam_size)
                    add_batch_more_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        more_than_beam).long()
                    add_batch_less_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        ~more_than_beam).long()
                    assert add_batch_more_idx.size(0) + add_batch_less_idx.size(0) == add_batch_idx.size(0)
                    worst_batch_idx = hyps_t['worst'][add_batch_idx].clone()

                    if add_batch_more_idx.numel() > 0:
                        worst_batch_idx[add_batch_more_idx] = min_worst_sort[add_batch_more_idx]
                    if add_batch_less_idx.numel() > 0:
                        worst_batch_idx[add_batch_less_idx] = min_worst_bscore[add_batch_less_idx]

                    hyps_t['worst'][add_batch_idx] = worst_batch_idx
                    hyps_t['len'] = torch.min(hyps_t['len'], hyps_t['len'].new([beam_size]).expand_as(hyps_t['len']))

                    assert (hyps_t['worst'] > -5e8).all(), 'worst -inf: {}'.format(hyps_t['worst'])

                # ----- step_end_of_sent == False
                add_next_sent = ((~is_step_eos) & allow_to_add)
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(add_next_sent).long()

                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]
            else:
                # all add to next_sent_beam
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(allow_to_add).long()
                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]

            # finish loop beam size
            if n_bucket_full.all():
                break

        # update next beam content get beam_size
        n_sent_b_t_out_val, n_sent_b_t_out_idx = torch.topk(n_sent_b_t_val, beam_size, dim=1, largest=True, sorted=True)
        n_sent_b_t_out_val = n_sent_b_t_out_val.masked_fill_(n_sent_b_t_out_val < -5e8, 0)

        n_sent_b_t_out_widx = n_sent_b_t_widx.gather(1, n_sent_b_t_out_idx)
        n_sent_b_t_out_pidx = n_sent_b_t_pidx.gather(1, n_sent_b_t_out_idx)

        beam_scores = beam_scores.new(n_sent_b_t_out_val.view(-1).type_as(beam_scores))
        beam_words = generated.new(n_sent_b_t_out_widx.view(-1))
        beam_idx = positions.new(n_sent_b_t_out_pidx.view(-1))
        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if torch.all(done_t):
            break

    output, hyp_len = beam_efficient_final_decoded(hyps_t, bs, src_len, pad_index, model.eos_index, beam_size, nbest)
    return output, hyp_len


def generate_beam_gpu_sample_topn(
        model, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping,
        max_len=200, nbest=None, sample_temperature=None, replacement=False,
        sample_topn=100,
        **kwargs
):
    """
        ** this version acquire the top_n words, and then sample them into 2 * beam_size
    :param model:
    :param src_enc:
    :param src_len:
    :param tgt_lang_id:
    :param beam_size:
    :param length_penalty:
    :param early_stopping:
    :param max_len:
    :param nbest:
    :param sample_temperature:
    :param sample_replacement:
    :param sample_topn:
    :param kwargs:
    :return:
    """
    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1
    assert sample_topn > 2 * beam_size
    if sample_temperature is None:
        sample_temperature = 1
    pad_index = model.pad_index

    # batch size / number of words
    bs = len(src_len)
    n_words = model.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(model.eos_index)  # we use <EOS> for <BOS> everywhere

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
    # generated hypotheses
    device = src_enc.device

    done_t = torch.zeros(bs, device=device).bool()
    # hyps_size = beam_size
    # hyps_size = hyp_size_multiple * beam_size
    hyps_size = beam_size
    hyps_t = {
        "len": torch.zeros(bs, device=device).long(),
        "hyps": generated.new(max_len, bs, hyps_size).fill_(pad_index),
        "worst": src_enc.new(bs).float().fill_(1e9),
        "score": src_enc.new(bs, hyps_size).float().fill_(-1e9),
        "hyp_len": torch.zeros(bs, hyps_size, device=device).long(),
    }
    # testing comapare
    while cur_len < max_len:

        # compute word scores
        tensor = model.forward(
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
        assert tensor.size() == (1, bs * beam_size, model.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = model.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
        logits = scores / sample_temperature
        lprobs = F.log_softmax(logits, dim=-1)  # (bs * beam_size, n_words)

        _lprobs = lprobs + beam_scores[:, None].expand_as(logits)  # (bs * beam_size, n_words)
        # _lprobs = _lprobs.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)
        leftover_lprobs, leftover_words = torch.topk(_lprobs, sample_topn, dim=1, largest=True, sorted=True)
        # leftover_lprobs:  (bs * beam_size, sample_topn)
        # leftover_words:   (bs * beam_size, sample_topn)
        leftover_logits = logits.gather(dim=1, index=leftover_words)
        leftover_probs = F.softmax(leftover_logits, dim=-1)

        # convert to beam_size * sample_topn
        leftover_lprobs = leftover_lprobs.view(bs, beam_size * sample_topn)
        leftover_words = leftover_words.view(bs, beam_size * sample_topn)
        if cur_len == 1:
            leftover_probs = leftover_probs.view(bs, beam_size, sample_topn).contiguous()
            leftover_probs[:, 1:, :] = 0
        leftover_probs = leftover_probs.view(bs, beam_size * sample_topn)

        # assert (leftover_probs > 0).sum(-1)
        assert ((leftover_probs > 0).sum(-1) >= 2 * beam_size).all(), '(leftover_probs > 0)<2beam, consider replacement'
        choice_sample_indices = torch.multinomial(leftover_probs, 2 * beam_size, replacement=replacement)
        choice_lprobs = leftover_lprobs.gather(1, choice_sample_indices)
        choice_words = leftover_words.gather(1, choice_sample_indices)

        # sort for scores and words
        next_scores, next_word_idx = torch.sort(choice_lprobs, dim=1, descending=True)
        next_words = choice_words.gather(1, next_word_idx)
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)
        assert (next_scores > -1e9).all(), 'cur_len={}, next_scores: {}'.format(cur_len, next_scores)

        # todo: get next words loop tensors, replacing for sent_id in range(bs) loop
        worst_better = (hyps_t['worst'] >= (next_scores.max(dim=-1)[0].detach() / ((max_len - 1) ** length_penalty)))
        len_less_beam = hyps_t['len'] < beam_size
        new_is_done = worst_better.masked_fill_(len_less_beam, torch.tensor(0).bool()).bool()
        done_t = done_t | new_is_done
        not_done = ~done_t
        # todo: next_words for sentences
        beam_ids_t = next_words // n_words
        word_ids_t = next_words % n_words
        sent_beam_idx_t = beam_ids_t + torch.arange(bs, device=device).unsqueeze_(-1) * beam_size

        is_eos = (word_ids_t == model.eos_index) | word_ids_t.new(bs, 2 * beam_size).fill_(
            int(cur_len + 1 == max_len)).bool()
        # next_sent_beam = [[] for i in range(bs)]
        n_sent_b_t_val = src_enc.new(bs, 2 * beam_size).float().fill_(-1e9)
        n_sent_b_t_widx = generated.new(bs, 2 * beam_size).fill_(pad_index)
        n_sent_b_t_pidx = generated.new(bs, 2 * beam_size).fill_(0)
        # todo: future work: how to GPU this loop!
        for i in range(2 * beam_size):
            value = next_scores[:, i]
            is_step_eos = is_eos[:, i]
            beam_ids = beam_ids_t[:, i]
            word_ids = word_ids_t[:, i]
            sent_beam_idx = sent_beam_idx_t[:, i]
            any_step_eos = torch.any(is_step_eos)

            add_supposed_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
            batch_bucket_not_full = (n_sent_b_t_widx != pad_index).int().sum(-1) < beam_size
            allow_to_add = not_done & batch_bucket_not_full

            n_bucket_full = (n_sent_b_t_widx[add_supposed_sent_idx] != pad_index).int().sum(-1) >= beam_size

            if any_step_eos:
                bscore = value.detach() / (cur_len ** length_penalty)  # FIXME: check cur_len=len(hyp) in Hypo.add()
                len_less_beamsize = hyps_t['len'] < beam_size
                bscore_more_worst = bscore > hyps_t['worst']

                lenlessbeam_bscoremoreworst = len_less_beamsize | bscore_more_worst
                # add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & not_done
                add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & allow_to_add
                add_batch_idx = torch.arange(bs, device=device).masked_select(add_or_not).long()
                if add_batch_idx.numel() > 0:

                    hyp = generated[:cur_len].view(cur_len, bs, beam_size)[:, add_batch_idx].clone()
                    hyp = hyp.gather(2, beam_ids[add_batch_idx].unsqueeze(0).unsqueeze(-1).expand(
                        cur_len, add_batch_idx.size(0), 1))
                    hyp = hyp.squeeze(-1).detach()
                    bscore_batch = bscore[add_batch_idx].detach()

                    hyps_t['hyps'][:cur_len, add_batch_idx, -1] = hyp
                    hyps_t['score'][add_batch_idx, -1] = bscore_batch

                    hyps_t['len'][add_batch_idx] += 1

                    # resort hyps and score
                    # next_hyp_score, next_hyp_idx = torch.topk(
                    #     hyps_t['score'][add_batch_idx], hyps_size, dim=1, largest=True, sorted=True)
                    next_hyp_score, next_hyp_idx = torch.sort(
                        hyps_t['score'][add_batch_idx], dim=1, descending=True)

                    hyps_t['score'][add_batch_idx] = next_hyp_score
                    hyps_t['hyps'][:cur_len, add_batch_idx] = hyps_t['hyps'][:cur_len, add_batch_idx].gather(
                        2, next_hyp_idx.unsqueeze(0).expand_as(hyps_t['hyps'][:cur_len, add_batch_idx]))

                    # assigning worst
                    min_worst_bscore = torch.min(hyps_t['worst'][add_batch_idx], bscore_batch)
                    min_worst_sort = next_hyp_score[:, -1]

                    more_than_beam = (hyps_t['len'][add_batch_idx] > beam_size)
                    add_batch_more_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        more_than_beam).long()
                    add_batch_less_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        ~more_than_beam).long()
                    assert add_batch_more_idx.size(0) + add_batch_less_idx.size(0) == add_batch_idx.size(0)
                    worst_batch_idx = hyps_t['worst'][add_batch_idx].clone()

                    if add_batch_more_idx.numel() > 0:
                        worst_batch_idx[add_batch_more_idx] = min_worst_sort[add_batch_more_idx]
                    if add_batch_less_idx.numel() > 0:
                        worst_batch_idx[add_batch_less_idx] = min_worst_bscore[add_batch_less_idx]

                    hyps_t['worst'][add_batch_idx] = worst_batch_idx
                    hyps_t['len'] = torch.min(hyps_t['len'], hyps_t['len'].new([beam_size]).expand_as(hyps_t['len']))

                    assert (hyps_t['worst'] > -5e8).all(), 'worst -inf: {}'.format(hyps_t['worst'])

                # ----- step_end_of_sent == False
                add_next_sent = ((~is_step_eos) & allow_to_add)
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(add_next_sent).long()

                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]
            else:
                # all add to next_sent_beam
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(allow_to_add).long()
                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]

            # finish loop beam size
            if n_bucket_full.all():
                break

        # update next beam content?
        n_sent_b_t_out_val, n_sent_b_t_out_idx = torch.topk(
            n_sent_b_t_val, beam_size, dim=1, largest=True, sorted=True)
        n_sent_b_t_out_val = n_sent_b_t_out_val.masked_fill_(n_sent_b_t_out_val < -5e8, 0)

        n_sent_b_t_out_widx = n_sent_b_t_widx.gather(1, n_sent_b_t_out_idx)
        n_sent_b_t_out_pidx = n_sent_b_t_pidx.gather(1, n_sent_b_t_out_idx)

        beam_scores = beam_scores.new(n_sent_b_t_out_val.view(-1).type_as(beam_scores))
        beam_words = generated.new(n_sent_b_t_out_widx.view(-1))
        beam_idx = positions.new(n_sent_b_t_out_pidx.view(-1))
        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if torch.all(done_t):
            break

    output, hyp_len = beam_efficient_final_decoded(hyps_t, bs, src_len, pad_index, model.eos_index, beam_size, nbest)
    return output, hyp_len


def generate_diverse_beam_search_gpu(
        model, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, num_groups, diversity_strength,
        max_len=200, nbest=None,
):
    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1
    assert num_groups is not None and beam_size % num_groups == 0
    assert diversity_strength is not None
    # diversity_strength : (0.2 - 0.8)
    diversity_strength = -diversity_strength
    sub_beam_size = beam_size // num_groups

    pad_index = model.pad_index

    # batch size / number of words
    bs = len(src_len)
    n_words = model.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(model.eos_index)  # we use <EOS> for <BOS> everywhere

    # positions
    positions = src_len.new(max_len).long()
    positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

    # language IDs
    langs = positions.clone().fill_(tgt_lang_id)

    # scores for each sentence in the beam
    # beam_scores = src_enc.new(bs, beam_size).fill_(0)
    # beam_scores[:, 1:] = torch.tensor(-1e9).type_as(beam_scores)
    # beam_scores = beam_scores.view(-1)
    beam_scores = src_enc.new(bs, beam_size).fill_(torch.tensor(-1e9).type_as(src_enc))
    beam_scores[:, :num_groups] = 0
    # [x, y, x, y, x, y] -> [0, 0, x, y, x, y]
    # [x, y, z, x, y, z, x, y, z] -> [0, 0, x, y, x, y]
    beam_scores = beam_scores.view(-1)


    # current position
    cur_len = 1

    # cache compute states
    cache = {'slen': 0}
    # generated hypotheses
    device = src_enc.device

    done_t = torch.zeros(bs, device=device).bool()
    # hyps_size = beam_size
    # hyps_size = hyp_size_multiple * beam_size
    hyps_size = beam_size
    hyps_t = {
        "len": torch.zeros(bs, device=device).long(),
        "hyps": generated.new(max_len, bs, hyps_size).fill_(pad_index),
        "worst": src_enc.new(bs).float().fill_(1e9),
        "score": src_enc.new(bs, hyps_size).float().fill_(-1e9),
        "hyp_len": torch.zeros(bs, hyps_size, device=device).long(),
    }
    # testing comapare
    diversity_buf = src_enc.new(bs, n_words).float().fill_(0)
    while cur_len < max_len:

        # compute word scores
        tensor = model.forward(
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
        assert tensor.size() == (1, bs * beam_size, model.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = model.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)

        lprobs = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
        assert lprobs.size() == (bs * beam_size, n_words)
        # select next words with scores
        # _scores = lprobs + beam_scores[:, None].expand_as(lprobs)  # (bs * beam_size, n_words)
        # _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)
        # next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
        # assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        # ----- diverse beam search----

        _lprobs = lprobs.view(bs, beam_size, n_words)
        _beam_scores = beam_scores.view(bs, beam_size).contiguous().unsqueeze_(-1).expand_as(_lprobs)
        scores_G, words_G, beams_G = [], [], []
        for g in range(num_groups):
            lprobs_g = _lprobs[:, g::num_groups]
            # scores_g = _beam_scores[:, g::num_groups, :] if cur_len > 1 else None
            scores_g = _beam_scores[:, g::num_groups, :]
            assert lprobs_g.size(1) == sub_beam_size, '{} != {}'.format(lprobs_g.size(1), sub_beam_size)
            if cur_len == 1:
                assert (scores_g[:, 1:] < -1e3).all() and (scores_g[:, 0] == 0).all(), 'scores_g: {}'.format(scores_g)

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(lprobs_g, diversity_strength, diversity_buf.unsqueeze(1))
            else:
                lprobs_g = lprobs_g.contiguous()

            # beam step for each group
            _sent_scores = lprobs_g + scores_g
            _sent_scores = _sent_scores.view(bs, sub_beam_size * n_words)
            _next_scores, _next_words = torch.topk(_sent_scores, 2 * sub_beam_size, dim=1, largest=True, sorted=True)
            assert _next_scores.size() == _next_words.size() == (bs, 2 * sub_beam_size)
            # num=2, subbeam=3, g=0, -> subbeam=0 => beam=0 ?
            # num=2, subbeam=3, g=0, -> subbeam=1 => beam=2 ?
            # num=2, subbeam=3, g=0, -> subbeam=2 => beam=4 ?

            # num=2, subbeam=3, g=1, -> subbeam=0 => beam=1 ?
            # num=2, subbeam=3, g=1, -> subbeam=1 => beam=3 ?
            # num=2, subbeam=3, g=1, -> subbeam=2 => beam=5 ?
            # it's interleaving
            _sub_beam_ids = _next_words // n_words
            _sub_beam_ids.mul_(num_groups).add_(g)
            _word_ids = _next_words % n_words

            scores_G.append(_next_scores)
            words_G.append(_word_ids)
            beams_G.append(_sub_beam_ids)
            # update diversity penalty
            diversity_buf = diversity_buf.scatter_add_(
                dim=1, index=_word_ids, src=diversity_buf.new_ones(_word_ids.size())
            )
        next_scores = torch.stack(scores_G, dim=2).view(bs, 2 * beam_size)
        word_ids_t = torch.stack(words_G, dim=2).view(bs, 2 * beam_size)
        beam_ids_t = torch.stack(beams_G, dim=2).view(bs, 2 * beam_size)

        # ---------------------------

        # todo: get next words loop tensors, replacing for sent_id in range(bs) loop
        worst_better = (hyps_t['worst'] >= (next_scores.max(dim=-1)[0].detach() / ((max_len - 1) ** length_penalty)))
        len_less_beam = hyps_t['len'] < beam_size
        new_is_done = worst_better.masked_fill_(len_less_beam, torch.tensor(0).bool()).bool()
        done_t = done_t | new_is_done
        not_done = ~done_t
        # todo: next_words for sentences
        # beam_ids_t = next_words // n_words
        # word_ids_t = next_words % n_words
        sent_beam_idx_t = beam_ids_t + torch.arange(bs, device=device).unsqueeze_(-1) * beam_size

        is_eos = (word_ids_t == model.eos_index) | word_ids_t.new(bs, 2 * beam_size).fill_(
            int(cur_len + 1 == max_len)).bool()
        # next_sent_beam = [[] for i in range(bs)]
        n_sent_b_t_val = src_enc.new(bs, 2 * beam_size).float().fill_(-1e9)
        n_sent_b_t_widx = generated.new(bs, 2 * beam_size).fill_(pad_index)
        n_sent_b_t_pidx = generated.new(bs, 2 * beam_size).fill_(0)
        # todo: future work: how to GPU this loop!
        for i in range(2 * beam_size):
            value = next_scores[:, i]
            is_step_eos = is_eos[:, i]
            beam_ids = beam_ids_t[:, i]
            word_ids = word_ids_t[:, i]
            sent_beam_idx = sent_beam_idx_t[:, i]
            any_step_eos = torch.any(is_step_eos)

            add_supposed_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
            batch_bucket_not_full = (n_sent_b_t_widx != pad_index).int().sum(-1) < beam_size
            allow_to_add = not_done & batch_bucket_not_full

            n_bucket_full = (n_sent_b_t_widx[add_supposed_sent_idx] != pad_index).int().sum(-1) >= beam_size

            if any_step_eos:
                bscore = value.detach() / (cur_len ** length_penalty)  # FIXME: check cur_len=len(hyp) in Hypo.add()
                len_less_beamsize = hyps_t['len'] < beam_size
                bscore_more_worst = bscore > hyps_t['worst']

                lenlessbeam_bscoremoreworst = len_less_beamsize | bscore_more_worst
                # add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & not_done
                add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & allow_to_add
                add_batch_idx = torch.arange(bs, device=device).masked_select(add_or_not).long()
                if add_batch_idx.numel() > 0:

                    hyp = generated[:cur_len].view(cur_len, bs, beam_size)[:, add_batch_idx].clone()
                    hyp = hyp.gather(2, beam_ids[add_batch_idx].unsqueeze(0).unsqueeze(-1).expand(
                        cur_len, add_batch_idx.size(0), 1))
                    hyp = hyp.squeeze(-1).detach()
                    bscore_batch = bscore[add_batch_idx].detach()

                    hyps_t['hyps'][:cur_len, add_batch_idx, -1] = hyp
                    hyps_t['score'][add_batch_idx, -1] = bscore_batch

                    hyps_t['len'][add_batch_idx] += 1

                    # resort hyps and score
                    # next_hyp_score, next_hyp_idx = torch.topk(
                    #     hyps_t['score'][add_batch_idx], hyps_size, dim=1, largest=True, sorted=True)
                    next_hyp_score, next_hyp_idx = torch.sort(
                        hyps_t['score'][add_batch_idx], dim=1, descending=True)

                    hyps_t['score'][add_batch_idx] = next_hyp_score
                    hyps_t['hyps'][:cur_len, add_batch_idx] = hyps_t['hyps'][:cur_len, add_batch_idx].gather(
                        2, next_hyp_idx.unsqueeze(0).expand_as(hyps_t['hyps'][:cur_len, add_batch_idx]))

                    # assigning worst
                    min_worst_bscore = torch.min(hyps_t['worst'][add_batch_idx], bscore_batch)
                    min_worst_sort = next_hyp_score[:, -1]

                    more_than_beam = (hyps_t['len'][add_batch_idx] > beam_size)
                    add_batch_more_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        more_than_beam).long()
                    add_batch_less_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        ~more_than_beam).long()
                    assert add_batch_more_idx.size(0) + add_batch_less_idx.size(0) == add_batch_idx.size(0)
                    worst_batch_idx = hyps_t['worst'][add_batch_idx].clone()

                    if add_batch_more_idx.numel() > 0:
                        worst_batch_idx[add_batch_more_idx] = min_worst_sort[add_batch_more_idx]
                    if add_batch_less_idx.numel() > 0:
                        worst_batch_idx[add_batch_less_idx] = min_worst_bscore[add_batch_less_idx]

                    hyps_t['worst'][add_batch_idx] = worst_batch_idx
                    hyps_t['len'] = torch.min(hyps_t['len'], hyps_t['len'].new([beam_size]).expand_as(hyps_t['len']))

                    assert (hyps_t['worst'] > -5e8).all(), 'worst -inf: {}'.format(hyps_t['worst'])

                # ----- step_end_of_sent == False
                add_next_sent = ((~is_step_eos) & allow_to_add)
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(add_next_sent).long()

                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]
            else:
                # all add to next_sent_beam
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(allow_to_add).long()
                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != pad_index).int().sum(-1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]

            # finish loop beam size
            if n_bucket_full.all():
                break

        # update next beam content? get beam_size
        n_sent_b_t_out_val, n_sent_b_t_out_idx = torch.topk(
            n_sent_b_t_val, beam_size, dim=1, largest=True, sorted=True)
        n_sent_b_t_out_val = n_sent_b_t_out_val.masked_fill_(n_sent_b_t_out_val < -5e8, 0)

        n_sent_b_t_out_widx = n_sent_b_t_widx.gather(1, n_sent_b_t_out_idx)
        n_sent_b_t_out_pidx = n_sent_b_t_pidx.gather(1, n_sent_b_t_out_idx)

        beam_scores = beam_scores.new(n_sent_b_t_out_val.view(-1).type_as(beam_scores))
        beam_words = generated.new(n_sent_b_t_out_widx.view(-1))
        beam_idx = positions.new(n_sent_b_t_out_pidx.view(-1))
        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if torch.all(done_t):
            break

    output, hyp_len = beam_efficient_final_decoded(hyps_t, bs, src_len, pad_index, model.eos_index, beam_size, nbest)
    return output, hyp_len


def generate_beam_gpu_backup1(
        model, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping,
        max_len=200, nbest=None, sample_temperature=None, hyps_size_multiple=1
):
    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1

    # batch size / number of words
    bs = len(src_len)
    n_words = model.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(model.pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(model.eos_index)  # we use <EOS> for <BOS> everywhere

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
    # generated hypotheses
    device = src_enc.device

    done_t = torch.zeros(bs, device=device).bool()
    # hyps_size = beam_size
    hyps_size = hyps_size_multiple * beam_size
    hyps_t = {
        "len": torch.zeros(bs, device=device).long(),
        "hyps": generated.new(max_len, bs, hyps_size).fill_(model.pad_index),
        "score": src_enc.new(bs, hyps_size).fill_(-1e4),
        "hyp_len": torch.zeros(bs, hyps_size, device=device).long(),
        "worst": src_enc.new(bs).fill_(1e4),
    }

    while cur_len < max_len:

        # compute word scores
        tensor = model.forward(
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
        assert tensor.size() == (1, bs * beam_size, model.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = model.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)

        if sample_temperature is None or sample_temperature == 1.0:
            scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)
            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)
        else:
            lprobs = F.log_softmax(scores / sample_temperature, dim=-1)  # (bs * beam_size, n_words)

            _lprobs = lprobs + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _lprobs = _lprobs.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)
            choice_words = torch.multinomial(_lprobs.exp(), 2 * beam_size, replacement=False)
            choice_lprobs = _lprobs.gather(1, choice_words)

            next_scores, next_word_idx = torch.topk(choice_lprobs, 2 * beam_size, dim=1, largest=True, sorted=True)
            next_words = choice_words.gather(1, next_word_idx)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        # todo: get next words loop tensors, replacing for sent_id in range(bs) loop
        worst_better = (hyps_t['worst'] >= (next_scores.max(dim=-1)[0].detach() / ((max_len - 1) ** length_penalty)))
        is_done = worst_better.masked_fill_(hyps_t['len'] < beam_size, torch.tensor(0).bool()).bool()
        # old_done_t = done_t
        done_t = done_t | is_done
        not_done = ~done_t
        # todo: next_words for sentences
        beam_ids_t = next_words // n_words
        word_ids_t = next_words % n_words
        sent_beam_idx_t = beam_ids_t + torch.arange(bs, device=device).unsqueeze_(-1) * beam_size

        is_eos = (word_ids_t == model.eos_index) | word_ids_t.new(bs, 2 * beam_size).fill_(
            int(cur_len + 1 == max_len)).bool()
        # next_sent_beam = [[] for i in range(bs)]
        n_sent_b_t_val = src_enc.new(bs, 2 * beam_size).fill_(-1e4)
        n_sent_b_t_widx = generated.new(bs, 2 * beam_size).fill_(model.pad_index)
        n_sent_b_t_pidx = generated.new(bs, 2 * beam_size).fill_(0)
        # todo: future work: how to GPU this loop!
        for i in range(2 * beam_size):
            # idx = next_words[:, i]
            value = next_scores[:, i]
            is_step_eos = is_eos[:, i]
            beam_ids = beam_ids_t[:, i]
            word_ids = word_ids_t[:, i]
            sent_beam_idx = sent_beam_idx_t[:, i]
            any_step_eos = torch.any(is_step_eos)

            if any_step_eos:
                bscore = value.detach() / (cur_len ** length_penalty)  # FIXME: check cur_len=len(hyp) in Hypo.add()
                len_less_beamsize = hyps_t['len'] < beam_size
                bscore_more_worst = bscore > hyps_t['worst']

                # lenlessbeam_bscoremoreworst = len_less_beamsize | bscore_more_worst
                # add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & not_done
                add_or_not = (len_less_beamsize | bscore_more_worst) & is_step_eos & not_done
                add_batch_idx = torch.arange(bs, device=device).masked_select(add_or_not).long()
                if add_batch_idx.numel() > 0:

                    hyp = generated[:cur_len].view(cur_len, bs, beam_size)[:, add_batch_idx].clone()
                    hyp = hyp.gather(2, beam_ids[add_batch_idx].unsqueeze(0).unsqueeze(-1).expand(
                        cur_len, add_batch_idx.size(0), 1))
                    hyp = hyp.squeeze(-1).detach()
                    bscore_batch = bscore[add_batch_idx].detach().type_as(hyps_t['score'])

                    hyps_t['hyps'][:cur_len, add_batch_idx, -1] = hyp
                    hyps_t['score'][add_batch_idx, -1] = bscore_batch

                    # resort hyps and score
                    next_hyp_score, next_hyp_idx = torch.topk(
                        hyps_t['score'][add_batch_idx], hyps_size, dim=1, largest=True, sorted=True)

                    min_worst_bscore = torch.min(hyps_t['worst'][add_batch_idx], bscore_batch)
                    min_worst_sort = next_hyp_score[:, -1]

                    more_than_beam = (hyps_t['len'][add_batch_idx] >= beam_size)
                    add_batch_more_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        more_than_beam).long()
                    add_batch_less_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        ~more_than_beam).long()
                    # assert add_batch_more_idx.size(0) + add_batch_less_idx.size(0) == add_batch_idx.size(0)
                    worst_batch_idx = hyps_t['worst'][add_batch_idx].clone()

                    if add_batch_more_idx.numel() > 0:
                        worst_batch_idx[add_batch_more_idx] = min_worst_sort[add_batch_more_idx]
                    if add_batch_less_idx.numel() > 0:
                        worst_batch_idx[add_batch_less_idx] = min_worst_bscore[add_batch_less_idx]

                    hyps_t['worst'][add_batch_idx] = worst_batch_idx

                    hyps_t['score'][add_batch_idx] = next_hyp_score
                    hyps_t['hyps'][:cur_len, add_batch_idx] = hyps_t['hyps'][:cur_len, add_batch_idx].gather(
                        2, next_hyp_idx.unsqueeze(0).expand_as(hyps_t['hyps'][:cur_len, add_batch_idx]))

                    hyps_t['len'][add_batch_idx] += 1

                # ----- step_end_of_sent == False
                add_next_sent = ((~is_step_eos) & not_done)
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(add_next_sent).long()
                n_sent_b_t_val[add_next_sent_idx, i] = value[add_next_sent_idx].type_as(n_sent_b_t_val)
                n_sent_b_t_widx[add_next_sent_idx, i] = word_ids[add_next_sent_idx]
                n_sent_b_t_pidx[add_next_sent_idx, i] = sent_beam_idx[add_next_sent_idx]
            else:
                # all add to next_sent_beam
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
                n_sent_b_t_val[add_next_sent_idx, i] = value[add_next_sent_idx].type_as(n_sent_b_t_val)
                n_sent_b_t_widx[add_next_sent_idx, i] = word_ids[add_next_sent_idx]
                n_sent_b_t_pidx[add_next_sent_idx, i] = sent_beam_idx[add_next_sent_idx]

            # finish loop beam size
            if done_t.all():
                break

        # update next beam content?
        n_sent_b_t_out_val, n_sent_b_t_out_idx = torch.topk(
            n_sent_b_t_val, beam_size, dim=1, largest=True, sorted=True)
        n_sent_b_t_out_val = n_sent_b_t_out_val.masked_fill_(n_sent_b_t_out_val < -1e3, 0)
        # assert (n_sent_b_t_out_val > -1e3).all() or cur_len + 1 == max_len or torch.all(
        #     done_t), 'len={}/{} n_sent_b_t_out_val: \n{}\n{}'.format(
        #     cur_len, max_len, n_sent_b_t_out_val, done_t)

        n_sent_b_t_out_widx = n_sent_b_t_widx.gather(1, n_sent_b_t_out_idx)
        n_sent_b_t_out_pidx = n_sent_b_t_pidx.gather(1, n_sent_b_t_out_idx)

        beam_scores = beam_scores.new(n_sent_b_t_out_val.view(-1))
        beam_words = generated.new(n_sent_b_t_out_widx.view(-1))
        beam_idx = positions.new(n_sent_b_t_out_pidx.view(-1))
        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if torch.all(done_t):
            break

    return beam_efficient_final_decoded(hyps_t, bs, src_len, model.pad_index, model.eos_index, beam_size, nbest)


def generate_beam_efficient_validate_cpu(
        self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200, nbest=None,
        sample_temperature=None, hyps_size_multiple=1
):
    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1

    # batch size / number of words
    bs = len(src_len)
    n_words = self.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view(
        (bs * beam_size,) + src_enc.shape[1:])
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # current position
    cur_len = 1

    # TODO: CPU variables
    # generated sentences (batch with beam current hypotheses)
    generated_cpu = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated_cpu.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
    generated_cpu[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
    # generated hypotheses
    generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping=False) for _ in range(bs)]
    # positions
    positions_cpu = src_len.new(max_len).long()
    positions_cpu = torch.arange(max_len, out=positions_cpu).unsqueeze(1).expand_as(generated_cpu)
    # language IDs
    langs_cpu = positions_cpu.clone().fill_(tgt_lang_id)
    # scores for each sentence in the beam
    beam_scores_cpu = src_enc.new(bs, beam_size).fill_(0)
    beam_scores_cpu[:, 1:] = torch.tensor(-1e9).type_as(beam_scores_cpu)
    beam_scores_cpu = beam_scores_cpu.view(-1)
    # cache compute states
    cache_cpu = {'slen': 0}
    # done sentences
    done_cpu = [False for _ in range(bs)]

    # TODO: GPU variables
    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
    # positions
    positions = src_len.new(max_len).long()
    positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
    # language IDs
    langs = positions.clone().fill_(tgt_lang_id)

    # scores for each sentence in the beam
    beam_scores = src_enc.new(bs, beam_size).fill_(0)
    beam_scores[:, 1:] = torch.tensor(-1e9).type_as(beam_scores)
    beam_scores = beam_scores.view(-1)
    # cache compute states
    cache = {'slen': 0}
    device = src_enc.device
    done_t = torch.zeros(bs, device=device).bool()
    # hyps_size = hyps_size_multiple * beam_size

    hyps_size = beam_size
    hyps_t = {
        "len": torch.zeros(bs, device=device).long(),
        "hyps": generated.new(max_len, bs, hyps_size).fill_(self.pad_index),
        "worst": src_enc.new(bs).float().fill_(1e9),
        "score": src_enc.new(bs, hyps_size).float().fill_(-1e9),
        "hyp_len": torch.zeros(bs, hyps_size, device=device).long()
    }

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
        # FIXME: understand beam? why tensor len=1 ? it should be cur_len ?
        assert tensor.size() == (1, bs * beam_size, self.dim)
        tensor = tensor.data[-1, :, :]  # (bs * beam_size, dim)
        scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
        scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
        assert scores.size() == (bs * beam_size, n_words)

        # select next words with scores
        _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
        _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

        next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)
        # assert next_scores.dtype == src_enc.dtype, '{} != {}'.format(next_scores.dtype, src_enc.dtype)

        # FIXME: CPU computations ======================================
        next_batch_beam = []
        gen_hyp_is_done = [generated_hyps[j].is_done(next_scores[j].max().item()) for j in range(bs)]
        # ---- test is done logic
        gen_hyp_len_less = [len(generated_hyps[j]) < generated_hyps[j].n_hyp for j in range(bs)]
        gen_hyp_worst_better = [
            generated_hyps[j].worst_score >= next_scores[j].max().item() / (max_len - 1) ** length_penalty for j in
            range(bs)]
        gen_hyp_is_done_manual = list(gen_hyp_worst_better)
        for j in range(bs):
            if gen_hyp_len_less[j]:
                gen_hyp_is_done_manual[j] = False

        assert all(x == y for x, y in zip(gen_hyp_is_done, gen_hyp_is_done_manual)), 'logic wrong: \n{}\n{}'.format(
            gen_hyp_is_done, gen_hyp_is_done_manual
        )

        # ---------------------------
        add_or_not_cpu = [[] for i in range(bs)]
        loop_next_words_cpu = [0] * bs

        # for each sentence
        for sent_id in range(bs):
            # if we are done with this sentence
            done_cpu[sent_id] = done_cpu[sent_id] or gen_hyp_is_done[sent_id]
            if done_cpu[sent_id]:
                next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                continue
            # next sentence beam content
            next_sent_beam = []
            # add_or_not_cpu.append([])
            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                # get beam and word IDs
                beam_id = idx // n_words
                word_id = idx % n_words
                # end of sentence, or next word
                if word_id == self.eos_index or cur_len + 1 == max_len:
                    add_or_not_ = generated_hyps[sent_id].add(
                        generated_cpu[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                else:
                    next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))
                    add_or_not_ = False

                add_or_not_cpu[sent_id].append(add_or_not_)
                loop_next_words_cpu[sent_id] = loop_next_words_cpu[sent_id] + 1
                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break
            add_or_not_cpu[sent_id] = add_or_not_cpu[sent_id] + [False] * (2 * beam_size - len(add_or_not_cpu[sent_id]))

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (sent_id + 1)
        # sanity check / prepare next batch
        assert len(next_batch_beam) == bs * beam_size
        beam_scores_cpu = beam_scores_cpu.new([x[0] for x in next_batch_beam])
        beam_words_cpu = generated_cpu.new([x[1] for x in next_batch_beam])
        beam_idx_cpu = src_len.new([x[2] for x in next_batch_beam])
        # re-order batch and internal states
        generated_cpu = generated_cpu[:, beam_idx_cpu]
        generated_cpu[cur_len] = beam_words_cpu
        for k in cache_cpu.keys():
            if k != 'slen':
                cache_cpu[k] = (cache_cpu[k][0][beam_idx_cpu], cache_cpu[k][1][beam_idx_cpu])

        # FIXME: GPU computations ======================================
        worst_better = (hyps_t['worst'] >= (next_scores.max(dim=-1)[0].detach() / ((max_len - 1) ** length_penalty)))
        len_less_beam = hyps_t['len'] < beam_size
        new_is_done = worst_better.masked_fill_(len_less_beam, torch.tensor(0).bool()).bool()
        old_done_t = done_t
        done_t = done_t | new_is_done
        # assert (done_t.new(done_cpu) == done_t).all(), 'done not right: gpu\n{}\ndone_cpu\n'.format(
        #     done_t.tolist(), done_cpu
        # )
        if (done_t.new(done_cpu) != done_t).any():
            # invest each element
            worst_better_cpu = gen_hyp_worst_better
            len_less_beam_cpu = gen_hyp_len_less
            if (worst_better != worst_better.new(worst_better_cpu)).any():
                logger.info('Worst_better_wrong: gpu{}\nworst_bteer_cpu\n{}\n{}\nworst_gpu\n{}\nworst_cpu{}'.format(
                    worst_better.tolist(), worst_better_cpu, worst_better == worst_better.new(worst_better_cpu),
                    hyps_t['worst'].tolist(), [gen.worst_score for gen in generated_hyps]
                ))
            if (len_less_beam != len_less_beam.new(len_less_beam_cpu)).any():
                logger.info('len_less_beam_wrong: gpu{}\nlen_less_beam_cpu\n{}\n{}\nlen_gpu\n{}\nlen_cpu{}'.format(
                    len_less_beam.tolist(), len_less_beam_cpu, len_less_beam != len_less_beam.new(len_less_beam_cpu),
                    hyps_t['len'].tolist(), [len(gen) for gen in generated_hyps]
                ))
            # assert (len_less_beam == len_less_beam.new(len_less_beam_cpu)).all(), 'len_lessbeam:\n{}\n{}'.format(
            #     len_less_beam.tolist(), len_less_beam_cpu
            # )
            raise ValueError('done wrong')

        not_done = ~done_t
        # todo: next_words for sentences
        beam_ids_t = next_words // n_words
        word_ids_t = next_words % n_words
        sent_beam_idx_t = beam_ids_t + torch.arange(bs, device=device).unsqueeze_(-1) * beam_size

        is_eos = (word_ids_t == self.eos_index) | word_ids_t.new(bs, 2 * beam_size).fill_(
            int(cur_len + 1 == max_len)).bool()
        # next_sent_beam = [[] for i in range(bs)]
        n_sent_b_t_val = src_enc.new(bs, 2 * beam_size).float().fill_(-1e9)
        n_sent_b_t_widx = generated.new(bs, 2 * beam_size).fill_(self.pad_index)
        n_sent_b_t_pidx = generated.new(bs, 2 * beam_size).fill_(0)
        add_or_not_gpu = generated.new(bs, 2 * beam_size).fill_(0).bool()
        loop_next_words_gpu = generated.new(bs).fill_(0)

        for i in range(2 * beam_size):
            # idx = next_words[:, i]
            value = next_scores[:, i]
            is_step_eos = is_eos[:, i]
            beam_ids = beam_ids_t[:, i]
            word_ids = word_ids_t[:, i]
            sent_beam_idx = sent_beam_idx_t[:, i]
            any_step_eos = torch.any(is_step_eos)

            add_supposed_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
            batch_bucket_not_full = (n_sent_b_t_widx != self.pad_index).int().sum(-1) < beam_size
            allow_to_add = not_done & batch_bucket_not_full

            n_bucket_full = (n_sent_b_t_widx[add_supposed_sent_idx] != self.pad_index).int().sum(-1) >= beam_size
            n_bucket_not_full = ~n_bucket_full
            loop_next_words_gpu[add_supposed_sent_idx] = loop_next_words_gpu[
                                                             add_supposed_sent_idx] + n_bucket_not_full.type_as(
                loop_next_words_gpu)

            if any_step_eos:
                bscore = value.detach() / (cur_len ** length_penalty)  # FIXME: check if cur_len=len(hyp) in Hypo.add()
                len_less_beamsize = hyps_t['len'] < beam_size
                bscore_more_worst = bscore > hyps_t['worst']

                lenlessbeam_bscoremoreworst = len_less_beamsize | bscore_more_worst
                # add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & not_done
                add_or_not = lenlessbeam_bscoremoreworst & is_step_eos & allow_to_add
                add_or_not_gpu[:, i] = add_or_not
                # add_or_not_back = ((hyps_t['len'] < beam_size) | (bscore > hyps_t['worst']) & is_step_eos) & not_done
                # assert torch.all(add_or_not == add_or_not_back), "add_or_not: \n{} \n{}".format(add_or_not, add_or_not_back)
                add_batch_idx = torch.arange(bs, device=device).masked_select(add_or_not).long()
                if add_batch_idx.numel() > 0:

                    if cur_len + 1 < max_len:
                        assert torch.all(word_ids[
                                             add_batch_idx] == self.eos_index), 'curlen={}, words_ids:\n{}\nlen_less\n{}\nbscore_more_worst\n{}\nadd_ornot\n{}'.format(
                            cur_len, word_ids, len_less_beamsize, bscore_more_worst, add_or_not)

                    hyp = generated[:cur_len].view(cur_len, bs, beam_size)[:, add_batch_idx].clone()
                    hyp = hyp.gather(2, beam_ids[add_batch_idx].unsqueeze(0).unsqueeze(-1).expand(
                        cur_len, add_batch_idx.size(0), 1))
                    hyp = hyp.squeeze(-1)
                    # bvalue_batch = value[add_batch_idx].clone().type_as(hyps_t['score']).detach()
                    bscore_batch = bscore[add_batch_idx].clone().detach()

                    hyps_t['hyps'][:cur_len, add_batch_idx, -1] = hyp
                    hyps_t['score'][add_batch_idx, -1] = bscore_batch

                    hyps_t['len'][add_batch_idx] += 1

                    # resort hyps and score
                    next_hyp_score, next_hyp_idx = torch.topk(
                        hyps_t['score'][add_batch_idx], hyps_size, dim=1, largest=True, sorted=True)

                    # logger.info('Add with eos: (curlen={},i={}) batch:\n{}\nworst\n{}\n{}\n{}\nhyp_len\n{}'.format(
                    #     cur_len, i, add_batch_idx, min_worst_bscore, min_worst_sort, hyps_t['worst'][add_batch_idx],
                    #     hyps_t['len'][add_batch_idx]
                    # ))
                    # what happen here:
                    #   len=0: worst=1e4  => len=1, worst=-0.5      score=[-0.5, -1e4, -1e4]
                    #   len=1: worst=-0.5 => len=2, worst=-0.6      score=[-0.5, -0.6, -1e4]
                    #   len=2: worst=-0.6 => len=3, worst=-0.7      score=[-
                    #   len=3: worst=-0.7 => len=1, worst=

                    hyps_t['score'][add_batch_idx] = next_hyp_score
                    hyps_t['hyps'][:cur_len, add_batch_idx] = hyps_t['hyps'][:cur_len, add_batch_idx].gather(
                        # 2, next_hyp_idx.unsqueeze(0).expand_as(cur_len, add_batch_idx.size(0), beam_size))
                        2, next_hyp_idx.unsqueeze(0).expand_as(hyps_t['hyps'][:cur_len, add_batch_idx]))

                    # assigning worst
                    min_worst_bscore = torch.min(hyps_t['worst'][add_batch_idx], bscore_batch)
                    min_worst_sort = next_hyp_score[:, -1]

                    more_than_beam = (hyps_t['len'][add_batch_idx] > beam_size)
                    add_batch_more_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        more_than_beam).long()
                    add_batch_less_idx = torch.arange(add_batch_idx.size(0), device=device).masked_select(
                        ~more_than_beam).long()
                    assert add_batch_more_idx.size(0) + add_batch_less_idx.size(0) == add_batch_idx.size(0)
                    worst_batch_idx = hyps_t['worst'][add_batch_idx].clone()

                    if add_batch_more_idx.numel() > 0:
                        worst_batch_idx[add_batch_more_idx] = min_worst_sort[add_batch_more_idx]
                    if add_batch_less_idx.numel() > 0:
                        worst_batch_idx[add_batch_less_idx] = min_worst_bscore[add_batch_less_idx]

                    hyps_t['worst'][add_batch_idx] = worst_batch_idx
                    hyps_t['len'] = torch.min(hyps_t['len'], hyps_t['len'].new([beam_size]).expand_as(hyps_t['len']))

                    assert (hyps_t['worst'] > -5e8).all(), 'worst -inf: {}'.format(hyps_t['worst'])

                # ----- step_end_of_sent == False
                # FIXME: make sure that n_sent_b_t_val/n_sent_b_t_widx .. only contains beam ones
                add_next_sent = ((~is_step_eos) & allow_to_add)
                add_next_sent_idx = torch.arange(bs, device=device).masked_select(add_next_sent).long()

                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != self.pad_index).int().sum(
                    -1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                # n_sent_b_t_val[add_next_sent_idx, i] = value[add_next_sent_idx].type_as(n_sent_b_t_val)
                # n_sent_b_t_widx[add_next_sent_idx, i] = word_ids[add_next_sent_idx]
                # n_sent_b_t_pidx[add_next_sent_idx, i] = sent_beam_idx[add_next_sent_idx]
                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]

            else:
                # all add to next_sent_beam
                # add_next_sent_idx = torch.arange(bs, device=device).masked_select(not_done).long()
                # n_sent_b_t_val[add_next_sent_idx, i] = value[add_next_sent_idx].type_as(n_sent_b_t_val)
                # n_sent_b_t_widx[add_next_sent_idx, i] = word_ids[add_next_sent_idx]
                # n_sent_b_t_pidx[add_next_sent_idx, i] = sent_beam_idx[add_next_sent_idx]

                add_next_sent_idx = torch.arange(bs, device=device).masked_select(allow_to_add).long()
                n_sent_bucket_not_full = (n_sent_b_t_widx[add_next_sent_idx] != self.pad_index).int().sum(
                    -1) < beam_size
                n_sent_add_idx = add_next_sent_idx.masked_select(n_sent_bucket_not_full)

                n_sent_b_t_val[n_sent_add_idx, i] = value[n_sent_add_idx]
                n_sent_b_t_widx[n_sent_add_idx, i] = word_ids[n_sent_add_idx]
                n_sent_b_t_pidx[n_sent_add_idx, i] = sent_beam_idx[n_sent_add_idx]

            if n_bucket_full.all():
                break

            # in what case  n_sent_b_t_val not filled up?
            #       all(done_t) or add_or_not    -> what cpu looks like?
            #       cpu: not done!, none are add to hyps_cpu? Or is it?
        # update next beam content?
        # assert ((n_sent_b_t_widx != self.pad_index).int().sum(-1) == beam_size).all(), 'not pad_index: {}'.format(n_sent_b_t_widx)

        n_sent_b_t_out_val, n_sent_b_t_out_idx = torch.topk(
            n_sent_b_t_val, beam_size, dim=1, largest=True, sorted=True)
        n_sent_b_t_out_val = n_sent_b_t_out_val.masked_fill_(n_sent_b_t_out_val < -1e8, 0)
        assert (n_sent_b_t_out_val > -1e8).all() or cur_len + 1 == max_len or torch.all(
            done_t), 'len={}/{} n_sent_b_t_out_val: \n{}\n{}'.format(
            cur_len, max_len, n_sent_b_t_out_val, done_t)

        n_sent_b_t_out_widx = n_sent_b_t_widx.gather(1, n_sent_b_t_out_idx)
        n_sent_b_t_out_pidx = n_sent_b_t_pidx.gather(1, n_sent_b_t_out_idx)
        assert n_sent_b_t_out_pidx.size() == (bs, beam_size) and beam_idx_cpu.size() == (bs * beam_size,)

        beam_scores = beam_scores.new(n_sent_b_t_out_val.view(-1).type_as(beam_scores))
        beam_words = generated.new(n_sent_b_t_out_widx.view(-1))
        beam_idx = positions.new(n_sent_b_t_out_pidx.view(-1))
        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # FIXME: sanatify check
        #   generated vs generated_cpu
        #   beam_scores vs beam_scores_cpu
        #   beam_idx vs beam_idx_cpu
        assert beam_idx.size() == beam_idx_cpu.size(), 'size: {} != {}'.format(beam_idx.size(), beam_idx_cpu.size())
        test_fit = True
        hyp_lens_gpu = hyps_t['len']
        hyp_lens_cpu = torch.tensor([len(g) for g in generated_hyps], device=device).type_as(hyp_lens_gpu)
        # add_or_not_cpu = add_or_not_gpu.new(add_or_not_cpu)

        loop_next_words_cpu = loop_next_words_gpu.new(loop_next_words_cpu)
        worst_score_cpu = hyps_t['worst'].new([gen.worst_score for gen in generated_hyps])

        if torch.any(loop_next_words_gpu != loop_next_words_cpu):
            logger.info('loop_next_words_cpu: curlen={}/{}, loop_next_words_gpu\n{}\nloop_next_words_cpu\n{}'.format(
                cur_len, max_len, loop_next_words_gpu, loop_next_words_cpu
            ))
            test_fit = False

        # if torch.any(add_or_not_cpu != add_or_not_gpu):
        #     logger.info('add_or_not_cpu: curlen={}/{}, add_or_not_gpu\n{}\nadd_or_not_cpu\n{}'.format(
        #         cur_len, max_len, add_or_not_gpu, add_or_not_cpu
        #     ))
        #     test_fit = False
        # if torch.any(worst_score_cpu != hyps_t['worst']):
        if not torch.allclose(worst_score_cpu, hyps_t['worst']):
            logger.info('worst: curlen={}/{}, worst_gpu\n{}\nworst_cpu\n{}\n{}'.format(
                cur_len, max_len, hyps_t['worst'], worst_score_cpu, hyps_t['worst'] == worst_score_cpu,
            ))
            test_fit = False

        if torch.any(hyp_lens_gpu != hyp_lens_cpu):
            logger.info('hyp_lens_cpu: curlen={}/{}, hyp_lens_gpu\n{}hyp_lens_cpu\n{}'.format(
                cur_len, max_len, hyp_lens_gpu, hyp_lens_cpu
            ))
            test_fit = False

        if torch.any(beam_scores != beam_scores_cpu):
            logger.info('beam_scores: curlen={}/{}, beam_scores\n{}\nbeam_scores_cpu\n{}\n{}'.format(
                cur_len, max_len, beam_scores, beam_scores_cpu, beam_scores == beam_scores_cpu
            ))
            test_fit = False

        # if torch.any(beam_idx != beam_idx_cpu):
        #     # not triggering this because the 2 beam idx may have the same score they are order differenly
        #     logger.info('beam_idx: curlen={}, beam_idx\n{}beam_idx_cpu\n{}\n{}'.format(
        #         cur_len, beam_idx, beam_idx_cpu, beam_idx == beam_idx_cpu
        #     ))
        #     test_fit = False
        # if torch.any(beam_words != beam_words_cpu):
        #     # not triggering this because the 2 words may have the same score they are order differenly
        #     # instead of Raise exception, report number of wrongs
        #     logger.info('beam_words: curlen={}, beam_words\n{}beam_words_cpus\n{}\n{}'.format(
        #         cur_len, beam_words, beam_words_cpu, beam_words == beam_words_cpu
        #     ))
        #     test_fit = False
        #     num_wrongs = (beam_words != beam_words_cpu).int().sum()
        done_cpu_t = done_t.new(done_cpu)
        if (done_t != done_cpu_t).any():
            logger.info('done_t: curlen={}/{}, done_t\n{}\ndone_cpu_t\n{}\n{}'.format(
                cur_len, max_len, done_t.tolist(), done_cpu_t.tolist(), (done_t == done_cpu_t).tolist()
            ))
            test_fit = False

        if not test_fit:
            logger.info('---- Final Log of Failed Test! -------------')
            for sent_id in range(bs):
                if done_t[sent_id] != done_cpu[sent_id]:
                    logger.info('test_done[curlen={}][b={}]: {}/{} -> is_done: {}'.format(
                        cur_len, sent_id, done_t[sent_id], done_cpu[sent_id],
                        generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                    ))
            logger.info('Done test: old_done:\n{}\nworst_better:\n{}\nhyps_len:\n{}\nis_done:\n{}\nworst\n{}'.format(
                old_done_t, worst_better, hyps_t['len'], new_is_done, hyps_t['worst']
            ))
            raise ValueError('some problem occur: done_t:\n{}\ndone\n{}\nhyps_cpu\n{}'.format(
                done_t.tolist(), done_cpu, [len(x) for x in generated_hyps]
            ))

        cur_len += 1
        if torch.all(done_t) & all(done_cpu):
            break

    output, hyp_len = beam_efficient_final_decoded(
        hyps_t, bs, src_len, self.pad_index, self.eos_index, beam_size, nbest)
    output_cpu, hyp_len_cpu = compute_final_decoded(
        generated_hyps, bs, src_len, self.pad_index, self.eos_index, beam_size, nbest)
    final_fit = True
    # if (hyp_len != hyp_len_cpu).any():
    #     logger.info('Final hyp_len wring: hyp_len:\n{}\nhyp_len_cpu\n{}\n{}'.format(
    #         hyp_len, hyp_len_cpu, hyp_len == hyp_len_cpu
    #     ))
    #     final_fit =False
    if output.size(0) != output_cpu.size(0):
        logger.info('Final output not the same lengths')
        return output, hyp_len

    if (output != output_cpu).any():
        logger.info('Final output_cpu wrong: output:\n{}\noutput_cpu\n{}\n{}'.format(
            output, output_cpu, output == output_cpu
        ))
        diff = (output != output_cpu).view(output.size(0), -1)
        output_diff = output.view(output.size(0), -1).masked_select(diff)
        output_diff_cpu = output_cpu.view(output.size(0), -1).masked_select(diff)
        # ? multiple of 0.03125 = 1/32
        logger.info('Final output_cpu wrong %: {} diff_gpu:\n{}\ndiff_cpu\n{}'.format(
            diff.int().sum(0).float().mean(), output_diff, output_diff_cpu
        ))
        final_fit = False

    output = output[:, :, 0]
    hyp_len = output[:, 0]
    #
    # if not final_fit:
    #     raise ValueError('final output wrong!')

    return output, hyp_len
