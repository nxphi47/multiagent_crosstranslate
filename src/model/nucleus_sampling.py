import math

import torch


# Nucleus has 2 version, beam and non-beam, the beam one just duplicates it, and sample each beam indepdently
def sample_topp(lprobs, sampling_topp, remove_indices=None):
    """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

    See `"The Curious Case of Neural Text Degeneration"
    (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

    Args:
        lprobs: (bsz x input_beam_size x vocab_size)
            the model's log-probabilities over the vocabulary at the current step

    Return: A tuple of (trimed_probs, truncated_indices) where:
        trimed_probs: (bsz x input_beam_size x ?)
            the model's probabilities over the elements selected to sample from. The
            width of the third dimension is determined by top-P.
        truncated_indices: (bsz x input_beam_size x ?)
            the indices of the chosen elements.

    ** for XLM code:
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        # so we have to remove bos and pad only, but not eos
        but in the original, they truncated first 2 words, then the vocab indices will be shifted
    """
    if remove_indices is None:
        remove_indices = [0, 2]
    probs = lprobs.exp_()
    probs[:, :, remove_indices] = 0

    # sort the last dimension (vocab dimension) in descending order
    sorted_probs, sorted_indices = probs.sort(descending=True)

    # compute a mask to indicate the words to be included in the top-P set.
    cumsum_probs = sorted_probs.cumsum(dim=2)
    mask = cumsum_probs.lt(sampling_topp)

    # note that mask was computed by 'lt'. One more word needs to be included
    # so that the cumulative probability mass can exceed p.
    cumsum_mask = mask.cumsum(dim=2)
    last_included = cumsum_mask[:, :, -1:]
    last_included.clamp_(0, mask.size()[2] - 1)
    mask = mask.scatter_(2, last_included, 1)

    # truncate unnecessary dims.
    max_dim = last_included.max()
    truncated_mask = mask[:, :, :max_dim + 1]
    truncated_probs = sorted_probs[:, :, :max_dim + 1]
    truncated_indices = sorted_indices[:, :, :max_dim + 1]

    # trim the words that are not in top-P by setting their probabilities
    # to 0, so that they would not be sampled later.
    trim_mask = (~truncated_mask)
    trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
    return trimed_probs, truncated_indices


