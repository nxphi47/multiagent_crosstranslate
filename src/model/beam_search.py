import torch


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9
        # self.worst_score = 1e4

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        add_or_not = False
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            add_or_not = True
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
        return add_or_not

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


def compute_final_decoded(generated_hyps, bs, src_len, pad_index, eos_index, beam_size, nbest=None):
    if nbest is None or nbest <= 0:
        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = eos_index

        # sanity check, why 2 * bs
        assert (decoded == eos_index).sum() == 2 * bs

        return decoded, tgt_len
    else:
        assert isinstance(nbest, int) and 1 <= nbest <= beam_size
        bests = []
        tgt_len = src_len.new(bs, nbest)
        for i, hypotheses in enumerate(generated_hyps):
            best_hyps = [z[1] for z in sorted(hypotheses.hyp, key=lambda x: x[0], reverse=True)[:nbest]]
            for j, x in enumerate(best_hyps):
                tgt_len[i, j] = len(x) + 1
            # tgt_len[i, :] = max([len(x) for x in best_hyps]) + 1  # +1 for the <EOS> symbol
            bests.append(best_hyps)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), nbest, bs).fill_(pad_index)
        # [max_len, nbest, bs]
        for i, hypos in enumerate(bests):
            for j, hyp in enumerate(hypos):
                decoded[:tgt_len[i, j] - 1, j, i] = hyp
                decoded[tgt_len[i, j] - 1, j, i] = eos_index

        # sanity check
        assert (decoded == eos_index).sum() == 2 * bs * nbest

        return decoded, tgt_len