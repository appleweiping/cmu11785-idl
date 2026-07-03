"""Connectionist Temporal Classification (CTC) loss for MyTorch (HW3P1).

Implements the classic forward-backward (alpha/beta) algorithm on the
blank-extended target sequence, producing the CTC loss and the gradient
w.r.t. the input log-probabilities' softmax outputs.

Convention: ``logits`` are already softmax **probabilities** of shape
``(T, B, C)`` where C includes the blank symbol at index ``blank`` (default 0).
"""

import numpy as np


class CTC:
    def __init__(self, BLANK=0):
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Insert blanks between/around the label sequence.

        target: (target_len,) -> extended (2*target_len + 1,) plus a
        ``skip_connect`` mask marking positions that may skip two back.
        """
        extended = [self.BLANK]
        skip = [0, 0]
        for i, s in enumerate(target):
            extended.append(s)
            extended.append(self.BLANK)
            if i > 0:
                # a skip connection is allowed if this label differs from the
                # previous label (so we can jump over the intervening blank)
                skip.append(1 if s != target[i - 1] else 0)
            if i < len(target) - 1 or True:
                skip.append(0)
        extended = np.array(extended)
        skip = np.array(skip[:len(extended)])
        return extended, skip

    def get_forward_probs(self, logits, extended, skip):
        """alpha[t, s] = P(prefix ending at extended[s] at time t)."""
        T = logits.shape[0]
        S = len(extended)
        alpha = np.zeros((T, S))
        alpha[0, 0] = logits[0, extended[0]]
        alpha[0, 1] = logits[0, extended[1]]
        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] * logits[t, extended[0]]
            for s in range(1, S):
                a = alpha[t - 1, s] + alpha[t - 1, s - 1]
                if skip[s]:
                    a += alpha[t - 1, s - 2]
                alpha[t, s] = a * logits[t, extended[s]]
        return alpha

    def get_backward_probs(self, logits, extended, skip):
        """beta[t, s] = P(suffix from extended[s] at time t)."""
        T = logits.shape[0]
        S = len(extended)
        beta = np.zeros((T, S))
        beta[T - 1, S - 1] = 1.0
        beta[T - 1, S - 2] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t, S - 1] = beta[t + 1, S - 1] * logits[t + 1, extended[S - 1]]
            for s in range(S - 2, -1, -1):
                b = (beta[t + 1, s] * logits[t + 1, extended[s]]
                     + beta[t + 1, s + 1] * logits[t + 1, extended[s + 1]])
                if s + 2 < S and skip[s + 2]:
                    b += beta[t + 1, s + 2] * logits[t + 1, extended[s + 2]]
                beta[t, s] = b
        return beta

    def get_posterior_probs(self, alpha, beta):
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-12
        return gamma / gamma_sum


class CTCLoss:
    def __init__(self, BLANK=0):
        self.BLANK = BLANK
        self.ctc = CTC(BLANK)

    def forward(self, logits, target, input_lengths, target_lengths):
        """
        logits:  (T, B, C) softmax probabilities
        target:  (B, max_target_len)
        Returns scalar mean loss over the batch.
        """
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        B = logits.shape[1]
        self.extended_list = []
        self.gammas = []

        total = 0.0
        for b in range(B):
            Ti = int(input_lengths[b])
            Li = int(target_lengths[b])
            tgt = target[b, :Li]
            ext, skip = self.ctc.extend_target_with_blank(tgt)
            lg = logits[:Ti, b, :]
            alpha = self.ctc.get_forward_probs(lg, ext, skip)
            beta = self.ctc.get_backward_probs(lg, ext, skip)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            self.extended_list.append((ext, skip, Ti))
            self.gammas.append(gamma)

            # CTC loss = -log P(target), with the total sequence probability
            # P = alpha[T-1, S-1] + alpha[T-1, S-2] (both valid final states).
            # gamma (= alpha*beta / P) gives the gradient below; note that
            # sum_s alpha[t,s]*beta[t,s] == P for every t, so the posterior
            # normalisation used by get_posterior_probs recovers exactly this.
            P = alpha[Ti - 1, len(ext) - 1] + alpha[Ti - 1, len(ext) - 2]
            total += -np.log(P + 1e-12)
        return total / B

    def backward(self):
        """Gradient dL/dlogits, shape (T, B, C)."""
        T, B, C = self.logits.shape
        dY = np.zeros((T, B, C))
        for b in range(B):
            ext, skip, Ti = self.extended_list[b]
            gamma = self.gammas[b]
            lg = self.logits[:Ti, b, :]
            for s, sym in enumerate(ext):
                dY[:Ti, b, sym] -= gamma[:, s] / (lg[:, sym] + 1e-12)
        return dY / B
