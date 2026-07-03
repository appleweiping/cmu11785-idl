"""Loss functions for MyTorch (CMU 11-785 HW1P1).

Both losses take model output ``A`` (N, C) and target ``Y`` (N, C) and return a
scalar mean loss.  ``backward`` returns ``dLdA`` of shape (N, C).
"""

import numpy as np


class MSELoss:
    r"""Mean-squared error averaged over batch and features.

    :math:`L = \frac{1}{N C} \sum_{n,c} (A_{nc} - Y_{nc})^2`.
    """

    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape
        se = (A - Y) * (A - Y)
        # sum of squared error then normalise by N*C
        sse = np.ones((1, self.N)) @ se @ np.ones((self.C, 1))
        mse = sse / (self.N * self.C)
        return mse[0, 0]

    def backward(self):
        dLdA = 2.0 * (self.A - self.Y) / (self.N * self.C)
        return dLdA


class CrossEntropyLoss:
    r"""Softmax + negative-log-likelihood, averaged over the batch.

    Forward computes row-wise softmax then
    :math:`L = -\frac{1}{N}\sum_n \sum_c Y_{nc}\log \mathrm{softmax}(A)_{nc}`.
    Targets ``Y`` are one-hot rows of shape (N, C).
    """

    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape

        # numerically-stable row-wise softmax
        Z = A - np.max(A, axis=1, keepdims=True)
        expZ = np.exp(Z)
        self.softmax = expZ / np.sum(expZ, axis=1, keepdims=True)

        crossentropy = -np.sum(Y * np.log(self.softmax + 1e-12), axis=1, keepdims=True)
        L = np.sum(crossentropy) / self.N
        return L

    def backward(self):
        # d/dA of softmax-CE (mean over batch): (softmax - Y) / N
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA
