"""1-D batch normalisation for MyTorch (CMU 11-785 HW1P1).

Input ``Z`` has shape (N, C).  Learnable ``BW`` (gamma) and ``Bb`` (beta) each
have shape (1, C).  Running statistics are tracked for evaluation mode.
"""

import numpy as np


class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha            # running-stat momentum
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))   # gamma
        self.Bb = np.zeros((1, num_features))  # beta
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # running estimates used at eval time
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """Apply batchnorm.  ``eval=True`` uses running statistics."""
        self.Z = Z
        self.N = Z.shape[0]

        if eval:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            return self.BW * self.NZ + self.Bb

        self.M = np.mean(Z, axis=0, keepdims=True)                 # (1, C)
        self.V = np.var(Z, axis=0, keepdims=True)                  # (1, C)
        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)        # normalised
        self.BZ = self.BW * self.NZ + self.Bb                      # scaled+shifted

        # update running statistics
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW
        istd = 1.0 / np.sqrt(self.V + self.eps)
        Zmu = self.Z - self.M

        dLdV = -0.5 * np.sum(dLdNZ * Zmu * (istd ** 3), axis=0, keepdims=True)
        dLdM = -np.sum(dLdNZ * istd, axis=0, keepdims=True) \
            - 2.0 / self.N * dLdV * np.sum(Zmu, axis=0, keepdims=True)

        dLdZ = dLdNZ * istd + dLdV * (2.0 / self.N) * Zmu + dLdM / self.N
        return dLdZ
