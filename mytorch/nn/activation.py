"""Activation functions for MyTorch (CMU 11-785 HW1P1).

Every activation follows the MyTorch convention:

    A = activation.forward(Z)          # forward, caches A
    dLdZ = activation.backward(dLdA)   # backward, uses cached A

Shapes are ``(N, C)`` with N the batch size and C the number of features.
"""

import numpy as np


class Identity:
    """Identity activation: ``A = Z``."""

    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        # dA/dZ = 1
        return dLdA * np.ones(self.A.shape, dtype=self.A.dtype)


class Sigmoid:
    r"""Logistic sigmoid: :math:`A = \frac{1}{1 + e^{-Z}}`."""

    def forward(self, Z):
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        # dA/dZ = A * (1 - A)
        dAdZ = self.A - self.A * self.A
        return dLdA * dAdZ


class Tanh:
    r"""Hyperbolic tangent: :math:`A = \tanh(Z)`."""

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        # dA/dZ = 1 - A^2
        dAdZ = 1.0 - self.A * self.A
        return dLdA * dAdZ


class ReLU:
    """Rectified Linear Unit: ``A = max(0, Z)``."""

    def forward(self, Z):
        self.A = np.maximum(0.0, Z)
        return self.A

    def backward(self, dLdA):
        # dA/dZ = 1 where Z > 0 else 0; A > 0 iff Z > 0
        dAdZ = (self.A > 0).astype(self.A.dtype)
        return dLdA * dAdZ


class GELU:
    r"""Gaussian Error Linear Unit (exact, erf-based).

    :math:`A = \tfrac{1}{2} Z \left(1 + \operatorname{erf}(Z/\sqrt 2)\right)`.
    Used by HW4 attention / transformer blocks.
    """

    def forward(self, Z):
        from scipy.special import erf  # local import; only GELU needs scipy

        self.Z = Z
        self.A = 0.5 * Z * (1.0 + erf(Z / np.sqrt(2.0)))
        return self.A

    def backward(self, dLdA):
        from scipy.special import erf

        Z = self.Z
        cdf = 0.5 * (1.0 + erf(Z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * Z * Z) / np.sqrt(2.0 * np.pi)
        dAdZ = cdf + Z * pdf
        return dLdA * dAdZ


class Softmax:
    """Row-wise softmax activation (used standalone and inside CE loss).

    Forward maps ``(N, C) -> (N, C)``; backward applies the full Jacobian
    per row.
    """

    def forward(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shift)
        self.A = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        N, C = self.A.shape
        dLdZ = np.zeros((N, C), dtype=self.A.dtype)
        for i in range(N):
            a = self.A[i].reshape(-1, 1)          # (C, 1)
            J = np.diagflat(a) - a @ a.T           # (C, C) Jacobian
            dLdZ[i] = (dLdA[i].reshape(1, -1) @ J).reshape(-1)
        return dLdZ
