"""Linear (fully-connected) layer for MyTorch (CMU 11-785 HW1P1).

Forward (batched):  ``Z = A @ W.T + b.T``  with

    A    : (N, C0)   input
    W    : (C1, C0)  weights
    b    : (C1, 1)   bias
    Z    : (N, C1)   output

Backward stores ``dLdW`` (C1, C0), ``dLdb`` (C1, 1) and returns ``dLdA``.
"""

import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))
        self.debug = debug

    def forward(self, A):
        self.A = A
        self.N = A.shape[0]
        self.ones = np.ones((self.N, 1))
        Z = self.A @ self.W.T + self.ones @ self.b.T
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ @ self.W                 # (N, C0)
        self.dLdW = dLdZ.T @ self.A          # (C1, C0)
        self.dLdb = dLdZ.T @ self.ones       # (C1, 1)

        if self.debug:
            self.dLdA = dLdA
        return dLdA
