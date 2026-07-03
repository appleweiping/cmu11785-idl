"""Up/down-sampling primitives for MyTorch CNNs (CMU 11-785 HW2P1).

Strided convolution is decomposed into a stride-1 convolution followed by a
downsampling step (and the transpose for upsampling), which keeps the conv
kernels simple.  These operate on the last (spatial) axis/axes.
"""

import numpy as np


class Upsample1d:
    def __init__(self, upsampling_factor):
        self.k = upsampling_factor

    def forward(self, A):
        # A: (N, C, W_in) -> (N, C, k*(W_in-1)+1)
        N, C, W_in = A.shape
        self.W_in = W_in
        W_out = self.k * (W_in - 1) + 1
        Z = np.zeros((N, C, W_out), dtype=A.dtype)
        Z[:, :, ::self.k] = A
        return Z

    def backward(self, dLdZ):
        return dLdZ[:, :, ::self.k]


class Downsample1d:
    def __init__(self, downsampling_factor):
        self.k = downsampling_factor

    def forward(self, A):
        # A: (N, C, W_in) -> (N, C, ceil(W_in/k))
        self.W_in = A.shape[2]
        return A[:, :, ::self.k]

    def backward(self, dLdZ):
        N, C, W_out = dLdZ.shape
        dLdA = np.zeros((N, C, self.W_in), dtype=dLdZ.dtype)
        dLdA[:, :, ::self.k] = dLdZ
        return dLdA


class Upsample2d:
    def __init__(self, upsampling_factor):
        self.k = upsampling_factor

    def forward(self, A):
        N, C, H, W = A.shape
        self.H_in, self.W_in = H, W
        Z = np.zeros((N, C, self.k * (H - 1) + 1, self.k * (W - 1) + 1), dtype=A.dtype)
        Z[:, :, ::self.k, ::self.k] = A
        return Z

    def backward(self, dLdZ):
        return dLdZ[:, :, ::self.k, ::self.k]


class Downsample2d:
    def __init__(self, downsampling_factor):
        self.k = downsampling_factor

    def forward(self, A):
        N, C, H, W = A.shape
        self.H_in, self.W_in = H, W
        return A[:, :, ::self.k, ::self.k]

    def backward(self, dLdZ):
        N, C, H_out, W_out = dLdZ.shape
        dLdA = np.zeros((N, C, self.H_in, self.W_in), dtype=dLdZ.dtype)
        dLdA[:, :, ::self.k, ::self.k] = dLdZ
        return dLdA
