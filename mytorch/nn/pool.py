"""2-D pooling layers for MyTorch (CMU 11-785 HW2P1).

Max and mean pooling, each as a stride-1 core plus a strided wrapper built from
:class:`Downsample2d`.  Operates on ``(N, C, H, W)`` tensors.
"""

import numpy as np

from .resampling import Downsample2d


class MaxPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        H_out, W_out = H - K + 1, W - K + 1
        Z = np.zeros((N, C, H_out, W_out))
        # store argmax positions for backward
        self.argmax = np.zeros((N, C, H_out, W_out, 2), dtype=int)
        for i in range(H_out):
            for j in range(W_out):
                win = A[:, :, i:i + K, j:j + K].reshape(N, C, -1)
                idx = np.argmax(win, axis=2)
                Z[:, :, i, j] = np.max(win, axis=2)
                self.argmax[:, :, i, j, 0] = idx // K
                self.argmax[:, :, i, j, 1] = idx % K
        return Z

    def backward(self, dLdZ):
        N, C, H_out, W_out = dLdZ.shape
        K = self.kernel
        dLdA = np.zeros(self.A.shape)
        for i in range(H_out):
            for j in range(W_out):
                di = self.argmax[:, :, i, j, 0]
                dj = self.argmax[:, :, i, j, 1]
                for n in range(N):
                    for c in range(C):
                        dLdA[n, c, i + di[n, c], j + dj[n, c]] += dLdZ[n, c, i, j]
        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        H_out, W_out = H - K + 1, W - K + 1
        Z = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                Z[:, :, i, j] = np.mean(A[:, :, i:i + K, j:j + K], axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        N, C, H_out, W_out = dLdZ.shape
        K = self.kernel
        dLdA = np.zeros(self.A.shape)
        for i in range(H_out):
            for j in range(W_out):
                dLdA[:, :, i:i + K, j:j + K] += (
                    dLdZ[:, :, i, j][:, :, None, None] / (K * K))
        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.core = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        return self.downsample2d.forward(self.core.forward(A))

    def backward(self, dLdZ):
        return self.core.backward(self.downsample2d.backward(dLdZ))


class MeanPool2d:
    def __init__(self, kernel, stride):
        self.core = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        return self.downsample2d.forward(self.core.forward(A))

    def backward(self, dLdZ):
        return self.core.backward(self.downsample2d.backward(dLdZ))
