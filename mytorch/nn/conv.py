"""Convolution layers for MyTorch (CMU 11-785 HW2P1).

Implements stride-1 Conv1d / Conv2d cores (explicit forward + backward), the
strided wrappers built from resampling, and a Flatten layer.  Weight tensors
follow the PyTorch layout ``(out_channels, in_channels, *kernel)``.

Backward uses the standard results:
  * dLdW = correlate(input, dLdZ)
  * dLdA = full-convolution(dLdZ, flipped W)
"""

import numpy as np

from .resampling import Downsample1d, Downsample2d


# ============================================================ 1-D convolution
class Conv1d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)
        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        # A: (N, C_in, W_in) -> Z: (N, C_out, W_out)
        self.A = A
        N, C_in, W_in = A.shape
        W_out = W_in - self.kernel_size + 1
        Z = np.zeros((N, self.out_channels, W_out))
        for j in range(W_out):
            seg = A[:, :, j:j + self.kernel_size]          # (N, C_in, K)
            # (N, C_in, K) x (C_out, C_in, K) -> (N, C_out)
            Z[:, :, j] = np.tensordot(seg, self.W, axes=([1, 2], [1, 2]))
        Z += self.b.reshape(1, -1, 1)
        return Z

    def backward(self, dLdZ):
        N, C_out, W_out = dLdZ.shape
        K = self.kernel_size

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        self.dLdW = np.zeros(self.W.shape)
        for j in range(K):
            seg = self.A[:, :, j:j + W_out]                # (N, C_in, W_out)
            # sum over N and W_out: (C_out, C_in)
            self.dLdW[:, :, j] = np.tensordot(dLdZ, seg, axes=([0, 2], [0, 2]))

        # dLdA via full convolution with flipped kernel
        pad = K - 1
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad)))
        W_flip = self.W[:, :, ::-1]                        # (C_out, C_in, K)
        W_in = self.A.shape[2]
        dLdA = np.zeros((N, self.in_channels, W_in))
        for j in range(W_in):
            seg = dLdZ_pad[:, :, j:j + K]                  # (N, C_out, K)
            # (N,C_out,K) x (C_out,C_in,K) -> (N, C_in)
            dLdA[:, :, j] = np.tensordot(seg, W_flip, axes=([1, 2], [0, 2]))
        return dLdA


class Conv1d:
    """Strided Conv1d = stride-1 conv followed by downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        if self.pad:
            A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))
        Z = self.conv1d_stride1.forward(A)
        return self.downsample1d.forward(Z)

    def backward(self, dLdZ):
        dLdZ = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdZ)
        if self.pad:
            dLdA = dLdA[:, :, self.pad:-self.pad]
        return dLdA

    # expose weights for convenience
    @property
    def W(self):
        return self.conv1d_stride1.W

    @property
    def dLdW(self):
        return self.conv1d_stride1.dLdW


# ============================================================ 2-D convolution
class Conv2d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels, in_channels, kernel_size, kernel_size)
        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        # A: (N, C_in, H, W) -> Z: (N, C_out, H_out, W_out)
        self.A = A
        N, C_in, H, W = A.shape
        K = self.kernel_size
        H_out, W_out = H - K + 1, W - K + 1
        Z = np.zeros((N, self.out_channels, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                seg = A[:, :, i:i + K, j:j + K]            # (N, C_in, K, K)
                Z[:, :, i, j] = np.tensordot(
                    seg, self.W, axes=([1, 2, 3], [1, 2, 3]))
        Z += self.b.reshape(1, -1, 1, 1)
        return Z

    def backward(self, dLdZ):
        N, C_out, H_out, W_out = dLdZ.shape
        K = self.kernel_size

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        self.dLdW = np.zeros(self.W.shape)
        for i in range(K):
            for j in range(K):
                seg = self.A[:, :, i:i + H_out, j:j + W_out]   # (N,C_in,H_out,W_out)
                self.dLdW[:, :, i, j] = np.tensordot(
                    dLdZ, seg, axes=([0, 2, 3], [0, 2, 3]))

        pad = K - 1
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        W_flip = self.W[:, :, ::-1, ::-1]                  # (C_out, C_in, K, K)
        H, W = self.A.shape[2], self.A.shape[3]
        dLdA = np.zeros((N, self.in_channels, H, W))
        for i in range(H):
            for j in range(W):
                seg = dLdZ_pad[:, :, i:i + K, j:j + K]      # (N, C_out, K, K)
                dLdA[:, :, i, j] = np.tensordot(
                    seg, W_flip, axes=([1, 2, 3], [0, 2, 3]))
        return dLdA


class Conv2d:
    """Strided Conv2d = stride-1 conv followed by 2-D downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        if self.pad:
            A = np.pad(A, ((0, 0), (0, 0),
                           (self.pad, self.pad), (self.pad, self.pad)))
        Z = self.conv2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.conv2d_stride1.backward(dLdZ)
        if self.pad:
            dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dLdA

    @property
    def W(self):
        return self.conv2d_stride1.W

    @property
    def dLdW(self):
        return self.conv2d_stride1.dLdW


# ==================================================================== flatten
class Flatten:
    """Flatten (N, C, *) -> (N, C*prod(*)) with a shape-restoring backward."""

    def forward(self, A):
        self.shape = A.shape
        return A.reshape(A.shape[0], -1)

    def backward(self, dLdZ):
        return dLdZ.reshape(self.shape)
