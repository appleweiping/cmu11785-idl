"""HW2P1 verification: MyTorch CNN components vs. PyTorch reference."""

import numpy as np
import torch
import pytest

from mytorch.nn import (
    Conv1d_stride1, Conv1d, Conv2d_stride1, Conv2d, Flatten,
    MaxPool2d, MeanPool2d,
    Upsample1d, Downsample1d, Upsample2d, Downsample2d,
)

RTOL, ATOL = 1e-4, 1e-5
rng = np.random.default_rng(1)


# --------------------------------------------------------------- resampling
def test_upsample1d_downsample1d_inverse():
    A = rng.standard_normal((2, 3, 5))
    up = Upsample1d(3)
    Z = up.forward(A)
    assert Z.shape == (2, 3, 3 * (5 - 1) + 1)
    # downsampling the upsampled signal recovers it
    down = Downsample1d(3)
    assert np.allclose(down.forward(Z), A)


# ------------------------------------------------------------------ conv1d
def test_conv1d_stride1_vs_torch():
    N, Ci, Co, K, Win = 2, 3, 4, 3, 10
    conv = Conv1d_stride1(Ci, Co, K)
    conv.W = rng.standard_normal(conv.W.shape)
    conv.b = rng.standard_normal(conv.b.shape)
    A = rng.standard_normal((N, Ci, Win))
    Z = conv.forward(A)

    At = torch.tensor(A, requires_grad=True)
    tconv = torch.nn.Conv1d(Ci, Co, K, bias=True)
    tconv.weight = torch.nn.Parameter(torch.tensor(conv.W))
    tconv.bias = torch.nn.Parameter(torch.tensor(conv.b))
    Zt = tconv(At)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = conv.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(conv.dLdW, tconv.weight.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(conv.dLdb, tconv.bias.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_conv1d_strided_vs_torch():
    N, Ci, Co, K, Win, S, P = 2, 3, 5, 3, 12, 2, 1
    conv = Conv1d(Ci, Co, K, stride=S, padding=P)
    conv.conv1d_stride1.W = rng.standard_normal(conv.conv1d_stride1.W.shape)
    conv.conv1d_stride1.b = rng.standard_normal(conv.conv1d_stride1.b.shape)
    A = rng.standard_normal((N, Ci, Win))
    Z = conv.forward(A)

    At = torch.tensor(A, requires_grad=True)
    tconv = torch.nn.Conv1d(Ci, Co, K, stride=S, padding=P, bias=True)
    tconv.weight = torch.nn.Parameter(torch.tensor(conv.conv1d_stride1.W))
    tconv.bias = torch.nn.Parameter(torch.tensor(conv.conv1d_stride1.b))
    Zt = tconv(At)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = conv.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------ conv2d
def test_conv2d_stride1_vs_torch():
    N, Ci, Co, K, H, W = 2, 3, 4, 3, 8, 9
    conv = Conv2d_stride1(Ci, Co, K)
    conv.W = rng.standard_normal(conv.W.shape)
    conv.b = rng.standard_normal(conv.b.shape)
    A = rng.standard_normal((N, Ci, H, W))
    Z = conv.forward(A)

    At = torch.tensor(A, requires_grad=True)
    tconv = torch.nn.Conv2d(Ci, Co, K, bias=True)
    tconv.weight = torch.nn.Parameter(torch.tensor(conv.W))
    tconv.bias = torch.nn.Parameter(torch.tensor(conv.b))
    Zt = tconv(At)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = conv.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(conv.dLdW, tconv.weight.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(conv.dLdb, tconv.bias.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_conv2d_strided_padded_vs_torch():
    N, Ci, Co, K, H, W, S, P = 2, 3, 4, 3, 10, 10, 2, 1
    conv = Conv2d(Ci, Co, K, stride=S, padding=P)
    conv.conv2d_stride1.W = rng.standard_normal(conv.conv2d_stride1.W.shape)
    conv.conv2d_stride1.b = rng.standard_normal(conv.conv2d_stride1.b.shape)
    A = rng.standard_normal((N, Ci, H, W))
    Z = conv.forward(A)

    At = torch.tensor(A, requires_grad=True)
    tconv = torch.nn.Conv2d(Ci, Co, K, stride=S, padding=P, bias=True)
    tconv.weight = torch.nn.Parameter(torch.tensor(conv.conv2d_stride1.W))
    tconv.bias = torch.nn.Parameter(torch.tensor(conv.conv2d_stride1.b))
    Zt = tconv(At)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = conv.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------- pooling
def test_maxpool2d_vs_torch():
    N, C, H, W, K, S = 2, 3, 8, 8, 2, 2
    pool = MaxPool2d(K, S)
    A = rng.standard_normal((N, C, H, W))
    Z = pool.forward(A)

    At = torch.tensor(A, requires_grad=True)
    Zt = torch.nn.functional.max_pool2d(At, K, stride=S)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = pool.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_meanpool2d_vs_torch():
    N, C, H, W, K, S = 2, 3, 8, 8, 2, 2
    pool = MeanPool2d(K, S)
    A = rng.standard_normal((N, C, H, W))
    Z = pool.forward(A)

    At = torch.tensor(A, requires_grad=True)
    Zt = torch.nn.functional.avg_pool2d(At, K, stride=S)
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal(Z.shape)
    dLdA = pool.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_flatten_roundtrip():
    A = rng.standard_normal((4, 3, 5))
    f = Flatten()
    Z = f.forward(A)
    assert Z.shape == (4, 15)
    assert np.allclose(f.backward(Z), A)
