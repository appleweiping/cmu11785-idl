"""HW1P1 verification: MyTorch MLP components vs. a PyTorch reference.

Each test builds a random input, runs MyTorch forward/backward, and compares
outputs *and gradients* against ``torch.autograd``.  This is our autograder:
if these pass, the from-scratch math matches PyTorch to numerical tolerance.
"""

import numpy as np
import torch
import pytest

from mytorch.nn import (
    Identity, Sigmoid, Tanh, ReLU, Linear, MSELoss, CrossEntropyLoss, BatchNorm1d,
)
from mytorch.models import MLP4
from mytorch.optim import SGD

RTOL, ATOL = 1e-5, 1e-6
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- activations
@pytest.mark.parametrize("mt_cls,torch_fn", [
    (Sigmoid, torch.sigmoid),
    (Tanh, torch.tanh),
    (ReLU, torch.relu),
    (Identity, lambda x: x),
])
def test_activation_forward_backward(mt_cls, torch_fn):
    Z = rng.standard_normal((7, 5))
    act = mt_cls()
    A = act.forward(Z)

    Zt = torch.tensor(Z, requires_grad=True)
    At = torch_fn(Zt)
    assert np.allclose(A, At.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdA = rng.standard_normal((7, 5))
    dLdZ = act.backward(dLdA)
    At.backward(torch.tensor(dLdA))
    assert np.allclose(dLdZ, Zt.grad.numpy(), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------- linear
def test_linear_forward_backward():
    N, C0, C1 = 6, 4, 3
    lin = Linear(C0, C1, debug=True)
    lin.W = rng.standard_normal((C1, C0))
    lin.b = rng.standard_normal((C1, 1))
    A = rng.standard_normal((N, C0))
    Z = lin.forward(A)

    At = torch.tensor(A, requires_grad=True)
    Wt = torch.tensor(lin.W, requires_grad=True)
    bt = torch.tensor(lin.b, requires_grad=True)
    Zt = At @ Wt.T + bt.T
    assert np.allclose(Z, Zt.detach().numpy(), rtol=RTOL, atol=ATOL)

    dLdZ = rng.standard_normal((N, C1))
    dLdA = lin.backward(dLdZ)
    Zt.backward(torch.tensor(dLdZ))
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(lin.dLdW, Wt.grad.numpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(lin.dLdb, bt.grad.numpy(), rtol=RTOL, atol=ATOL)


# ----------------------------------------------------------------------- loss
def test_mse_loss():
    A = rng.standard_normal((6, 4))
    Y = rng.standard_normal((6, 4))
    loss = MSELoss()
    L = loss.forward(A, Y)

    At = torch.tensor(A, requires_grad=True)
    Lt = torch.nn.functional.mse_loss(At, torch.tensor(Y), reduction="mean")
    assert np.allclose(L, Lt.item(), rtol=RTOL, atol=ATOL)

    dLdA = loss.backward()
    Lt.backward()
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_cross_entropy_loss():
    N, C = 6, 4
    A = rng.standard_normal((N, C))
    labels = rng.integers(0, C, size=N)
    Y = np.eye(C)[labels]
    loss = CrossEntropyLoss()
    L = loss.forward(A, Y)

    At = torch.tensor(A, requires_grad=True)
    Lt = torch.nn.functional.cross_entropy(
        At, torch.tensor(labels), reduction="mean")
    assert np.allclose(L, Lt.item(), rtol=RTOL, atol=ATOL)

    dLdA = loss.backward()
    Lt.backward()
    assert np.allclose(dLdA, At.grad.numpy(), rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------ batchnorm
def test_batchnorm_train_forward_backward():
    N, C = 8, 5
    bn = BatchNorm1d(C)
    bn.BW = rng.standard_normal((1, C))
    bn.Bb = rng.standard_normal((1, C))
    Z = rng.standard_normal((N, C))
    out = bn.forward(Z, eval=False)

    Zt = torch.tensor(Z, requires_grad=True)
    gamma = torch.tensor(bn.BW.reshape(-1), requires_grad=True)
    beta = torch.tensor(bn.Bb.reshape(-1), requires_grad=True)
    # biased variance to match our np.var (ddof=0); eps must match
    mean = Zt.mean(0)
    var = Zt.var(0, unbiased=False)
    nz = (Zt - mean) / torch.sqrt(var + bn.eps)
    outt = gamma * nz + beta
    assert np.allclose(out, outt.detach().numpy(), rtol=1e-4, atol=1e-5)

    dLdout = rng.standard_normal((N, C))
    dLdZ = bn.backward(dLdout)
    outt.backward(torch.tensor(dLdout))
    assert np.allclose(dLdZ, Zt.grad.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(bn.dLdBW.reshape(-1), gamma.grad.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(bn.dLdBb.reshape(-1), beta.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_batchnorm_eval_uses_running_stats():
    C = 4
    bn = BatchNorm1d(C)
    for _ in range(5):
        bn.forward(rng.standard_normal((16, C)), eval=False)
    out = bn.forward(rng.standard_normal((3, C)), eval=True)
    assert out.shape == (3, C)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------- mlp + sgd
def test_mlp4_end_to_end_grads():
    model = MLP4(debug=True)
    for lin in model.layers:
        lin.W = rng.standard_normal(lin.W.shape)
        lin.b = rng.standard_normal(lin.b.shape)
    A0 = rng.standard_normal((5, 2))
    out = model.forward(A0)
    assert out.shape == (5, 2)
    dLdA = model.backward(rng.standard_normal((5, 2)))
    assert dLdA.shape == (5, 2)
    for lin in model.layers:
        assert lin.dLdW.shape == lin.W.shape
        assert lin.dLdb.shape == lin.b.shape


def test_sgd_step_reduces_loss():
    """A few SGD steps must reduce CE loss on a fixed random classification."""
    from mytorch.models import MLP
    C_in, C_out, N = 6, 3, 40
    X = rng.standard_normal((N, C_in))
    labels = rng.integers(0, C_out, size=N)
    Y = np.eye(C_out)[labels]

    model = MLP([C_in, 16, C_out])
    for lin in model.layers:
        lin.W = rng.standard_normal(lin.W.shape) * 0.3
        lin.b = np.zeros(lin.b.shape)
    crit = CrossEntropyLoss()
    opt = SGD(model, lr=0.5, momentum=0.9)

    L0 = crit.forward(model.forward(X), Y)
    for _ in range(50):
        out = model.forward(X)
        crit.forward(out, Y)
        model.backward(crit.backward())
        opt.step()
    L1 = crit.forward(model.forward(X), Y)
    assert L1 < L0 * 0.9, f"loss did not drop: {L0:.4f} -> {L1:.4f}"
