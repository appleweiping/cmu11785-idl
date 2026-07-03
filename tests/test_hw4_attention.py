"""HW4P1 verification: attention vs PyTorch reference + gradchecks."""

import numpy as np
import torch
import pytest

from mytorch.nn import ScaledDotProductAttention, MultiHeadAttention

RTOL, ATOL = 1e-5, 1e-6
rng = np.random.default_rng(4)


def test_scaled_dot_product_forward_vs_torch():
    B, L, d_k = 2, 5, 4
    Q = rng.standard_normal((B, L, d_k))
    K = rng.standard_normal((B, L, d_k))
    V = rng.standard_normal((B, L, d_k))
    out = ScaledDotProductAttention().forward(Q, K, V)

    Qt, Kt, Vt = (torch.tensor(t) for t in (Q, K, V))
    outt = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
    assert np.allclose(out, outt.numpy(), rtol=RTOL, atol=ATOL)


def test_scaled_dot_product_causal_mask():
    B, L, d_k = 1, 4, 3
    Q = rng.standard_normal((B, L, d_k))
    K = rng.standard_normal((B, L, d_k))
    V = rng.standard_normal((B, L, d_k))
    mask = np.triu(np.full((L, L), -np.inf), k=1)[None, :, :]
    out = ScaledDotProductAttention().forward(Q, K, V, mask)

    Qt, Kt, Vt = (torch.tensor(t) for t in (Q, K, V))
    outt = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True)
    assert np.allclose(out, outt.numpy(), rtol=RTOL, atol=ATOL)


def test_scaled_dot_product_backward_gradcheck():
    B, L, d_k = 1, 4, 3
    Q = rng.standard_normal((B, L, d_k))
    K = rng.standard_normal((B, L, d_k))
    V = rng.standard_normal((B, L, d_k))
    attn = ScaledDotProductAttention()
    out = attn.forward(Q, K, V)
    g = rng.standard_normal(out.shape)
    dQ, dK, dV = attn.backward(g)

    # compare to torch autograd
    Qt = torch.tensor(Q, requires_grad=True)
    Kt = torch.tensor(K, requires_grad=True)
    Vt = torch.tensor(V, requires_grad=True)
    outt = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
    outt.backward(torch.tensor(g))
    assert np.allclose(dQ, Qt.grad.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(dK, Kt.grad.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(dV, Vt.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_multihead_forward_vs_torch():
    B, L, d_model, h = 2, 6, 8, 2
    mha = MultiHeadAttention(d_model, h)
    x = rng.standard_normal((B, L, d_model))
    out = mha.forward(x)

    tmha = torch.nn.MultiheadAttention(d_model, h, bias=False, batch_first=True)
    # torch packs in_proj as [Wq; Wk; Wv] each (d_model, d_model), rows=out
    in_proj = np.vstack([mha.W_q.T, mha.W_k.T, mha.W_v.T])
    tmha.in_proj_weight = torch.nn.Parameter(torch.tensor(in_proj))
    tmha.out_proj.weight = torch.nn.Parameter(torch.tensor(mha.W_o.T))
    xt = torch.tensor(x)
    outt, _ = tmha(xt, xt, xt)
    assert np.allclose(out, outt.detach().numpy(), rtol=1e-4, atol=1e-5)
