"""Attention for MyTorch (CMU 11-785 HW4P1).

Scaled dot-product attention with an explicit backward pass, plus a multi-head
wrapper.  Shapes use ``(B, L, D)`` (batch, sequence length, model dim).  An
optional additive causal/pad mask of shape ``(B, L, L)`` (with ``-inf`` for
disallowed positions) may be supplied.
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    r"""Attention(Q,K,V) = softmax(Q Kᵀ / √d_k) V."""

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: (B, L, d_k)
        self.Q, self.K, self.V = Q, K, V
        self.d_k = Q.shape[-1]
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        self.attn = _softmax(scores, axis=-1)          # (B, L, L)
        out = np.matmul(self.attn, V)                  # (B, L, d_k)
        return out

    def backward(self, dOut):
        # dOut: (B, L, d_k)
        dAttn = np.matmul(dOut, np.swapaxes(self.V, -1, -2))   # (B, L, L)
        dV = np.matmul(np.swapaxes(self.attn, -1, -2), dOut)

        # softmax backward per row: dS = A * (dAttn - sum(dAttn*A))
        s = np.sum(dAttn * self.attn, axis=-1, keepdims=True)
        dScores = self.attn * (dAttn - s)
        dScores /= np.sqrt(self.d_k)

        dQ = np.matmul(dScores, self.K)
        dK = np.matmul(np.swapaxes(dScores, -1, -2), self.Q)
        return dQ, dK, dV


class MultiHeadAttention:
    """Multi-head attention with learnable Q/K/V/output projections."""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        r = np.sqrt(1.0 / d_model)
        self.W_q = np.random.uniform(-r, r, (d_model, d_model))
        self.W_k = np.random.uniform(-r, r, (d_model, d_model))
        self.W_v = np.random.uniform(-r, r, (d_model, d_model))
        self.W_o = np.random.uniform(-r, r, (d_model, d_model))
        self.attn = ScaledDotProductAttention()

    def _split(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):
        B, h, L, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, h * d_k)

    def forward(self, x, mask=None):
        self.x = x
        B, L, _ = x.shape
        Q = self._split(x @ self.W_q)      # (B, h, L, d_k)
        K = self._split(x @ self.W_k)
        V = self._split(x @ self.W_v)
        m = None if mask is None else mask[:, None, :, :]
        ctx = self.attn.forward(Q, K, V, m)  # (B, h, L, d_k)
        self.ctx = self._merge(ctx)          # (B, L, d_model)
        out = self.ctx @ self.W_o
        return out
