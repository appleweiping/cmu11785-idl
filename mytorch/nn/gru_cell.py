"""GRU cell for MyTorch (CMU 11-785 HW3P1).

Single-example (unbatched) GRU cell matching the HW3P1 spec / PyTorch GRUCell:

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)     reset gate
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)     update gate
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))  candidate
    h' = (1 - z) * n + z * h

The backward pass returns ``(dx, dh)`` and accumulates parameter gradients.
"""

import numpy as np

from .activation import Sigmoid, Tanh


class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h, d = hidden_size, input_size

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)
        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)
        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()
        self.zero_grad()

    def zero_grad(self):
        for name in ["Wrx", "Wzx", "Wnx", "Wrh", "Wzh", "Wnh",
                     "brx", "bzx", "bnx", "brh", "bzh", "bnh"]:
            setattr(self, "d" + name, np.zeros_like(getattr(self, name)))

    def forward(self, x, h_prev_t):
        """x: (d,), h_prev_t: (h,) -> h_t: (h,)."""
        self.x = x
        self.hidden = h_prev_t

        self.r = self.r_act.forward(
            self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(
            self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act.forward(
            self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))

        h_t = (1 - self.z) * self.n + self.z * h_prev_t
        return h_t

    def backward(self, delta):
        """delta: dL/dh_t, shape (h,). Returns (dx (d,), dh_prev (h,))."""
        x = self.x.reshape(-1, 1)          # (d,1)
        h = self.hidden.reshape(-1, 1)     # (h,1)

        # h_t = (1-z)*n + z*h
        dn = delta * (1 - self.z)
        dz = delta * (self.hidden - self.n)
        dh_prev = delta * self.z

        # n = tanh(...)
        dn_pre = dn * (1 - self.n ** 2)
        self.dWnx += np.outer(dn_pre, x)
        self.dbnx += dn_pre
        # inner term r * (Wnh h + bnh)
        Wnh_h = self.Wnh @ self.hidden + self.bnh
        dr = dn_pre * Wnh_h
        d_inner = dn_pre * self.r
        self.dWnh += np.outer(d_inner, h)
        self.dbnh += d_inner
        dh_prev += self.Wnh.T @ d_inner

        # z = sigmoid(...)
        dz_pre = dz * self.z * (1 - self.z)
        self.dWzx += np.outer(dz_pre, x)
        self.dbzx += dz_pre
        self.dWzh += np.outer(dz_pre, h)
        self.dbzh += dz_pre

        # r = sigmoid(...)
        dr_pre = dr * self.r * (1 - self.r)
        self.dWrx += np.outer(dr_pre, x)
        self.dbrx += dr_pre
        self.dWrh += np.outer(dr_pre, h)
        self.dbrh += dr_pre

        dx = (self.Wnx.T @ dn_pre + self.Wzx.T @ dz_pre + self.Wrx.T @ dr_pre)
        dh_prev += self.Wzh.T @ dz_pre + self.Wrh.T @ dr_pre
        return dx.reshape(1, -1), dh_prev.reshape(1, -1)
