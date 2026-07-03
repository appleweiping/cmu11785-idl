"""Elman RNN cell for MyTorch (CMU 11-785 HW3P1).

    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)

Weight layout follows PyTorch's ``nn.RNNCell``.  ``backward`` accumulates
parameter gradients across timesteps (caller resets them each sequence).
"""

import numpy as np

from .activation import Tanh


class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = Tanh()

        h, d = hidden_size, input_size
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_ih = np.zeros_like(self.b_ih)
        self.db_hh = np.zeros_like(self.b_hh)

    def zero_grad(self):
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_ih = np.zeros_like(self.b_ih)
        self.db_hh = np.zeros_like(self.b_hh)

    def forward(self, x, h_prev_t):
        # x: (N, d), h_prev_t: (N, h) -> h_t: (N, h)
        z = (x @ self.W_ih.T + self.b_ih) + (h_prev_t @ self.W_hh.T + self.b_hh)
        h_t = self.activation.forward(z)
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """One BPTT step.

        Args:
            delta:     dL/dh_t incoming, shape (N, h)
            h_t:       output of this cell, (N, h) -- used for tanh'(z)=1-h_t^2
            h_prev_l:  input x_t to this cell (previous layer), (N, d)
            h_prev_t:  previous-timestep hidden state, (N, h)
        Returns:
            dx  (dL/dx_t)      shape (N, d)
            dh_prev_t (dL/dh_{t-1}) shape (N, h)
        """
        dz = (1 - h_t * h_t) * delta                 # (N, h)

        batch = delta.shape[0]
        self.dW_ih += dz.T @ h_prev_l / batch
        self.dW_hh += dz.T @ h_prev_t / batch
        self.db_ih += dz.sum(0) / batch
        self.db_hh += dz.sum(0) / batch

        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh
        return dx, dh_prev_t
