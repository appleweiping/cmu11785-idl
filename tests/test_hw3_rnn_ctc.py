"""HW3P1 verification: RNN/GRU cells, CTC loss, and CTC decoders vs PyTorch."""

import numpy as np
import torch
import pytest

from mytorch.nn import RNNCell, GRUCell, CTCLoss, GreedySearchDecoder, BeamSearchDecoder

RTOL, ATOL = 1e-5, 1e-6
rng = np.random.default_rng(3)


# --------------------------------------------------------------- RNN cell fwd
def test_rnncell_forward_vs_torch():
    N, d, h = 4, 5, 6
    cell = RNNCell(d, h)
    cell.W_ih = rng.standard_normal((h, d))
    cell.W_hh = rng.standard_normal((h, h))
    cell.b_ih = rng.standard_normal(h)
    cell.b_hh = rng.standard_normal(h)
    x = rng.standard_normal((N, d))
    h0 = rng.standard_normal((N, h))
    out = cell.forward(x, h0)

    tc = torch.nn.RNNCell(d, h, nonlinearity="tanh")
    tc.weight_ih = torch.nn.Parameter(torch.tensor(cell.W_ih))
    tc.weight_hh = torch.nn.Parameter(torch.tensor(cell.W_hh))
    tc.bias_ih = torch.nn.Parameter(torch.tensor(cell.b_ih))
    tc.bias_hh = torch.nn.Parameter(torch.tensor(cell.b_hh))
    outt = tc(torch.tensor(x), torch.tensor(h0))
    assert np.allclose(out, outt.detach().numpy(), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------- GRU cell fwd
def test_grucell_forward_vs_torch():
    d, h = 5, 6
    cell = GRUCell(d, h)
    x = rng.standard_normal(d)
    h0 = rng.standard_normal(h)
    out = cell.forward(x, h0)

    tc = torch.nn.GRUCell(d, h)
    # torch stacks gates as [r, z, n]; our layout is r, z, n
    W_ih = np.vstack([cell.Wrx, cell.Wzx, cell.Wnx])
    W_hh = np.vstack([cell.Wrh, cell.Wzh, cell.Wnh])
    b_ih = np.concatenate([cell.brx, cell.bzx, cell.bnx])
    b_hh = np.concatenate([cell.brh, cell.bzh, cell.bnh])
    tc.weight_ih = torch.nn.Parameter(torch.tensor(W_ih))
    tc.weight_hh = torch.nn.Parameter(torch.tensor(W_hh))
    tc.bias_ih = torch.nn.Parameter(torch.tensor(b_ih))
    tc.bias_hh = torch.nn.Parameter(torch.tensor(b_hh))
    outt = tc(torch.tensor(x).unsqueeze(0), torch.tensor(h0).unsqueeze(0))
    assert np.allclose(out, outt.detach().numpy().reshape(-1), rtol=RTOL, atol=ATOL)


def test_grucell_backward_gradcheck():
    """Numerical gradient check of the GRU backward w.r.t. input x."""
    d, h = 4, 5
    cell = GRUCell(d, h)
    x = rng.standard_normal(d)
    h0 = rng.standard_normal(h)
    delta = rng.standard_normal(h)

    out = cell.forward(x, h0)
    dx, dh = cell.backward(delta)
    dx = dx.reshape(-1)

    eps = 1e-6
    num_dx = np.zeros(d)
    for i in range(d):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        fp = np.sum(cell.forward(xp, h0) * delta)
        fm = np.sum(cell.forward(xm, h0) * delta)
        num_dx[i] = (fp - fm) / (2 * eps)
    assert np.allclose(dx, num_dx, rtol=1e-3, atol=1e-4)


# ------------------------------------------------------------------ CTC loss
def test_ctc_loss_vs_torch():
    T, B, C = 12, 1, 5   # includes blank at 0
    logits = rng.standard_normal((T, B, C))
    # softmax over classes
    probs = np.exp(logits - logits.max(2, keepdims=True))
    probs /= probs.sum(2, keepdims=True)

    target = np.array([[1, 3, 2, 4]])  # length-4 label seq (non-blank)
    input_lengths = np.array([T])
    target_lengths = np.array([4])

    myctc = CTCLoss(BLANK=0)
    my_loss = myctc.forward(probs, target, input_lengths, target_lengths)

    # Our loss is the true -log P(target) averaged over the batch. PyTorch's
    # reduction="mean" additionally divides each example by its target length,
    # so compare against reduction="sum" / B (which is exactly -sum log P / B).
    log_probs = torch.log(torch.tensor(probs))       # (T, B, C)
    tloss = torch.nn.functional.ctc_loss(
        log_probs, torch.tensor(target), torch.tensor(input_lengths),
        torch.tensor(target_lengths), blank=0, reduction="sum")
    assert np.allclose(my_loss, tloss.item() / B, rtol=1e-4, atol=1e-5)


def test_ctc_backward_gradcheck():
    """Numerical gradient check for CTC backward w.r.t. softmax probs."""
    T, B, C = 8, 1, 4
    logits = rng.standard_normal((T, B, C))
    probs = np.exp(logits - logits.max(2, keepdims=True))
    probs /= probs.sum(2, keepdims=True)
    target = np.array([[1, 2, 3]])
    il = np.array([T]); tl = np.array([3])

    ctc = CTCLoss(BLANK=0)
    ctc.forward(probs, target, il, tl)
    grad = ctc.backward()

    eps = 1e-6
    # check a handful of random entries
    for _ in range(8):
        t = rng.integers(0, T); c = rng.integers(0, C)
        pp = probs.copy(); pp[t, 0, c] += eps
        pm = probs.copy(); pm[t, 0, c] -= eps
        fp = CTCLoss(0).forward(pp, target, il, tl)
        fm = CTCLoss(0).forward(pm, target, il, tl)
        num = (fp - fm) / (2 * eps)
        assert abs(grad[t, 0, c] - num) < 1e-3


# ---------------------------------------------------------------- decoders
def test_greedy_decode():
    symbols = ["a", "b", "c"]
    T = 6
    # build a probability sequence whose greedy path is a,a,blank,b,b,c
    idx_path = [1, 1, 0, 2, 2, 3]     # 0 = blank; 1->a,2->b,3->c
    y = np.full((4, T, 1), 0.01)
    for t, k in enumerate(idx_path):
        y[k, t, 0] = 0.97
    dec = GreedySearchDecoder(symbols)
    s, p = dec.decode(y)
    assert s == "abc"           # aa -> a, blank dropped, bb -> b, c
    assert p > 0


def test_beam_decode_matches_greedy_on_peaky():
    symbols = ["a", "b", "c"]
    T = 5
    idx_path = [1, 0, 2, 0, 3]
    y = np.full((4, T, 1), 0.001)
    for t, k in enumerate(idx_path):
        y[k, t, 0] = 0.997
    greedy, _ = GreedySearchDecoder(symbols).decode(y)
    best, scores = BeamSearchDecoder(symbols, beam_width=3).decode(y)
    assert greedy == "abc"
    assert best == "abc"
