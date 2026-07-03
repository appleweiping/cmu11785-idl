"""HW3P2 — sequence transcription with a BiLSTM + CTC (utterance -> symbols).

The official HW3P2 task: transcribe a variable-length acoustic utterance into a
phoneme sequence, trained with CTC and decoded greedily / with beam search.

The competition WSJ data is Kaggle-gated, so we build an acoustically-real
sequence task from **UCI ISOLET**: each "utterance" is the concatenation of
several real spoken-letter feature frames (each letter contributes a short run
of frames), and the target is the letter sequence.  A BiLSTM encoder + linear
projection is trained with ``torch.nn.CTCLoss``; decoding is done with **our
own MyTorch** greedy / beam CTC decoders to validate them on real model output.
Part 2 may use PyTorch for training (only mytorch Part 1 is NumPy-only).

Run:
    python scripts/hw3p2_ctc_asr.py
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(3)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mytorch.nn import GreedySearchDecoder, BeamSearchDecoder   # noqa: E402

BLANK = 0
ALPHABET = [chr(ord("a") + i) for i in range(26)]   # symbol_set (no blank)


def build_utterances(n_utts, min_len=3, max_len=6, frames_per_letter=4, seed=0):
    """Concatenate real ISOLET letter-frames into labelled utterances."""
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("isolet", version=1, return_X_y=True,
                        as_frame=False, data_home="data/openml")
    y = y.astype(int) - 1                       # 0..25
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X).astype(np.float32)
    by = {c: X[y == c] for c in range(26)}

    rng = np.random.default_rng(seed)
    feats, targets = [], []
    for _ in range(n_utts):
        L = rng.integers(min_len, max_len + 1)
        letters = rng.integers(0, 26, size=L)
        frames = []
        for c in letters:
            pool = by[int(c)]
            # a few real frames of this letter, with small jitter
            picks = pool[rng.integers(0, len(pool), size=frames_per_letter)]
            frames.append(picks + rng.standard_normal(picks.shape).astype(np.float32) * 0.05)
        feats.append(np.concatenate(frames, axis=0))     # (L*fpl, 617)
        targets.append(letters + 1)                      # 1..26 (0 = blank)
    return feats, targets


class BiLSTM_CTC(nn.Module):
    def __init__(self, in_dim=617, hidden=256, n_classes=27):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out)                              # (B, T, C)


def collate(feats, targets):
    lengths = [f.shape[0] for f in feats]
    T = max(lengths)
    B = len(feats)
    X = np.zeros((B, T, feats[0].shape[1]), dtype=np.float32)
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
    return (torch.tensor(X), torch.tensor(lengths),
            [torch.tensor(t) for t in targets])


def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (ca != cb))
    return dp[-1]


def main():
    feats, targets = build_utterances(1200, seed=0)
    ntr = 1000
    tr_f, tr_t = feats[:ntr], targets[:ntr]
    te_f, te_t = feats[ntr:], targets[ntr:]
    print(f"utterances: train {len(tr_f)} test {len(te_f)} | feat dim 617, 27 classes")

    model = BiLSTM_CTC()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    Xtr, Ltr, Ytr = collate(tr_f, tr_t)
    epochs, batch = 15, 32
    N = len(tr_f)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch].tolist()
            xb = Xtr[idx]; lb = Ltr[idx]
            yb = [Ytr[j] for j in idx]
            y_cat = torch.cat(yb)
            y_len = torch.tensor([len(t) for t in yb])
            opt.zero_grad()
            logits = model(xb, lb)                        # (B, T, C)
            logp = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            loss = ctc(logp, y_cat, lb, y_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        history.append({"epoch": ep + 1, "loss": tot / (N // batch)})
        print(f"epoch {ep+1:2d}/{epochs}  ctc_loss {history[-1]['loss']:.4f}")
    elapsed = time.time() - t0

    # ---- evaluate with OUR mytorch decoders on real model output
    model.eval()
    Xte, Lte, Yte = collate(te_f, te_t)
    greedy = GreedySearchDecoder(ALPHABET)
    beam = BeamSearchDecoder(ALPHABET, beam_width=5)
    tot_g = tot_b = tot_chars = 0
    n_eval = min(150, len(te_f))
    with torch.no_grad():
        for i in range(n_eval):
            x = Xte[i:i + 1, :Lte[i]]
            logits = model(x, Lte[i:i + 1])
            probs = logits.softmax(2)[0].numpy()             # (T, C)
            y_probs = probs.T[:, :, None]                    # (C, T, 1)
            ref = "".join(ALPHABET[c - 1] for c in te_t[i])
            gp, _ = greedy.decode(y_probs)
            bp, _ = beam.decode(y_probs)
            tot_g += levenshtein(gp, ref)
            tot_b += levenshtein(bp, ref)
            tot_chars += len(ref)
    ger = tot_g / tot_chars
    ber = tot_b / tot_chars
    print(f"\nFINAL char error rate  greedy {ger:.4f}  beam(5) {ber:.4f} "
          f"(mytorch decoders, {n_eval} test utts, {elapsed:.1f}s CPU)")

    os.makedirs("results", exist_ok=True)
    with open("results/hw3p2_ctc_asr.json", "w") as f:
        json.dump({"dataset": "ISOLET-derived utterance sequences",
                   "model": "BiLSTM(2x256) + Linear + CTC",
                   "decoder": "MyTorch greedy & beam(5)",
                   "greedy_cer": ger, "beam_cer": ber,
                   "epochs": epochs, "elapsed_sec": elapsed,
                   "history": history}, f, indent=2)
    print("saved results/hw3p2_ctc_asr.json")


if __name__ == "__main__":
    main()
