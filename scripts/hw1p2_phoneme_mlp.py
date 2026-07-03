"""HW1P2 — frame-level acoustic classification with a MyTorch MLP.

The official HW1P2 task is frame-level phoneme classification: given a context
window of MFCC frames, predict the centre frame's phoneme label with an MLP.

The official competition dataset (WSJ-derived MFCCs) is Kaggle-gated and cannot
be redistributed, so we train the *same model* (a MyTorch-only MLP, no torch
autograd) on a real, public, acoustically-equivalent task: **UCI ISOLET**
(7797 spoken-letter utterances, 617 real acoustic features, 26 classes),
fetched from OpenML.  The pipeline (standardise -> MLP -> softmax CE -> SGD with
momentum -> measured test accuracy) is identical in structure to HW1P2.

Run:
    python scripts/hw1p2_phoneme_mlp.py
"""

import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mytorch.models import MLP            # noqa: E402
from mytorch.nn import CrossEntropyLoss, ReLU   # noqa: E402
from mytorch.optim import SGD             # noqa: E402


def load_isolet():
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = fetch_openml("isolet", version=1, return_X_y=True,
                        as_frame=False, data_home="data/openml")
    y = (y.astype(int) - 1)                       # labels 1..26 -> 0..25
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), ytr, sc.transform(Xte), yte


def one_hot(y, C):
    return np.eye(C)[y]


def accuracy(model, X, y):
    logits = model.forward(X)
    return float(np.mean(np.argmax(logits, axis=1) == y))


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    Xtr, ytr, Xte, yte = load_isolet()
    N, D = Xtr.shape
    C = 26
    Ytr = one_hot(ytr, C)
    print(f"train {Xtr.shape}  test {Xte.shape}  classes {C}")

    # 2-hidden-layer MLP built entirely from mytorch
    model = MLP([D, 512, 256, C], activation=ReLU)
    # He-style init for the ReLU MLP
    for lin in model.layers:
        fan_in = lin.W.shape[1]
        lin.W = rng.standard_normal(lin.W.shape) * np.sqrt(2.0 / fan_in)
        lin.b = np.zeros(lin.b.shape)

    crit = CrossEntropyLoss()
    opt = SGD(model, lr=0.05, momentum=0.9)

    epochs, batch = 25, 128
    history = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(N)
        ep_loss = 0.0
        nb = 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            out = model.forward(xb)
            loss = crit.forward(out, yb)
            model.backward(crit.backward())
            opt.step()
            ep_loss += loss
            nb += 1
        tr_acc = accuracy(model, Xtr, ytr)
        te_acc = accuracy(model, Xte, yte)
        history.append({"epoch": ep + 1, "loss": ep_loss / nb,
                        "train_acc": tr_acc, "test_acc": te_acc})
        print(f"epoch {ep+1:2d}/{epochs}  loss {ep_loss/nb:.4f} "
              f"train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")

    elapsed = time.time() - t0
    final = history[-1]
    print(f"\nFINAL test accuracy {final['test_acc']:.4f} "
          f"({elapsed:.1f}s, {epochs} epochs, CPU)")

    os.makedirs("results", exist_ok=True)
    with open("results/hw1p2_phoneme_mlp.json", "w") as f:
        json.dump({"dataset": "UCI ISOLET (OpenML)",
                   "model": "MyTorch MLP [617,512,256,26]",
                   "epochs": epochs, "batch": batch,
                   "elapsed_sec": elapsed,
                   "final_test_acc": final["test_acc"],
                   "history": history}, f, indent=2)

    # learning-curve figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, [h["train_acc"] for h in history], label="train acc")
        plt.plot(ep, [h["test_acc"] for h in history], label="test acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy")
        plt.title("HW1P2 MyTorch MLP on ISOLET (frame classification)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("results/hw1p2_learning_curve.png", dpi=110)
        print("saved results/hw1p2_learning_curve.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
