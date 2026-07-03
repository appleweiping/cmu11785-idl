"""HW2P2 — face classification + verification with a CNN.

The official HW2P2 task: (1) classify a face into one of N identities, and
(2) *verify* whether two faces are the same identity, scored by AUC on the
similarity of learned embeddings.

The competition dataset is Kaggle-gated, so we train the same two-headed
pipeline on **Labeled Faces in the Wild (LFW)** — real face photographs from
sklearn's ``fetch_lfw_people`` — restricted to identities with enough images.
Part 2 is allowed to use PyTorch (only Part 1 / mytorch must be NumPy-only).

Run:
    python scripts/hw2p2_face_cnn.py
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(3)


def load_lfw():
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    people = fetch_lfw_people(min_faces_per_person=40, resize=0.5,
                              color=False, data_home="data/lfw")
    X = people.images.astype(np.float32) / 255.0      # (N, H, W)
    y = people.target.astype(np.int64)
    n_classes = len(people.target_names)
    # per-image standardisation
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    return (Xtr[:, None], ytr, Xte[:, None], yte, n_classes,
            people.images.shape[1:])


class FaceCNN(nn.Module):
    """Small ResNet-ish CNN producing an embedding + classification logits."""

    def __init__(self, n_classes, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        emb = self.embedding(f)
        logits = self.classifier(F.relu(emb))
        return logits, emb


def make_verification_pairs(emb, labels, n_pairs=3000, seed=0):
    """Return (cosine_similarities, is_same) for random same/diff pairs."""
    rng = np.random.default_rng(seed)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims, same = [], []
    by_class = {}
    for i, l in enumerate(labels):
        by_class.setdefault(int(l), []).append(i)
    classes = [c for c, v in by_class.items() if len(v) >= 2]
    for _ in range(n_pairs // 2):
        c = rng.choice(classes)
        i, j = rng.choice(by_class[c], 2, replace=False)
        sims.append(float(emb[i] @ emb[j])); same.append(1)
        c1, c2 = rng.choice(classes, 2, replace=False)
        i = rng.choice(by_class[c1]); j = rng.choice(by_class[c2])
        sims.append(float(emb[i] @ emb[j])); same.append(0)
    return np.array(sims), np.array(same)


def main():
    Xtr, ytr, Xte, yte, n_classes, img_shape = load_lfw()
    print(f"LFW: train {Xtr.shape} test {Xte.shape} classes {n_classes} img {img_shape}")

    dev = torch.device("cpu")
    model = FaceCNN(n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte)

    epochs, batch = 20, 64
    N = Xtr.shape[0]
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits, _ = model(Xtr_t[idx])
            loss = crit(logits, ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits, _ = model(Xte_t)
            te_acc = (logits.argmax(1) == yte_t).float().mean().item()
        history.append({"epoch": ep + 1, "loss": tot / N, "test_acc": te_acc})
        print(f"epoch {ep+1:2d}/{epochs}  loss {tot/N:.4f}  test_acc {te_acc:.4f}")
    elapsed = time.time() - t0

    # ---- verification via embeddings + AUC
    model.eval()
    with torch.no_grad():
        _, emb = model(Xte_t)
    emb = emb.numpy()
    sims, same = make_verification_pairs(emb, yte)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(same, sims)
    cls_acc = history[-1]["test_acc"]
    print(f"\nFINAL classification acc {cls_acc:.4f}  verification AUC {auc:.4f} "
          f"({elapsed:.1f}s CPU)")

    os.makedirs("results", exist_ok=True)
    with open("results/hw2p2_face_cnn.json", "w") as f:
        json.dump({"dataset": "LFW (min_faces_per_person=40)",
                   "n_classes": int(n_classes),
                   "classification_test_acc": cls_acc,
                   "verification_auc": float(auc),
                   "epochs": epochs, "elapsed_sec": elapsed,
                   "history": history}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ep = [h["epoch"] for h in history]
        ax[0].plot(ep, [h["test_acc"] for h in history], marker="o")
        ax[0].set_title(f"Classification (final {cls_acc:.3f})")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test acc"); ax[0].grid(alpha=.3)
        ax[1].hist(sims[same == 1], bins=30, alpha=.6, label="same")
        ax[1].hist(sims[same == 0], bins=30, alpha=.6, label="different")
        ax[1].set_title(f"Verification cosine sim (AUC {auc:.3f})")
        ax[1].set_xlabel("cosine similarity"); ax[1].legend()
        plt.tight_layout(); plt.savefig("results/hw2p2_face_results.png", dpi=110)
        print("saved results/hw2p2_face_results.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
