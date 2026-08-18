"""
Training script for the EC3D rep-correctness classifier.

  python train.py --smoke-test          # verifies the training loop runs, using random data
  python train.py --data-dir ./ec3d     # trains on real EC3D data once you've downloaded + prepped it

--- Getting the real data (do this yourself; not fetched automatically) ---
1. Download EC3D from the paper's repo: arXiv:2208.03257,
   "3D Pose Based Feedback for Physical Exercises" -- check the paper for
   the current dataset link (GitHub, at last check).
2. Follow the PRD's recommended split for comparability with the paper:
   subjects 1-3 for train, subject 4 for test. Hold out a small slice of
   the training subjects for validation.
3. Fill in `EC3DDataset` below to load their per-rep joint-angle sequences
   and label vectors instead of the synthetic generator. The exact label
   set per exercise is in PRD 6.1 (e.g. squats: Correct, Feet too wide,
   Knees inward, Not low enough, Front bent).

Everything else (model, training loop, metrics) is dataset-agnostic and
should not need to change.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score

from model import RepClassifier1DCNN

JOINT_ANGLE_CHANNELS = ["knee_flexion", "hip_flexion", "spine_angle", "ankle_dorsiflexion"]
WINDOW_LEN = 60  # frames per rep window; resample each rep to this length


class SyntheticRepDataset(Dataset):
    """Stand-in for EC3D while you're getting the real data prepped --
    lets you verify the whole train/eval loop actually runs end-to-end."""

    def __init__(self, n_samples=300, n_classes=5, seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n_samples, len(JOINT_ANGLE_CHANNELS), WINDOW_LEN)).astype("float32")
        self.y = (rng.random((n_samples, n_classes)) > 0.7).astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EC3DDataset(Dataset):
    """TODO: implement once you have EC3D downloaded and parsed.
    Expected: self.X shape (N, len(JOINT_ANGLE_CHANNELS), WINDOW_LEN),
              self.y shape (N, num_error_classes), multi-hot."""

    def __init__(self, data_dir: str, split: str):
        raise NotImplementedError(
            "Fill this in once EC3D is downloaded -- see the module docstring for the subject split."
        )


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        all_preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "exact_match_acc": accuracy_score(targets, preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-classes", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke_test or not args.data_dir:
        print("Running with synthetic data (pass --data-dir once EC3D is ready).")
        dataset = SyntheticRepDataset(n_classes=args.n_classes)
    else:
        dataset = EC3DDataset(args.data_dir, split="train")

    n_val = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = RepClassifier1DCNN(
        in_channels=len(JOINT_ANGLE_CHANNELS), num_classes=args.n_classes, window_len=WINDOW_LEN
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        print(
            f"epoch {epoch:02d}  train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_f1_macro={val_metrics['f1_macro']:.3f}  "
            f"val_exact_acc={val_metrics['exact_match_acc']:.3f}"
        )

    torch.save(model.state_dict(), "rep_classifier.pt")
    print("Saved rep_classifier.pt")


if __name__ == "__main__":
    main()
