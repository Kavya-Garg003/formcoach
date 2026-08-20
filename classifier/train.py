"""
Training script for the EC3D rep-correctness classifier.

  python train.py --smoke-test          # verifies the training loop runs, using random data
  python train.py --data-dir ./ec3d     # trains on real EC3D data once you've downloaded it

--- Getting the real data (do this yourself; not fetched automatically) ---
The dataset is hosted on Google Drive, not GitHub, so it can't be fetched
by a script automatically:
1. Paper: arXiv:2208.03257, "3D Pose Based Feedback for Physical Exercises"
   (Zhao et al., ACCV 2022). Reference implementation + exact download link:
   https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises
   -> Google Drive folder linked in that repo's README.
2. Download `data_3D.pickle` (the file the dataset's own README recommends
   for reproducing results) into `--data-dir`, e.g. `classifier/ec3d/data_3D.pickle`.
3. PRD's recommended split for comparability with the paper: subjects 1-3
   for train, subject 4 for test (already wired into EC3DDataset below).

--- Confirmed pickle structure (from the dataset authors' own dataset.py) ---
  data["poses"]  -> ndarray, shape (N_frames_total, 3, 25): xyz coords for
                     25 body joints (NTU RGB+D-style layout), concatenated
                     across every frame of every recorded rep.
  data["labels"] -> array-like with columns ["act", "sub", "lab", "rep", "frame"]:
                     act=exercise, sub=subject id (1-4), lab=instruction
                     label id (0="Correct", 1+=a specific error, meaning
                     differs per exercise per PRD 6.1's Table 1), rep=rep
                     index, frame=frame index within that rep.

IMPORTANT: EC3D's labels are single-label per rep (each rep was recorded
under ONE instruction), not true multi-label. EC3DDataset one-hot-encodes
them so they still work with this file's multi-label BCE training loop
without changes -- switch to plain integer labels + CrossEntropyLoss if
you'd rather train strictly single-label (more faithful to how the data
was actually collected).

Also note: EC3D ships raw xyz joint coordinates, not the 4 joint angles
this project's live pipeline computes (frontend/js/jointAngles.js). So
EC3DDataset uses 75 input channels (3 xyz * 25 joints, flattened) instead
of JOINT_ANGLE_CHANNELS' 4 -- `main()` below sizes the model to whichever
dataset you actually pass it, so this isn't a manual step.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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


def _resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (channels, n_frames) array to (channels, target_len),
    so reps of different durations all become fixed-size model inputs."""
    n_frames = x.shape[1]
    if n_frames == target_len:
        return x
    if n_frames == 1:
        return np.repeat(x, target_len, axis=1)
    src_idx = np.linspace(0, n_frames - 1, num=target_len)
    out = np.zeros((x.shape[0], target_len), dtype=x.dtype)
    for c in range(x.shape[0]):
        out[c] = np.interp(src_idx, np.arange(n_frames), x[c])
    return out


class EC3DDataset(Dataset):
    """Loads EC3D's `data_3D.pickle` -- see the module docstring above for
    the confirmed pickle structure and the download link (Google Drive,
    linked from the paper's reference GitHub repo).

    Expects `data_3D.pickle` directly inside `data_dir`.
    """

    TRAIN_SUBJECTS = ["Hugues", "Sena", "Vidit", 1, 2, 3]
    TEST_SUBJECTS = ["Isinsu", 4]

    def __init__(self, data_dir: str, split: str):
        pickle_path = Path(data_dir) / "data_3D.pickle"
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"{pickle_path} not found. Download data_3D.pickle from the EC3D dataset's "
                "Google Drive link (see this file's module docstring) and place it there."
            )
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        poses = np.asarray(data["poses"])  # (N_frames_total, 3, 25)
        labels_df = pd.DataFrame(data["labels"], columns=["act", "sub", "lab", "rep", "frame"])
        labels_df["lab"] = labels_df["lab"].astype(int)
        try:
            labels_df["sub"] = labels_df["sub"].astype(int)
        except (ValueError, TypeError):
            pass

        subs = self.TRAIN_SUBJECTS if split == "train" else self.TEST_SUBJECTS
        keep_mask = labels_df["sub"].isin(subs).to_numpy()
        labels_df = labels_df[keep_mask].reset_index(drop=True)
        poses = poses[keep_mask]

        n_classes = int(labels_df["lab"].max()) + 1
        X, y = [], []
        for (_act, _sub, lab, _rep), group in labels_df.groupby(["act", "sub", "lab", "rep"]):
            idx = group.index.to_numpy()
            seq = poses[idx]  # (n_frames_in_rep, 3, 25)
            seq_flat = seq.reshape(seq.shape[0], -1).T  # (75, n_frames_in_rep)
            X.append(_resample_to_length(seq_flat, WINDOW_LEN).astype("float32"))

            onehot = np.zeros(n_classes, dtype="float32")
            onehot[lab] = 1.0
            y.append(onehot)

        if not X:
            raise ValueError(f"No sequences found for split={split!r} (subjects={subs}) -- check the pickle contents.")

        self.X = np.stack(X)
        self.y = np.stack(y)
        self.num_classes = n_classes
        self.in_channels = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        in_channels = len(JOINT_ANGLE_CHANNELS)
        n_classes = args.n_classes
    else:
        dataset = EC3DDataset(args.data_dir, split="train")
        in_channels = dataset.in_channels  # 75 (3 xyz * 25 joints) for real EC3D data
        n_classes = dataset.num_classes
        print(f"Loaded {len(dataset)} EC3D training reps, {in_channels} input channels, {n_classes} classes.")

    n_val = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = RepClassifier1DCNN(
        in_channels=in_channels, num_classes=n_classes, window_len=WINDOW_LEN
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
