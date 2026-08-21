"""
Serves the classifier trained by classifier/train.py, if you've trained
one and pointed this at it. This is what was previously missing entirely --
train.py could produce rep_classifier.pt, but nothing ever loaded it back
up or used it. This module is that missing piece.

Torch is a large, optional dependency (kept out of backend/requirements.txt
on purpose -- see README). This module must NOT crash backend startup if
torch or the checkpoint aren't available; every function below degrades to
"classifier unavailable" instead of raising, so the rest of the app (rule-
based deviation detection, chat, everything else) keeps working regardless.

Read backend/app/ntu_mapping.py's module docstring before trusting this
model's predictions on live data -- there's a real, documented coordinate-
space caveat between what the model was trained on and what live MediaPipe
data looks like.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from functools import lru_cache

from .ntu_mapping import blazepose_sequence_to_ntu25

DEFAULT_CHECKPOINT_PATH = os.environ.get(
    "FORMCOACH_CLASSIFIER_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "rep_classifier.pt"),
)


def _meta_path_for(checkpoint_path: str) -> str:
    return str(Path(checkpoint_path).with_suffix("")) + ".meta.json"


def _resample_to_length(x, target_len: int):
    """Same linear-resample approach as classifier/train.py's
    _resample_to_length, duplicated here so this module doesn't need
    classifier/ on its import path. x: (channels, n_frames) -> (channels, target_len)."""
    import numpy as np

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


class ClassifierService:
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.meta_path = _meta_path_for(checkpoint_path)
        self._model = None
        self._meta = None
        self._load_error = None
        self._try_load()

    def _try_load(self):
        if not Path(self.checkpoint_path).exists():
            self._load_error = (
                f"No checkpoint at {self.checkpoint_path}. Train one with "
                f"classifier/train.py (see its --out flag) and either point "
                f"FORMCOACH_CLASSIFIER_PATH at it, or copy it to that default path."
            )
            return
        if not Path(self.meta_path).exists():
            self._load_error = (
                f"Found {self.checkpoint_path} but not its metadata sidecar "
                f"{self.meta_path}. Re-run classifier/train.py (the current version "
                f"saves this automatically) -- checkpoints saved before that fix "
                f"can't be loaded safely since we don't know their class count."
            )
            return

        try:
            import torch
        except ImportError:
            self._load_error = (
                "torch isn't installed in the backend's environment. Run "
                "`pip install -r classifier/requirements.txt` in the same "
                "venv as the backend, or set up a separate process for serving."
            )
            return

        try:
            with open(self.meta_path) as f:
                self._meta = json.load(f)

            import sys

            classifier_dir = str(Path(__file__).resolve().parent.parent.parent / "classifier")
            if classifier_dir not in sys.path:
                sys.path.insert(0, classifier_dir)
            from model import RepClassifier1DCNN

            model = RepClassifier1DCNN(
                in_channels=self._meta["in_channels"],
                num_classes=self._meta["num_classes"],
                window_len=self._meta["window_len"],
            )
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        except Exception as e:  # noqa: BLE001 -- deliberately broad, this must never crash startup
            self._load_error = f"Failed to load classifier: {type(e).__name__}: {e}"
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "error": self._load_error,
            "meta": self._meta,
        }

    def predict(self, frames: list[list[dict]]) -> dict:
        """
        @param frames: one rep's worth of frames, each a list of 33
            BlazePose landmarks ({"x","y","z"}).
        @returns: {available: False, reason: ...} if the classifier isn't
            loaded, or {available: True, predicted_label, confidence,
            class_probabilities, caveat} otherwise.
        """
        if not self.is_available():
            return {"available": False, "reason": self._load_error or "classifier not loaded"}

        if self._meta.get("feature_space") != "ec3d_xyz_25joint":
            return {
                "available": False,
                "reason": f"This checkpoint was trained on feature_space="
                f"{self._meta.get('feature_space')!r}, which this endpoint doesn't "
                f"know how to build from live landmarks yet.",
            }

        import numpy as np
        import torch

        ntu_seq = blazepose_sequence_to_ntu25(frames)  # (n_frames, 25, 3)
        arr = np.array(ntu_seq, dtype="float32")  # (n_frames, 25, 3)
        arr = arr.transpose(2, 1, 0).reshape(75, -1)  # (75, n_frames) -- matches training's (3*25, T) flatten order
        arr = _resample_to_length(arr, self._meta["window_len"])

        x = torch.from_numpy(arr).unsqueeze(0)  # (1, 75, window_len)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.sigmoid(logits)[0].tolist()

        idx_to_label = {v: k for k, v in self._meta["label_to_idx"].items()}
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_idx = ranked[0]

        return {
            "available": True,
            "predicted_label": idx_to_label.get(top_idx, f"class_{top_idx}"),
            "confidence": probs[top_idx],
            "class_probabilities": {idx_to_label.get(i, f"class_{i}"): p for i, p in enumerate(probs)},
            "caveat": (
                "Experimental: this model was trained on EC3D's camera-calibrated "
                "coordinates, not live MediaPipe data. Treat this as a secondary "
                "signal alongside the rule-based deviation check, not ground truth. "
                "See backend/app/ntu_mapping.py for details."
            ),
        }


@lru_cache(maxsize=1)
def get_classifier_service() -> ClassifierService:
    return ClassifierService()
