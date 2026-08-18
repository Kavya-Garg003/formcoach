"""
1D-CNN rep-correctness classifier (PRD 6.1 & 7 -- the "optional but
recommended" trained-model deliverable, layered on top of the rule-based
threshold baseline in frontend/js/deviationRules.js).

Input: a sliding window of joint-angle time series for one rep,
  shape (batch, num_joint_angle_channels, window_len)
Output: per-error-class logits (multi-label, since a rep can have more
  than one simultaneous error, e.g. both "knees inward" and "not low enough").

This is intentionally small -- EC3D has a few hundred sequences total
(132 squat / 127 lunge / 103 plank per PRD 6.1), so a large model will
overfit fast. Start here before reaching for anything bigger.
"""
import torch
import torch.nn as nn


class RepClassifier1DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, window_len: int = 60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, window_len)
        feats = self.net(x).squeeze(-1)  # (batch, 64)
        return self.head(feats)  # (batch, num_classes) -- raw logits, use BCEWithLogitsLoss for multi-label


class RepClassifierLSTM(nn.Module):
    """Alternative to the CNN above -- PRD 6.1 lists either as acceptable.
    Try both, keep whichever generalizes better on the EC3D val split."""

    def __init__(self, in_channels: int, num_classes: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, window_len) -> LSTM wants (batch, window_len, in_channels)
        x = x.transpose(1, 2)
        out, (h_n, _) = self.lstm(x)
        # concat final forward/backward hidden states
        final = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(final)
