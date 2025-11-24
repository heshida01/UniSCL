# -*- coding: utf-8 -*-
"""
Multi-task RNA subcellular localization - Inference Script

Usage:
    python test.py --input_fasta test.fasta --input_fea test_top120.csv --output preds.csv \
                   [--model_path best_model.pt] [--scaler_mean scaler_mean.npy] \
                   [--scaler_scale scaler_scale.npy] [--task lncrna]

Notes:
- input_fasta:
    * Can be FASTA format (">id" followed by sequence lines)
    * Or CSV format (must contain a 'sequence' column, and preferably 'task' / 'label')
- input_fea:
    * CSV file with 120-dimensional hand-crafted features per sample
    * If it contains 'task' or 'label' columns they will be used automatically
- output:
    * Output CSV file with predictions
- model_path:
    * Trained model file (state_dict), e.g. best_model.pt
- scaler_mean / scaler_scale:
    * StandardScaler mean and scale used in training; if missing, no feature normalization is applied
- task:
    * If no 'task' column is found in the input, this argument assigns the task for all samples
      ('lncrna' or 'mrna')
"""

import os
import argparse
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score,
)

# ============================================================
# 1. Utility
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_task(x):
    """Normalize task label to 'lncrna' or 'mrna'."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return "lncrna" if int(x) == 0 else "mrna"
    s = str(x).strip().lower()
    if s in ["lncrna", "lnc_rna", "lnc"]:
        return "lncrna"
    if s in ["mrna", "m_rna", "rna"]:
        return "mrna"
    return s


def parse_fasta(fasta_path: str) -> pd.DataFrame:
    """
    Parse a FASTA file and return a DataFrame with columns: ['id', 'sequence'].
    """
    ids: List[str] = []
    seqs: List[str] = []

    with open(fasta_path, "r") as f:
        current_id = None
        current_seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous record
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(current_seq))
                # New record
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Save last record
        if current_id is not None:
            ids.append(current_id)
            seqs.append("".join(current_seq))

    df = pd.DataFrame({"id": ids, "sequence": seqs})
    return df


# ============================================================
# 2. Dataset for Inference
# ============================================================

class RNAPredictDataset(Dataset):
    """
    Dataset used for inference / testing.

    df must contain:
        - 'sequence'
        - 'task' (lncrna / mrna)
        - optional 'label' for evaluation
    X_feat: np.ndarray, shape (N, feat_dim)
    """
    def __init__(self, df: pd.DataFrame, X_feat: np.ndarray, total_len: int = 450):
        assert len(df) == len(X_feat)
        self.df = df.reset_index(drop=True)
        self.Xf = X_feat
        self.total_len = total_len
        # A/C/G/T/U/N -> 0..4
        self.tk = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}

    def _head_tail(self, ids: List[int], L: int) -> List[int]:
        """
        Extract fixed-length sequence by taking head and tail segments.
        """
        half = L // 2
        head = ids[:half]
        remain = L - len(head)
        tail = ids[-remain:] if len(ids) > len(head) else []
        merged = head + tail
        if len(merged) < L:
            merged += [0] * (L - len(merged))
        return merged[:L]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row['sequence'])
        ids = [self.tk.get(ch.upper(), 4) for ch in seq]
        x_seq = self._head_tail(ids, self.total_len)

        task_name = norm_task(row['task'])
        task_idx = 0 if task_name == 'lncrna' else 1

        # Use true label if available; otherwise -1
        label = row['label'] if 'label' in self.df.columns else -1
        label = int(label) if label is not None and label != "" else -1

        feat = self.Xf[idx]

        return (
            torch.tensor(x_seq, dtype=torch.long),
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(task_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    xs, xf, t, y = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(xf),
        torch.stack(t),
        torch.stack(y)
    )


# ============================================================
# 3. Model (same as training)
# ============================================================

class DilatedCNN(nn.Module):
    def __init__(self, vocab=5, emb_dim=64, channels=128,
                 layers=4, kernel=7, dilations=(1, 2, 4, 8)):
        super().__init__()
        # padding_idx=4 -> N
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=4)
        convs = []
        in_c = emb_dim
        for i in range(layers):
            d = dilations[i % len(dilations)]
            convs += [
                nn.Conv1d(in_c, channels, kernel,
                          padding=d * (kernel // 2), dilation=d),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(0.2)
            ]
            in_c = channels
        self.net = nn.Sequential(*convs)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: B x L
        h = self.emb(x).transpose(1, 2)  # B x E x L
        h = self.net(h)
        g = self.pool(h).squeeze(-1)     # B x C
        return g


class MultiTaskCNNFusion(nn.Module):
    """Shared CNN + feature fusion + task-specific heads."""
    def __init__(self, feat_dim=120, cnn_dim=128,
                 num_classes=3, dropout=0.3, emb_dim=64):
        super().__init__()
        self.cnn = DilatedCNN(emb_dim=emb_dim, channels=cnn_dim)
        self.fuse_dim = cnn_dim + feat_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fuse_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
            for _ in range(2)  # two tasks: lncrna, mrna
        ])

    def forward(self, seq, feat):
        z = self.cnn(seq)  # B x cnn_dim
        fused = torch.cat([z, feat], dim=1)  # B x (cnn_dim + feat_dim)
        out = [head(fused) for head in self.heads]  # list of [B x C]
        return out


# ============================================================
# 4. Metrics & Prediction
# ============================================================

def compute_metrics(trues, preds):
    macro = {
        "precision": precision_score(trues, preds, average="macro", zero_division=0),
        "recall":    recall_score(trues, preds, average="macro", zero_division=0),
        "f1":        f1_score(trues, preds, average="macro", zero_division=0),
        "accuracy":  accuracy_score(trues, preds)
    }
    weighted = {
        "precision": precision_score(trues, preds, average="weighted", zero_division=0),
        "recall":    recall_score(trues, preds, average="weighted", zero_division=0),
        "f1":        f1_score(trues, preds, average="weighted", zero_division=0),
        "accuracy":  accuracy_score(trues, preds)
    }
    return macro, weighted


@torch.no_grad()
def predict(model, loader, device):
    """
    Run prediction on all samples in a DataLoader.

    Returns:
        all_preds: (N,) predicted label index
        all_probs: (N, num_classes) probabilities
        all_trues: (N,) ground-truth labels if available, otherwise -1
    """
    model.eval()
    num_classes = 3

    all_preds = []
    all_probs = []
    all_trues = []

    for seq, feat, t, y in loader:
        seq, feat, t, y = seq.to(device), feat.to(device), t.to(device), y.to(device)

        outs = model(seq, feat)  # list of [B x C]

        B = seq.size(0)
        batch_probs = torch.zeros(B, num_classes, device=device)

        for task_idx in [0, 1]:
            mask = (t == task_idx)
            if mask.any():
                logits = outs[task_idx][mask]  # n_task x C
                probs = F.softmax(logits, dim=-1)
                batch_probs[mask] = probs

        batch_preds = batch_probs.argmax(dim=-1)

        all_preds.extend(batch_preds.cpu().tolist())
        all_probs.append(batch_probs.cpu().numpy())
        all_trues.extend(y.cpu().tolist())

    if len(all_probs) == 0:
        raise ValueError("No data batches were processed. Please check your inputs.")

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds, dtype=int)
    all_trues = np.array(all_trues, dtype=int)

    return all_preds, all_probs, all_trues


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_fasta", type=str, required=True,
                        help="Path to input FASTA or CSV file with RNA sequences.")
    parser.add_argument("--input_fea", type=str, required=True,
                        help="Path to input feature CSV file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output prediction CSV.")

    parser.add_argument("--model_path", type=str, default="best_model.pt",
                        help="Path to trained model state_dict.")
    parser.add_argument("--scaler_mean", type=str, default=None,
                        help="Path to scaler_mean.npy (optional).")
    parser.add_argument("--scaler_scale", type=str, default=None,
                        help="Path to scaler_scale.npy (optional).")

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "If neither sequence CSV nor feature CSV has a 'task' column, "
            "use this task for all samples: 'lncrna' or 'mrna'."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_len", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ----- Seed & device -----
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load sequences (FASTA or CSV) -----
    input_lower = args.input_fasta.lower()

    if input_lower.endswith((".fasta", ".fa", ".fas")):
        # FASTA format: use parse_fasta
        df_seq = parse_fasta(args.input_fasta)  # columns: id, sequence
    else:
        # CSV format: read table; must have a 'sequence' column
        df_seq = pd.read_csv(args.input_fasta)
        if "sequence" not in df_seq.columns:
            raise ValueError(
                "Input file looks like CSV but has no 'sequence' column: "
                f"{args.input_fasta}"
            )
        # If 'id' column is missing, generate one
        if "id" not in df_seq.columns:
            df_seq.insert(0, "id", [f"seq_{i}" for i in range(len(df_seq))])

    # ----- Load features -----
    ft = pd.read_csv(args.input_fea)

    # Align sample count (by row order)
    if len(ft) != len(df_seq):
        print(
            f"[WARN] feature rows ({len(ft)}) != sequence rows ({len(df_seq)}), "
            f"will align by index and use the minimum length."
        )
        min_len = min(len(ft), len(df_seq))
        ft = ft.iloc[:min_len].reset_index(drop=True)
        df_seq = df_seq.iloc[:min_len].reset_index(drop=True)

    # Base DataFrame
    df = df_seq.copy()

    # ----- task column handling: priority is sequence CSV > feature CSV > --task -----
    if "task" in df.columns:
        df["task"] = df["task"].apply(norm_task)
    elif "task" in ft.columns:
        df["task"] = ft["task"].apply(norm_task)
    else:
        if args.task is None:
            raise ValueError(
                "Neither sequence CSV nor feature CSV has a 'task' column, "
                "and --task is not provided. Please set --task lncrna or --task mrna."
            )
        df["task"] = args.task

    # ----- label column (optional, for evaluation) -----
    if "label" in df.columns:
        # Use label already in the sequence table
        pass
    elif "label" in ft.columns:
        df["label"] = ft["label"].values

    # ----- Build feature matrix (drop non-feature columns) -----
    drop_cols = [c for c in ["task", "label", "id", "sequence", "seq"] if c in ft.columns]
    feat_df = ft.drop(columns=drop_cols, errors="ignore")
    X = feat_df.values.astype(np.float32)
    feat_dim = X.shape[1]

    # ----- Standardization (if scaler_mean/scale provided) -----
    scaler_mean = None
    scaler_scale = None

    # If not explicitly specified, try loading from the model directory
    if args.scaler_mean is None or args.scaler_scale is None:
        base_dir = os.path.dirname(os.path.abspath(args.model_path))
        default_mean = os.path.join(base_dir, "scaler_mean.npy")
        default_scale = os.path.join(base_dir, "scaler_scale.npy")
        if os.path.exists(default_mean) and os.path.exists(default_scale):
            args.scaler_mean = default_mean
            args.scaler_scale = default_scale

    if (
        args.scaler_mean is not None
        and args.scaler_scale is not None
        and os.path.exists(args.scaler_mean)
        and os.path.exists(args.scaler_scale)
    ):
        scaler_mean = np.load(args.scaler_mean)
        scaler_scale = np.load(args.scaler_scale)
        X = (X - scaler_mean) / scaler_scale
        print(f"[INFO] Applied StandardScaler using {args.scaler_mean} / {args.scaler_scale}.")
    else:
        print(
            "[WARN] scaler_mean / scaler_scale not found. "
            "Proceeding WITHOUT feature normalization."
        )

    # ----- Dataset & DataLoader -----
    dataset = RNAPredictDataset(df, X, total_len=args.total_len)
    if len(dataset) == 0:
        raise ValueError(
            "No samples loaded (dataset length = 0). "
            "Please check that --input_fasta points to a valid FASTA or CSV and is not empty."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ----- Load model -----
    model = MultiTaskCNNFusion(
        feat_dim=feat_dim,
        cnn_dim=256,      # must match training script
        dropout=0.3,      # must match training script
        emb_dim=64,       # must match training script
        num_classes=3
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded model weights from {args.model_path}.")

    # ----- Prediction -----
    preds, probs, trues = predict(model, loader, device)

    # ----- Optional evaluation (if labels available) -----
    has_labels = np.any(trues >= 0)
    if has_labels:
        mask = trues >= 0
        macro, weighted = compute_metrics(trues[mask], preds[mask])
        print("[INFO] Evaluation on provided labels:")
        print(f"  Macro-F1:    {macro['f1']:.4f}")
        print(f"  Weighted-F1: {weighted['f1']:.4f}")
        print(f"  Accuracy:    {macro['accuracy']:.4f}")

    # ----- Save output -----
    out_df = df_seq.copy()
    out_df["task"] = df["task"]
    if has_labels:
        out_df["label"] = np.where(trues >= 0, trues, np.nan)

    out_df["pred_label"] = preds
    # Probability columns
    for c in range(probs.shape[1]):
        out_df[f"prob_class_{c}"] = probs[:, c]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[INFO] Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
