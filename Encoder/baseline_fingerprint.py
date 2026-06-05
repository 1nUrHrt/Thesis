"""Morgan fingerprint + MLP baseline for DDIP.

Non-graph baseline: convert SMILES to ECFP (Morgan) fingerprints,
concatenate drug-pair fingerprints, train an MLP classifier.

Usage:
    python baseline_fingerprint.py [--fp-radius 2] [--fp-bits 2048]
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


# ─── Fingerprint extraction ───────────────────────────────────────────────


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ─── Dataset ────────────────────────────────────────────────────────────────


class FingerprintDDIDataset(Dataset):
    """DDI pairs with pre-computed Morgan fingerprints."""

    def __init__(self, itc_path: str, id2smiles_path: str, fp_radius=2, fp_bits=2048):
        self.itc = pd.read_csv(itc_path)
        id2smiles = pd.read_csv(id2smiles_path)

        # Build drug_id → fingerprint map
        self.fp_map = {}
        print("Computing Morgan fingerprints...")
        for _, row in tqdm(id2smiles.iterrows(), total=len(id2smiles)):
            drug_id = row.iloc[0]
            smiles = row.iloc[-1]
            self.fp_map[drug_id] = smiles_to_morgan(smiles, fp_radius, fp_bits)

        self.fp_dim = fp_bits

    def __len__(self):
        return len(self.itc)

    def __getitem__(self, idx):
        row = self.itc.iloc[idx]
        d1, d2, label = int(row[0]), int(row[1]), int(row[2])
        fp1 = self.fp_map.get(d1, np.zeros(self.fp_dim, dtype=np.float32))
        fp2 = self.fp_map.get(d2, np.zeros(self.fp_dim, dtype=np.float32))
        # Concatenate pair fingerprints
        fp_pair = np.concatenate([fp1, fp2])
        return fp_pair, label

    @property
    def labels(self):
        return self.itc.iloc[:, -1].values


def collate_fn(batch):
    fps, labels = zip(*batch)
    return torch.tensor(np.stack(fps)), torch.tensor(labels)


# ─── MLP Classifier ────────────────────────────────────────────────────────


class FingerprintMLP(nn.Module):
    """MLP classifier operating on concatenated drug-pair fingerprints."""

    def __init__(self, input_dim: int, num_classes: int = 87, hidden_dim: int = 1024, dp_r: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itc-path", default="data/KnownDDI.csv")
    parser.add_argument("--id2smiles-path", default="data/id2smiles.csv")
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dp-r", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Data
    dataset = FingerprintDDIDataset(args.itc_path, args.id2smiles_path, args.fp_radius, args.fp_bits)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=dataset.labels)

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = FingerprintMLP(
        input_dim=args.fp_bits * 2,
        num_classes=87,
        hidden_dim=args.hidden_dim,
        dp_r=args.dp_r,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for fps, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            fps, labels = fps.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(fps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for fps, labels in val_loader:
                fps, labels = fps.to(args.device), labels.to(args.device)
                logits = model(fps)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step(val_loss / len(val_loader))

        print(
            f"  Train Loss: {train_loss / len(train_loader):.4f}  "
            f"Val Loss: {val_loss / len(val_loader):.4f}  "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/baseline_fingerprint.pt")
            print(f"  ✓ New best F1: {best_f1:.4f}")

    print(f"\nBest Val F1 (Morgan FP + MLP): {best_f1:.4f}")


if __name__ == "__main__":
    main()
