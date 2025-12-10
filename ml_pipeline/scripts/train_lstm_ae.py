"""
LSTM Autoencoder Training for Anomaly Detection
================================================
Trains LSTM-AE on healthy data and uses reconstruction error for anomaly detection.

Usage:
    python train_lstm_ae.py --config ../config.json --data ../datasets --output ../models
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import warnings

warnings.filterwarnings("ignore")


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for sequence anomaly detection."""

    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=16):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[1])
        self.decoder_lstm1 = nn.LSTM(hidden_dims[1], hidden_dims[0], batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_dims[0], input_dim, batch_first=True)

    def encode(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.encoder_lstm1(x)
        out, (h, c) = self.encoder_lstm2(out)
        # Use last hidden state
        latent = self.encoder_fc(h.squeeze(0))
        return latent

    def decode(self, latent, seq_len):
        # Expand latent to sequence
        out = self.decoder_fc(latent)
        out = out.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm1(out)
        out, _ = self.decoder_lstm2(out)
        return out

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent, x.size(1))
        return reconstructed


class LSTMAETrainer:
    """Trains LSTM Autoencoder for anomaly detection."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.ae_config = self.config["models"]["lstm_autoencoder"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, data_dir: Path, window_type: str = "short"):
        """Load datasets."""
        print(f"\nLoading {window_type} window datasets...")

        train = pd.read_parquet(data_dir / f"{window_type}_train.parquet")
        val = pd.read_parquet(data_dir / f"{window_type}_val.parquet")
        test = pd.read_parquet(data_dir / f"{window_type}_test.parquet")

        print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        return train, val, test

    def get_feature_columns(self, df: pd.DataFrame):
        """Get feature columns for AE input."""
        meta_cols = ["window_id", "vehicle_id", "window_type", "start_ts", "end_ts"]
        label_cols = [
            "anomaly",
            "failure_7d",
            "failure_14d",
            "severity",
            "failure_type",
            "dtc_count",
        ]
        feature_cols = [c for c in df.columns if c not in meta_cols + label_cols]
        return feature_cols

    def prepare_sequences(self, df, feature_cols, seq_len=10):
        """Prepare sequences for LSTM training."""
        # For simplicity, treat each row as a single time step
        # In production, you'd create sliding windows from raw frames
        X = df[feature_cols].values

        # Create pseudo-sequences by grouping rows
        n_samples = len(X) // seq_len * seq_len
        X = X[:n_samples].reshape(-1, seq_len, len(feature_cols))

        return X.astype(np.float32)

    def filter_healthy_samples(self, train, feature_cols):
        """Filter only healthy (non-anomaly) samples for AE training."""
        healthy = train[train["anomaly"] == 0]
        print(
            f"  Healthy samples for training: {len(healthy):,} ({len(healthy)/len(train)*100:.1f}%)"
        )
        return healthy

    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch."""
        model.train()
        total_loss = 0

        for batch in dataloader:
            x = batch[0].to(self.device)

            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_epoch(self, model, dataloader, criterion):
        """Validate one epoch."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def compute_reconstruction_error(self, model, X):
        """Compute per-sample reconstruction error."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            reconstructed = model(X_tensor)
            # MSE per sample
            errors = ((X_tensor - reconstructed) ** 2).mean(dim=(1, 2))

        return errors.cpu().numpy()

    def train(self, train, val, feature_cols, output_dir: Path):
        """Train LSTM Autoencoder."""
        print("\n" + "=" * 50)
        print("Training LSTM Autoencoder")
        print("=" * 50)

        # Filter healthy samples
        healthy_train = self.filter_healthy_samples(train, feature_cols)

        # Prepare sequences
        seq_len = min(10, len(healthy_train) // 100)
        seq_len = max(seq_len, 2)  # Minimum sequence length

        print(f"  Sequence length: {seq_len}")

        X_train = self.prepare_sequences(healthy_train, feature_cols, seq_len)
        X_val = self.prepare_sequences(val, feature_cols, seq_len)

        print(f"  Training sequences: {X_train.shape}")
        print(f"  Validation sequences: {X_val.shape}")

        # Create model
        input_dim = len(feature_cols)
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dims=self.ae_config["encoder_layers"],
            latent_dim=self.ae_config["latent_dim"],
        ).to(self.device)

        print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Data loaders
        batch_size = self.ae_config["batch_size"]
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)), batch_size=batch_size
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.ae_config["learning_rate"]
        )

        # Training loop
        epochs = self.ae_config["epochs"]
        patience = self.ae_config["early_stopping_patience"]
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n  Training for up to {epochs} epochs (patience={patience})...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self.validate_epoch(model, val_loader, criterion)

            if (epoch + 1) % 10 == 0:
                print(
                    f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), output_dir / "lstm_ae_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(torch.load(output_dir / "lstm_ae_best.pt"))

        return model, seq_len

    def compute_threshold(self, model, X_healthy, percentile=99):
        """Compute anomaly threshold from healthy validation data."""
        errors = self.compute_reconstruction_error(model, X_healthy)
        threshold = np.percentile(errors, percentile)
        print(f"\n  Anomaly threshold ({percentile}th percentile): {threshold:.6f}")
        return threshold

    def evaluate(self, model, test, feature_cols, seq_len, threshold):
        """Evaluate anomaly detector on test set."""
        print("\n" + "=" * 50)
        print("Evaluating Anomaly Detector")
        print("=" * 50)

        X_test = self.prepare_sequences(test, feature_cols, seq_len)

        # Get labels (need to align with sequences)
        n_samples = len(test) // seq_len * seq_len
        y_test = test["anomaly"].values[:n_samples]
        y_test = y_test.reshape(-1, seq_len).max(axis=1)  # Any anomaly in sequence

        # Compute reconstruction errors
        errors = self.compute_reconstruction_error(model, X_test)
        y_pred = (errors > threshold).astype(int)
        y_score = errors / (errors.max() + 1e-8)  # Normalize for AUC

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_score)
        except:
            auc = 0.0

        f1 = f1_score(y_test, y_pred)

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        print(f"\n  AUC-ROC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP={tp:,} | FP={fp:,}")
        print(f"    FN={fn:,} | TN={tn:,}")

        metrics = {
            "auc_roc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "threshold": float(threshold),
        }

        return metrics

    def run(self, data_dir: str, output_dir: str, window_type: str = "short"):
        """Run complete training pipeline."""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("LSTM Autoencoder Training Pipeline")
        print("=" * 60)

        # Load data
        train, val, test = self.load_data(data_path, window_type)
        feature_cols = self.get_feature_columns(train)
        print(f"Features: {len(feature_cols)}")

        # Train model
        model, seq_len = self.train(train, val, feature_cols, output_path)

        # Compute threshold from healthy validation data
        healthy_val = val[val["anomaly"] == 0]
        X_val_healthy = self.prepare_sequences(healthy_val, feature_cols, seq_len)
        threshold = self.compute_threshold(model, X_val_healthy)

        # Evaluate
        metrics = self.evaluate(model, test, feature_cols, seq_len, threshold)

        # Save model config
        model_config = {
            "input_dim": len(feature_cols),
            "hidden_dims": self.ae_config["encoder_layers"],
            "latent_dim": self.ae_config["latent_dim"],
            "seq_len": seq_len,
            "threshold": float(threshold),
            "feature_columns": feature_cols,
        }

        with open(output_path / "lstm_ae_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        # Save metrics
        with open(output_path / "lstm_ae_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("LSTM-AE Training Complete!")
        print(f"Output directory: {output_path}")
        print("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--data", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument(
        "--window", type=str, default="short", choices=["short", "medium", "long"]
    )
    args = parser.parse_args()

    trainer = LSTMAETrainer(args.config)
    trainer.run(args.data, args.output, args.window)


if __name__ == "__main__":
    main()
