"""
LightGBM Failure Predictor Training
====================================
Trains LightGBM model for failure prediction with early stopping and evaluation.

Usage:
    python train_lgbm.py --config ../config.json --data ../datasets --output ../models
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")


class LightGBMTrainer:
    """Trains LightGBM models for failure prediction and severity classification."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.lgbm_config = self.config["models"]["lightgbm_failure"]
        self.severity_config = self.config["models"]["lightgbm_severity"]

    def load_data(self, data_dir: Path, window_type: str = "medium"):
        """Load train/val/test datasets for a specific window type."""
        print(f"\nLoading {window_type} window datasets...")

        train = pd.read_parquet(data_dir / f"{window_type}_train.parquet")
        val = pd.read_parquet(data_dir / f"{window_type}_val.parquet")
        test = pd.read_parquet(data_dir / f"{window_type}_test.parquet")

        print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        return train, val, test

    def get_feature_columns(self, df: pd.DataFrame):
        """Get feature columns (exclude metadata and labels)."""
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

    def prepare_data(self, train, val, test, feature_cols, target_col="failure_14d"):
        """Prepare X, y for training."""
        X_train = train[feature_cols].values
        y_train = train[target_col].values

        X_val = val[feature_cols].values
        y_val = val[target_col].values

        X_test = test[feature_cols].values
        y_test = test[target_col].values

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_failure_predictor(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM failure predictor."""
        print("\n" + "=" * 50)
        print("Training LightGBM Failure Predictor")
        print("=" * 50)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Training parameters
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": self.lgbm_config["num_leaves"],
            "learning_rate": self.lgbm_config["learning_rate"],
            "feature_fraction": self.lgbm_config.get("feature_fraction", 0.8),
            "bagging_fraction": self.lgbm_config.get("bagging_fraction", 0.8),
            "bagging_freq": self.lgbm_config.get("bagging_freq", 5),
            "verbose": -1,
            "seed": 42,
        }

        # Train with early stopping
        print(
            f"\nTraining with early stopping (patience={self.lgbm_config['early_stopping_rounds']})..."
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.lgbm_config["n_estimators"],
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self.lgbm_config["early_stopping_rounds"]),
                lgb.log_evaluation(100),
            ],
        )

        print(f"\nBest iteration: {model.best_iteration}")

        return model

    def train_severity_classifier(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM severity classifier (multi-class)."""
        print("\n" + "=" * 50)
        print("Training LightGBM Severity Classifier")
        print("=" * 50)

        # Filter only positive failure samples (severity > 0)
        train_mask = y_train > 0
        val_mask = y_val > 0

        if train_mask.sum() == 0:
            print("  No positive samples for severity training!")
            return None

        X_train_sev = X_train[train_mask]
        y_train_sev = y_train[train_mask]
        X_val_sev = X_val[val_mask] if val_mask.sum() > 0 else X_val[:100]
        y_val_sev = y_val[val_mask] if val_mask.sum() > 0 else y_val[:100]

        # Remap labels to 0-indexed (LightGBM multiclass requires 0 to num_class-1)
        # Original: 1-4 -> Remapped: 0-3
        y_train_sev = y_train_sev - 1
        y_val_sev = y_val_sev - 1

        # Clip to valid range just in case
        y_train_sev = np.clip(y_train_sev, 0, 3)
        y_val_sev = np.clip(y_val_sev, 0, 3)

        print(
            f"  Severity samples - Train: {len(X_train_sev):,} | Val: {len(X_val_sev):,}"
        )
        print(f"  Unique labels (remapped): {np.unique(y_train_sev)}")

        # Create datasets
        train_data = lgb.Dataset(
            X_train_sev, label=y_train_sev, feature_name=feature_names
        )
        val_data = lgb.Dataset(X_val_sev, label=y_val_sev, reference=train_data)

        # Training parameters
        params = {
            "objective": "multiclass",
            "num_class": 4,  # 4 severity levels (0-3)
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.severity_config.get("num_leaves", 32),
            "learning_rate": self.severity_config.get("learning_rate", 0.05),
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.severity_config.get("n_estimators", 500),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        return model

    def evaluate_failure_predictor(self, model, X_test, y_test):
        """Evaluate failure predictor on test set."""
        print("\n" + "=" * 50)
        print("Evaluating Failure Predictor on Test Set")
        print("=" * 50)

        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        print(f"\n  AUC-ROC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Precision at various thresholds
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Find threshold for 80% precision
        idx_80 = np.where(precision >= 0.8)[0]
        if len(idx_80) > 0:
            recall_at_80_precision = recall[idx_80[0]]
            thresh_80 = thresholds[idx_80[0]] if idx_80[0] < len(thresholds) else 0.5
            print(
                f"  Recall @ Precision=0.8: {recall_at_80_precision:.4f} (threshold={thresh_80:.3f})"
            )

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["No Failure", "Failure"]
            )
        )

        metrics = {
            "auc_roc": float(auc),
            "average_precision": float(ap),
            "f1_score": float(f1),
            "threshold": 0.5,
        }

        return metrics

    def get_feature_importance(self, model, feature_names, top_n=20):
        """Get top feature importances."""
        importance = model.feature_importance(importance_type="gain")
        feature_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"\nTop {top_n} Feature Importances:")
        for i, row in feature_imp.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        return feature_imp

    def save_model(self, model, output_dir: Path, name: str):
        """Save model to file."""
        model_path = output_dir / f"{name}.txt"
        model.save_model(str(model_path))
        print(f"\nSaved model to {model_path}")
        return model_path

    def run(self, data_dir: str, output_dir: str, window_type: str = "medium"):
        """Run complete training pipeline."""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("LightGBM Training Pipeline")
        print("=" * 60)

        # Load data
        train, val, test = self.load_data(data_path, window_type)
        feature_cols = self.get_feature_columns(train)
        print(f"Features: {len(feature_cols)}")

        # Prepare failure prediction data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
            train, val, test, feature_cols, "failure_14d"
        )

        # Train failure predictor
        failure_model = self.train_failure_predictor(
            X_train, y_train, X_val, y_val, feature_cols
        )

        # Evaluate
        metrics = self.evaluate_failure_predictor(failure_model, X_test, y_test)

        # Feature importance
        self.get_feature_importance(failure_model, feature_cols)

        # Save failure model
        self.save_model(failure_model, output_path, "lgbm_failure_predictor")

        # Train severity classifier
        X_train_sev, y_train_sev, X_val_sev, y_val_sev, _, _ = self.prepare_data(
            train, val, test, feature_cols, "severity"
        )

        severity_model = self.train_severity_classifier(
            X_train, y_train_sev, X_val, y_val_sev, feature_cols
        )

        if severity_model:
            self.save_model(severity_model, output_path, "lgbm_severity_classifier")

        # Save metrics
        metrics_file = output_path / "lgbm_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save feature list
        features_file = output_path / "feature_columns.json"
        with open(features_file, "w") as f:
            json.dump(feature_cols, f, indent=2)

        print("\n" + "=" * 60)
        print("LightGBM Training Complete!")
        print(f"Output directory: {output_path}")
        print("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM models")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--data", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument(
        "--window", type=str, default="medium", choices=["short", "medium", "long"]
    )
    args = parser.parse_args()

    trainer = LightGBMTrainer(args.config)
    trainer.run(args.data, args.output, args.window)


if __name__ == "__main__":
    main()
