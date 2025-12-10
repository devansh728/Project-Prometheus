"""
Preprocessing & Dataset Splitting for EV Failure Prediction
============================================================
Creates train/val/test splits by vehicle and prepares datasets for model training.

Usage:
    python preprocessing.py --config ../config.json --input ../synthetic_data --output ../datasets
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class DatasetPreprocessor:
    """Preprocesses aggregated windows and creates train/val/test splits."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.train_vehicles = self.config["train_val_test_split"]["train_vehicles"]
        self.val_vehicles = self.config["train_val_test_split"]["val_vehicles"]
        self.test_vehicles = self.config["train_val_test_split"]["test_vehicles"]

    def load_aggregates(self, input_dir: Path) -> pd.DataFrame:
        """Load aggregated windows from parquet."""
        agg_file = input_dir / "aggregates" / "all_windows.parquet"
        print(f"Loading aggregates from {agg_file}...")
        df = pd.read_parquet(agg_file)
        print(f"  Loaded {len(df):,} windows")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude metadata and labels)."""
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

    def split_by_vehicle(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by vehicle ID."""
        train_df = df[df["vehicle_id"].isin(self.train_vehicles)].copy()
        val_df = df[df["vehicle_id"].isin(self.val_vehicles)].copy()
        test_df = df[df["vehicle_id"].isin(self.test_vehicles)].copy()

        print(f"\nSplit by vehicle:")
        print(
            f"  Train: {len(train_df):,} windows ({len(self.train_vehicles)} vehicles)"
        )
        print(f"  Val:   {len(val_df):,} windows ({len(self.val_vehicles)} vehicles)")
        print(f"  Test:  {len(test_df):,} windows ({len(self.test_vehicles)} vehicles)")

        return train_df, val_df, test_df

    def compute_normalization_stats(
        self, train_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        """Compute mean and std from training data for normalization."""
        stats = {}
        for col in feature_cols:
            if col in train_df.columns:
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()
                # Avoid division by zero
                if std_val == 0 or pd.isna(std_val):
                    std_val = 1.0
                stats[col] = {"mean": float(mean_val), "std": float(std_val)}
        return stats

    def normalize_features(
        self, df: pd.DataFrame, stats: Dict, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Apply z-score normalization using precomputed stats."""
        df_norm = df.copy()
        for col in feature_cols:
            if col in stats and col in df_norm.columns:
                mean_val = stats[col]["mean"]
                std_val = stats[col]["std"]
                df_norm[col] = (df_norm[col] - mean_val) / std_val
        return df_norm

    def handle_missing_values(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Handle missing values in features."""
        df_clean = df.copy()

        # Fill NaN with 0 for numeric features
        for col in feature_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)

        # Replace inf values
        df_clean = df_clean.replace([np.inf, -np.inf], 0)

        return df_clean

    def prepare_xy(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "failure_14d"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target."""
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        return X, y

    def create_window_type_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
        feature_cols: List[str],
    ):
        """Create separate datasets for each window type."""
        for window_type in ["short", "medium", "long"]:
            print(f"\n  Creating {window_type} window datasets...")

            train_wt = train_df[train_df["window_type"] == window_type]
            val_wt = val_df[val_df["window_type"] == window_type]
            test_wt = test_df[test_df["window_type"] == window_type]

            if len(train_wt) > 0:
                # Save full DataFrames
                train_wt.to_parquet(
                    output_dir / f"{window_type}_train.parquet", index=False
                )
                val_wt.to_parquet(
                    output_dir / f"{window_type}_val.parquet", index=False
                )
                test_wt.to_parquet(
                    output_dir / f"{window_type}_test.parquet", index=False
                )

                print(
                    f"    {window_type}: train={len(train_wt):,}, val={len(val_wt):,}, test={len(test_wt):,}"
                )

    def run(self, input_dir: str, output_dir: str):
        """Run the complete preprocessing pipeline."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("EV Telemetry Dataset Preprocessing")
        print("=" * 60)

        # Load data
        df = self.load_aggregates(input_path)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        print(f"\nFeature columns: {len(feature_cols)}")

        # Handle missing values
        print("\nHandling missing values...")
        df = self.handle_missing_values(df, feature_cols)

        # Split by vehicle
        train_df, val_df, test_df = self.split_by_vehicle(df)

        # Compute normalization stats from training data
        print("\nComputing normalization statistics from training data...")
        norm_stats = self.compute_normalization_stats(train_df, feature_cols)

        # Save normalization stats
        stats_file = output_path / "normalization_stats.json"
        with open(stats_file, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  Saved normalization stats to {stats_file}")

        # Normalize all splits using training stats
        print("\nNormalizing features...")
        train_norm = self.normalize_features(train_df, norm_stats, feature_cols)
        val_norm = self.normalize_features(val_df, norm_stats, feature_cols)
        test_norm = self.normalize_features(test_df, norm_stats, feature_cols)

        # Save complete datasets
        print("\nSaving complete datasets...")
        train_norm.to_parquet(output_path / "train_full.parquet", index=False)
        val_norm.to_parquet(output_path / "val_full.parquet", index=False)
        test_norm.to_parquet(output_path / "test_full.parquet", index=False)

        # Create per-window-type datasets
        print("\nCreating per-window-type datasets...")
        self.create_window_type_datasets(
            train_norm, val_norm, test_norm, output_path, feature_cols
        )

        # Create split manifest
        manifest = {
            "train_vehicles": self.train_vehicles,
            "val_vehicles": self.val_vehicles,
            "test_vehicles": self.test_vehicles,
            "train_samples": len(train_norm),
            "val_samples": len(val_norm),
            "test_samples": len(test_norm),
            "feature_columns": feature_cols,
            "label_columns": ["anomaly", "failure_7d", "failure_14d", "severity"],
            "window_types": ["short", "medium", "long"],
            "normalization": "z-score (computed from training data)",
        }

        manifest_file = output_path / "split_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # Print label distribution
        print("\n" + "=" * 60)
        print("Label Distribution")
        print("=" * 60)

        for split_name, split_df in [
            ("Train", train_norm),
            ("Val", val_norm),
            ("Test", test_norm),
        ]:
            print(f"\n{split_name}:")
            print(f"  Total windows: {len(split_df):,}")
            print(
                f"  Anomaly=1: {split_df['anomaly'].sum():,} ({split_df['anomaly'].mean()*100:.1f}%)"
            )
            print(
                f"  Failure_7d=1: {split_df['failure_7d'].sum():,} ({split_df['failure_7d'].mean()*100:.1f}%)"
            )
            print(
                f"  Failure_14d=1: {split_df['failure_14d'].sum():,} ({split_df['failure_14d'].mean()*100:.1f}%)"
            )
            if "severity" in split_df.columns:
                print(
                    f"  Severity distribution: {split_df['severity'].value_counts().to_dict()}"
                )

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print(f"Output directory: {output_path}")
        print("=" * 60)

        return manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess EV telemetry data")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="synthetic_data",
        help="Input directory with aggregates",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets",
        help="Output directory for processed datasets",
    )
    args = parser.parse_args()

    preprocessor = DatasetPreprocessor(args.config)
    preprocessor.run(args.input, args.output)


if __name__ == "__main__":
    main()
