"""
Fast Aggregate Generator - Run only aggregation on existing raw frames.
Useful when raw frames already exist and you just need to regenerate aggregates.
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_synthetic import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregates from existing raw frames"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="synthetic_data",
        help="Data directory containing raw_frames/",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Fast Aggregate Generator")
    print("=" * 60)

    generator = SyntheticDataGenerator(args.config)
    generator.generate_aggregates_fast(Path(args.data_dir))

    print("\n" + "=" * 60)
    print("Aggregation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
