"""Run the pipeline stages in order. Implemented so far: data, train, extract.

Usage (from anywhere):  python scripts/run_all.py [--config configs/default.yaml] [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import DEFAULT_CONFIG, load_config  # noqa: E402
from src import data, extract, train  # noqa: E402


def main() -> None:
    """Parse arguments and run each stage; --force recomputes every stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="recompute all stages even if cached")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data.main(cfg, args.force)
    train.main(cfg, args.force)
    extract.main(cfg, args.force)
    # TODO: reduce -> metrics -> plots


if __name__ == "__main__":
    main()
