"""Shared paths and config loading."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
