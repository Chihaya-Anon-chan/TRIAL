"""Fail early when a TRIAL benchmark installation is incomplete."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _check_alfworld(errors: list[str]) -> None:
    data_root = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
    if not data_root.is_dir():
        errors.append(
            f"ALFWorld data directory not found: {data_root}. Run scripts/setup_alfworld.sh first."
        )


def _check_webshop(errors: list[str]) -> None:
    data_dir = ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json"):
        if not (data_dir / name).is_file():
            errors.append(f"Missing WebShop data file: {data_dir / name}")
    index_dir = ROOT / "agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k"
    if not index_dir.is_dir() or not any(index_dir.iterdir()):
        errors.append(f"Missing WebShop search index: {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("alfworld", "webshop"), required=True)
    args = parser.parse_args()

    errors: list[str] = []
    _check_alfworld(errors) if args.environment == "alfworld" else _check_webshop(errors)
    if errors:
        raise SystemExit("TRIAL preflight failed:\n- " + "\n- ".join(errors))
    print(f"TRIAL preflight passed for {args.environment}.")


if __name__ == "__main__":
    main()
