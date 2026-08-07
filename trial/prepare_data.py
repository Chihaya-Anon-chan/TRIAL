"""Create the minimal prompt parquet files used by online agent training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _rows(size: int, split: str) -> list[dict]:
    return [
        {
            "data_source": "trial",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ]


def write_split(path: Path, size: int, split: str) -> None:
    if size <= 0:
        raise ValueError(f"{split} size must be positive, got {size}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(_rows(size, split)), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()

    write_split(args.output_dir / "train.parquet", args.train_size, "train")
    write_split(args.output_dir / "test.parquet", args.val_size, "validation")
    print(f"Wrote TRIAL agent prompts to {args.output_dir}")


if __name__ == "__main__":
    main()
