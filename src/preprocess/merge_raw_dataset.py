"""Merge the 3 raw OpenAlex shard exports into a single raw dataset file.

This is STEP 1 of the offline data pipeline:

    Step 1  merge_raw_dataset.py   3 shards            -> 1 raw merged CSV
    Step 2  build_clean_dataset.py 1 raw merged CSV    -> 1 clean CSV + EDA report
    Step 3  build_*_cache.py       1 clean CSV         -> Dataset/dashboard_cache/*.csv
    Step 4  app.py                 dashboard_cache/*   -> visualisation (fast)

The merge is a pure byte-level streaming concatenation:
- the first shard is copied verbatim (keeps its header + BOM),
- every later shard has only its first physical line (the header) skipped.

Because it never parses CSV, it uses almost no memory and is safe even though
some text fields (titles, keywords) contain embedded newlines.

Usage:
    python src/preprocess/merge_raw_dataset.py \
        --input Dataset/ai_works_merge_2000_2009.csv \
                Dataset/ai_works_merge_2010_2019.csv \
                Dataset/ai_works_merge_2020_2025.csv \
        --output Dataset/clean/ai_works_merged_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

CHUNK = 8 * 1024 * 1024  # 8 MB streaming buffer


def _copy_skipping_header(src: Path, dst, *, skip_header: bool) -> None:
    """Stream-copy `src` into open file handle `dst`, optionally dropping line 1."""
    with src.open("rb") as fh:
        if skip_header:
            # Read until the first newline (the header line) and discard it.
            # CSV headers never contain embedded newlines, so this only removes
            # the header (and its leading BOM, if any).
            header = bytearray()
            while True:
                byte = fh.read(1)
                if not byte or byte == b"\n":
                    break
                header.extend(byte)
        # Copy the remaining bytes in large chunks, remembering the final byte.
        last_byte = b""
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            dst.write(block)
            last_byte = block[-1:]
        # Only add a newline if this shard didn't already end with one, so we
        # never inject a phantom blank row between shards.
        if last_byte not in (b"\n", b""):
            dst.write(b"\n")


def merge(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(inputs)} shard(s) -> {output}", flush=True)
    with output.open("wb") as dst:
        for i, src in enumerate(inputs):
            if not src.exists():
                raise FileNotFoundError(f"Input not found: {src}")
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  [{i + 1}/{len(inputs)}] {src.name}  ({size_mb:,.0f} MB)"
                  f"{'  (header kept)' if i == 0 else '  (header dropped)'}", flush=True)
            _copy_skipping_header(src, dst, skip_header=(i != 0))
    out_mb = output.stat().st_size / 1024 / 1024
    print(f"Done. Wrote {output}  ({out_mb:,.0f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True,
                        help="Raw shard CSV files, in chronological order.")
    parser.add_argument("--output", default="Dataset/clean/ai_works_merged_raw.csv",
                        help="Destination merged CSV.")
    args = parser.parse_args()
    merge([Path(p) for p in args.input], Path(args.output))


if __name__ == "__main__":
    main()
