#!/usr/bin/env python3
"""Create a compact GAVE2 submission ZIP.

The official Task 1/2 evaluator thresholds every RGB prediction channel at
0.5. Converting uint8 probability PNGs at the exact equivalent boundary
(>=128) preserves every evaluated binary mask while making the PNG files much
smaller. With ``--preserve-zero-condition``, exact zeros are also preserved:
0 remains 0, values 1..127 become 1, and values 128..255 become 255. This is
needed to preserve the evaluator's special R/B-both-zero fallback condition.
Task 3 text files are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_SIZE = (1536, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-zero-condition",
        action="store_true",
        help=(
            "Preserve exact zero locations as well as 0.5-threshold masks. "
            "Use this for full equivalence with the official A/V evaluator."
        ),
    )
    return parser.parse_args()


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 7, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def compact_png(
    content: bytes,
    name: str,
    preserve_zero_condition: bool,
) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(content)) as image:
        if image.mode != "RGB" or image.size != EXPECTED_SIZE:
            raise ValueError(
                f"{name}: expected RGB {EXPECTED_SIZE}, "
                f"got {image.mode} {image.size}"
            )
        original = np.asarray(image, dtype=np.uint8)
    if preserve_zero_condition:
        compact = np.where(
            original >= 128,
            255,
            np.where(original == 0, 0, 1),
        ).astype(np.uint8)
    else:
        compact = np.where(original >= 128, 255, 0).astype(np.uint8)
    mismatch = int(
        np.count_nonzero((original >= 128) != (compact >= 128))
    )
    zero_mismatch = int(
        np.count_nonzero((original == 0) != (compact == 0))
    )
    buffer = io.BytesIO()
    Image.fromarray(compact, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return buffer.getvalue(), mismatch, zero_mismatch


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    expected = {
        *(f"Task1/{stem}.png" for stem in EXPECTED_STEMS),
        *(f"Task2/{stem}.png" for stem in EXPECTED_STEMS),
        *(f"Task3/{stem}.txt" for stem in EXPECTED_STEMS),
    }
    entries: dict[str, bytes] = {}
    binary_mismatch = 0
    zero_condition_mismatch = 0
    with zipfile.ZipFile(args.input) as source:
        if source.testzip() is not None:
            raise ValueError("Input ZIP has a CRC error")
        actual = {
            name for name in source.namelist() if not name.endswith("/")
        }
        if actual != expected:
            raise ValueError("Input ZIP does not have exactly 150 task files")
        for name in sorted(expected):
            content = source.read(name)
            if name.startswith(("Task1/", "Task2/")):
                content, mismatch, zero_mismatch = compact_png(
                    content,
                    name,
                    args.preserve_zero_condition,
                )
                binary_mismatch += mismatch
                zero_condition_mismatch += zero_mismatch
            entries[name] = content

    if binary_mismatch != 0:
        raise RuntimeError(f"Binary prediction mismatch: {binary_mismatch}")
    if args.preserve_zero_condition and zero_condition_mismatch != 0:
        raise RuntimeError(
            f"Exact-zero mismatch: {zero_condition_mismatch}"
        )

    with zipfile.ZipFile(
        args.output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for name in sorted(entries):
            output.writestr(fixed_zip_info(name), entries[name])

    with zipfile.ZipFile(args.output) as output:
        bad_member = output.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Output ZIP CRC failure: {bad_member}")
        actual = {
            name for name in output.namelist() if not name.endswith("/")
        }
        if actual != expected:
            raise RuntimeError("Output ZIP entry set changed after writing")

    print(f"Created: {args.output}")
    print(f"Entries: {len(entries)}")
    print(f"Binary prediction mismatch: {binary_mismatch}")
    print(f"Exact-zero mismatch: {zero_condition_mismatch}")
    print(f"Preserved zero condition: {args.preserve_zero_condition}")
    print(f"Size bytes: {args.output.stat().st_size}")


if __name__ == "__main__":
    main()
