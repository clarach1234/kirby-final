#!/usr/bin/env python3
"""Build the final GAVE2 Task-2 vein-topology exploration submission.

The scored reproducible compact submission is the immutable base:

* Task 1 is copied byte-for-byte (R2-V2 R/B + kirby fold0@53160/fold1-4 G0.40).
* Task 3 is copied byte-for-byte (the scored best-field selection).
* Task 2 keeps the base R and Dc G channels byte-for-byte.  Its B channel is
  expanded with the binary vein mask from the existing veinskel10 candidate.

The Task-2 operation is an OR, never a replacement:

    B_new = B_base OR B_veinskel10

Consequently, no currently positive vein pixel can be deleted.  The base is
already in the metric-equivalent compact {0, 1, 255} representation; newly
positive B pixels are stored as 255 while all other byte values are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_SIZE = (1536, 1024)
EXPECTED_NAMES = {
    *(f"Task1/{stem}.png" for stem in EXPECTED_STEMS),
    *(f"Task2/{stem}.png" for stem in EXPECTED_STEMS),
    *(f"Task3/{stem}.txt" for stem in EXPECTED_STEMS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--veinskel10-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{path}: CRC failure in {bad_member}")
        names = {name for name in archive.namelist() if not name.endswith("/")}
        if names != EXPECTED_NAMES:
            raise ValueError(
                f"{path}: expected exactly 150 task entries; "
                f"missing={sorted(EXPECTED_NAMES - names)}, "
                f"extra={sorted(names - EXPECTED_NAMES)}"
            )


def load_rgb(content: bytes, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        if image.mode != "RGB" or image.size != EXPECTED_SIZE:
            raise ValueError(
                f"{name}: expected RGB {EXPECTED_SIZE}, "
                f"got {image.mode} {image.size}"
            )
        return np.asarray(image, dtype=np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return buffer.getvalue()


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 7, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    args = parse_args()
    for path in (args.output, args.manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    validate_zip(args.base_submission)
    validate_zip(args.veinskel10_submission)

    totals = {
        "pixels": 0,
        "base_B_positive": 0,
        "veinskel10_B_positive": 0,
        "final_B_positive": 0,
        "B_added": 0,
        "B_deleted": 0,
        "B_added_inside_Dc_G": 0,
        "B_added_outside_Dc_G": 0,
        "B_added_overlapping_R": 0,
        "R_byte_changes": 0,
        "G_byte_changes": 0,
    }
    per_image: dict[str, dict[str, int]] = {}

    with (
        zipfile.ZipFile(args.base_submission) as base_archive,
        zipfile.ZipFile(args.veinskel10_submission) as skeleton_archive,
        zipfile.ZipFile(
            args.output,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as output_archive,
    ):
        for name in sorted(EXPECTED_NAMES):
            content = base_archive.read(name)
            if not name.startswith("Task2/"):
                output_archive.writestr(fixed_zip_info(name), content)
                continue

            stem = Path(name).stem
            base = load_rgb(content, f"base:{name}")
            skeleton = load_rgb(
                skeleton_archive.read(name),
                f"veinskel10:{name}",
            )

            base_b = base[:, :, 2] >= 128
            skeleton_b = skeleton[:, :, 2] >= 128
            final_b = base_b | skeleton_b
            added = final_b & ~base_b
            deleted = base_b & ~final_b

            final = base.copy()
            final[:, :, 2][added] = 255
            output_archive.writestr(fixed_zip_info(name), encode_png(final))

            image_stats = {
                "B_added": int(added.sum()),
                "B_deleted": int(deleted.sum()),
            }
            per_image[stem] = image_stats
            totals["pixels"] += int(base_b.size)
            totals["base_B_positive"] += int(base_b.sum())
            totals["veinskel10_B_positive"] += int(skeleton_b.sum())
            totals["final_B_positive"] += int(final_b.sum())
            totals["B_added"] += image_stats["B_added"]
            totals["B_deleted"] += image_stats["B_deleted"]
            totals["B_added_inside_Dc_G"] += int(
                (added & (base[:, :, 1] >= 128)).sum()
            )
            totals["B_added_outside_Dc_G"] += int(
                (added & (base[:, :, 1] < 128)).sum()
            )
            totals["B_added_overlapping_R"] += int(
                (added & (base[:, :, 0] >= 128)).sum()
            )
            totals["R_byte_changes"] += int(
                np.count_nonzero(final[:, :, 0] != base[:, :, 0])
            )
            totals["G_byte_changes"] += int(
                np.count_nonzero(final[:, :, 1] != base[:, :, 1])
            )

    if totals["B_deleted"] != 0:
        raise RuntimeError(f"OR operation deleted {totals['B_deleted']} B pixels")
    if totals["R_byte_changes"] or totals["G_byte_changes"]:
        raise RuntimeError("Task 2 R or G changed unexpectedly")

    validate_zip(args.output)
    task1_byte_differences = 0
    task3_byte_differences = 0
    with (
        zipfile.ZipFile(args.base_submission) as base_archive,
        zipfile.ZipFile(args.output) as final_archive,
    ):
        for stem in EXPECTED_STEMS:
            task1_name = f"Task1/{stem}.png"
            task3_name = f"Task3/{stem}.txt"
            task1_byte_differences += int(
                final_archive.read(task1_name) != base_archive.read(task1_name)
            )
            task3_byte_differences += int(
                final_archive.read(task3_name) != base_archive.read(task3_name)
            )
    if task1_byte_differences or task3_byte_differences:
        raise RuntimeError("Task 1 or Task 3 changed unexpectedly")

    manifest = {
        "candidate": args.output.stem,
        "submission_zip": str(args.output.resolve()),
        "submission_zip_sha256": sha256_path(args.output),
        "submission_zip_size_bytes": args.output.stat().st_size,
        "zip_entries": len(EXPECTED_NAMES),
        "Task1": {
            "operation": "copied byte-for-byte from scored reproducible base",
            "measured_scores_for_identical_Task1": [8.19815, 8.23538],
            "composition": {
                "R": "public R2-V2 av artery, CFP-only, 8-view TTA",
                "G": "kirby fold0@53160 plus folds1-4 equal ensemble, threshold 0.40",
                "B": "public R2-V2 av vein, CFP-only, 8-view TTA",
            },
            "byte_differences_from_base": task1_byte_differences,
        },
        "Task2": {
            "operation": "B_new = B_current OR B_veinskel10",
            "R": "current scored endpoint-Db R, byte-identical",
            "G": "current scored Dc G, byte-identical",
            "B": "current scored endpoint B plus veinskel10 positives",
            "statistics": totals,
            "per_image": per_image,
            "measured_score": 8.25683,
        },
        "Task3": {
            "operation": "copied byte-for-byte from scored best-field base",
            "score": 7.5250,
            "byte_differences_from_base": task3_byte_differences,
        },
        "leaderboard_result": {
            "submitted_at_kst": "2026-07-31T21:34:00+09:00",
            "platform_account": "yle",
            "Task1": 8.19815,
            "Task2": 8.25683,
            "Task3": 7.525,
            "overall": 7.95236,
            "preliminary_final_rank": 7,
        },
        "validation": {
            "zip_crc": "passed",
            "entries": 150,
            "Task1_byte_identical": True,
            "Task3_byte_identical": True,
            "Task2_R_byte_identical": totals["R_byte_changes"] == 0,
            "Task2_G_byte_identical": totals["G_byte_changes"] == 0,
            "Task2_B_deletions": totals["B_deleted"],
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Created: {args.output}")
    print(f"SHA-256: {manifest['submission_zip_sha256']}")
    print(f"Size bytes: {manifest['submission_zip_size_bytes']}")
    print(f"Task2 B added/deleted: {totals['B_added']}/{totals['B_deleted']}")
    print(f"Task1 byte differences: {task1_byte_differences}")
    print(f"Task3 byte differences: {task3_byte_differences}")
    print("Measured leaderboard result: 7.95236 (preliminary rank 7)")


if __name__ == "__main__":
    main()
