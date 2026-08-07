#!/usr/bin/env python3
"""Raw-input inference and reproducibility CLI for team kirby."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_SIZE = (1536, 1024)
TASK3_KEYS = {
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
}
EXPECTED_NAMES = {
    *(f"Task1/{stem}.png" for stem in EXPECTED_STEMS),
    *(f"Task2/{stem}.png" for stem in EXPECTED_STEMS),
    *(f"Task3/{stem}.txt" for stem in EXPECTED_STEMS),
}
EXPECTED_FINAL_SHA256 = (
    "4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_task3(content: bytes, name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        normalized = line.replace(":", " ")
        key, raw = normalized.rsplit(maxsplit=1)
        if key in values:
            raise ValueError(f"{name}: duplicate Task-3 key {key}")
        values[key] = float(raw)
    if set(values) != TASK3_KEYS:
        raise ValueError(f"{name}: unexpected Task-3 keys {sorted(values)}")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"{name}: non-finite Task-3 value")
    return values


def load_rgb(content: bytes, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        if image.mode != "RGB" or image.size != EXPECTED_SIZE:
            raise ValueError(
                f"{name}: expected RGB {EXPECTED_SIZE}, got {image.mode} {image.size}"
            )
        return np.asarray(image, dtype=np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        buffer, format="PNG", optimize=True, compress_level=9
    )
    return buffer.getvalue()


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 7, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def validate_submission(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    channel_values = {"Task1": set(), "Task2": set()}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure: {bad_member}")
        names = {name for name in archive.namelist() if not name.endswith("/")}
        if names != EXPECTED_NAMES:
            raise ValueError(
                "Submission must contain exactly 150 expected entries; "
                f"missing={sorted(EXPECTED_NAMES - names)}, "
                f"extra={sorted(names - EXPECTED_NAMES)}"
            )
        for stem in EXPECTED_STEMS:
            for task in ("Task1", "Task2"):
                array = load_rgb(archive.read(f"{task}/{stem}.png"), f"{task}/{stem}")
                channel_values[task].update(np.unique(array).tolist())
            parse_task3(archive.read(f"Task3/{stem}.txt"), f"Task3/{stem}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "entries": 150,
        "stems": [EXPECTED_STEMS[0], EXPECTED_STEMS[-1]],
        "image_size": list(EXPECTED_SIZE),
        "channel_values": {
            task: sorted(values) for task, values in channel_values.items()
        },
        "crc": "passed",
        "task3": "50 files, seven finite values each",
    }


def check_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA-256 mismatch: expected {expected}, got {actual}")


def reproduce(args: argparse.Namespace) -> None:
    manifest = load_json(HERE / "artifacts_manifest.json")
    expected_inputs = manifest["exact_reconstruction_inputs"]
    check_hash(args.base, expected_inputs[0]["sha256"], "reproducible base")
    check_hash(args.veinskel, expected_inputs[1]["sha256"], "veinskel10 source")
    validate_submission(args.base)
    validate_submission(args.veinskel)

    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "B_added": 0,
        "B_deleted": 0,
        "R_byte_changes": 0,
        "G_byte_changes": 0,
    }
    with (
        zipfile.ZipFile(args.base) as base_archive,
        zipfile.ZipFile(args.veinskel) as skeleton_archive,
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
            base = load_rgb(content, f"base:{name}")
            skeleton = load_rgb(skeleton_archive.read(name), f"veinskel:{name}")
            base_b = base[:, :, 2] >= 128
            source_b = skeleton[:, :, 2] >= 128
            final_b = base_b | source_b
            added = final_b & ~base_b
            deleted = base_b & ~final_b
            final = base.copy()
            final[:, :, 2][added] = 255
            totals["B_added"] += int(added.sum())
            totals["B_deleted"] += int(deleted.sum())
            totals["R_byte_changes"] += int(
                np.count_nonzero(final[:, :, 0] != base[:, :, 0])
            )
            totals["G_byte_changes"] += int(
                np.count_nonzero(final[:, :, 1] != base[:, :, 1])
            )
            output_archive.writestr(fixed_zip_info(name), encode_png(final))

    result = validate_submission(args.output)
    expected = manifest["expected_output"]
    if result["sha256"] != expected["sha256"]:
        raise RuntimeError(
            f"Output differs from scored artifact: {result['sha256']} != {expected['sha256']}"
        )
    if totals != {
        "B_added": 527170,
        "B_deleted": 0,
        "R_byte_changes": 0,
        "G_byte_changes": 0,
    }:
        raise RuntimeError(f"Unexpected Task-2 composition statistics: {totals}")
    result["task2_or_statistics"] = totals
    print(json.dumps(result, indent=2, ensure_ascii=False))


def verify_weights(weights_root: Path) -> dict[str, object]:
    manifest = load_json(HERE / "weights_manifest.json")
    results = []
    for entry in manifest["local_weights"]:
        path = weights_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_path(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{path}: expected {entry['sha256']}, got {actual}")
        results.append({"path": str(path), "sha256": actual, "status": "verified"})
    for entry in manifest["external_weights"]:
        root = weights_root / entry["path"]
        if "sha256" in entry:
            path = root / "av.pth"
            check_hash(path, entry["sha256"], entry["name"])
            results.append(
                {"path": str(path), "sha256": entry["sha256"], "status": "verified"}
            )
            config = root / "av_config.json"
            check_hash(config, entry["configuration_sha256"], f"{entry['name']} config")
            results.append(
                {
                    "path": str(config),
                    "sha256": entry["configuration_sha256"],
                    "status": "verified",
                }
            )
        for relative, expected in entry.get("files", {}).items():
            path = root / relative
            check_hash(path, expected, f"{entry['name']}:{relative}")
            results.append(
                {"path": str(path), "sha256": expected, "status": "verified"}
            )
    return {"weights_root": str(weights_root.resolve()), "verified": results}


def require_validation_data(data_root: Path) -> dict[str, Path]:
    validation = data_root / "validation"
    folders = {
        "images": validation / "images",
        "masks": validation / "masks",
        "FFA_A": validation / "FFA_A",
        "FFA_AV": validation / "FFA_AV",
    }
    expected = {f"{stem}.png" for stem in EXPECTED_STEMS}
    for label, folder in folders.items():
        actual = {path.name for path in folder.glob("*.png")}
        if actual != expected:
            raise ValueError(
                f"{label}: expected g_051..g_100; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    return folders


def run_checked(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def infer_components(args: argparse.Namespace) -> None:
    verify_weights(args.weights_root)
    folders = require_validation_data(args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    task1_outputs = []
    for fold in range(5):
        output = args.output_root / "task1" / f"fold_{fold}"
        task1_outputs.append(output)
        run_checked([
            python, str(REPO / "get_predictions.py"),
            "--weights", str(args.weights_root / "task1" / f"fold_{fold}.pth"),
            "--images-path", str(folders["images"]),
            "--masks-path", str(folders["masks"]),
            "--a-path", str(folders["FFA_A"]),
            "--av-path", str(folders["FFA_AV"]),
            "--save-path", str(output),
            "--in_channels", "3", "--base_channels", "16", "--k", "5",
            "--tta", "none", "--gpu_id", str(args.gpu_id),
        ])
    run_checked([
        python, str(REPO / "ensemble_predictions.py"),
        "--inputs", *(str(path) for path in task1_outputs),
        "--output", str(args.output_root / "task1" / "equal_5fold"),
    ])

    task2_outputs = []
    for fold in range(3):
        output = args.output_root / "task2" / f"fold_{fold}"
        task2_outputs.append(output)
        run_checked([
            python, str(REPO / "get_predictions.py"),
            "--weights", str(args.weights_root / "task2" / f"fold_{fold}.pth"),
            "--images-path", str(folders["images"]),
            "--masks-path", str(folders["masks"]),
            "--a-path", str(folders["FFA_A"]),
            "--av-path", str(folders["FFA_AV"]),
            "--save-path", str(output),
            "--in_channels", "5", "--base_channels", "16", "--k", "1",
            "--tta", "none", "--gpu_id", str(args.gpu_id),
        ])
    run_checked([
        python, str(REPO / "ensemble_predictions.py"),
        "--inputs", *(str(path) for path in task2_outputs),
        "--output", str(args.output_root / "task2" / "topology_3fold"),
    ])
    print(
        json.dumps(
            {
                "status": "component inference complete",
                "task1_equal_5fold": str(args.output_root / "task1" / "equal_5fold"),
                "task2_topology_3fold": str(
                    args.output_root / "task2" / "topology_3fold"
                ),
                "note": (
                    "Final channel composition and fixed biomarker calibration are "
                    "validated by the exact reconstruction route."
                ),
            },
            indent=2,
        )
    )


def load_rgb_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != EXPECTED_SIZE:
            raise ValueError(
                f"{path}: expected RGB {EXPECTED_SIZE}, got {image.mode} {image.size}"
            )
        return np.asarray(image, dtype=np.uint8)


def save_rgb_path(array: np.ndarray, path: Path) -> None:
    if array.shape != (EXPECTED_SIZE[1], EXPECTED_SIZE[0], 3):
        raise ValueError(f"{path}: unexpected RGB shape {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def require_exact_folder(folder: Path, suffix: str) -> None:
    expected = {f"{stem}{suffix}" for stem in EXPECTED_STEMS}
    actual = {path.name for path in folder.glob(f"*{suffix}")}
    if actual != expected:
        raise ValueError(
            f"{folder}: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def reuse_complete_folder(folder: Path, suffix: str) -> bool:
    """Reuse a complete stage after interruption; delete an incomplete stage."""
    if not folder.exists():
        return False
    try:
        require_exact_folder(folder, suffix)
    except ValueError:
        shutil.rmtree(folder)
        return False
    print(f"Reusing complete stage: {folder}", flush=True)
    return True


def infer_local_model(
    *,
    weight: Path,
    folders: dict[str, Path],
    output: Path,
    input_channels: int,
    iterations: int,
    tta: str,
    gpu_id: int,
) -> None:
    if reuse_complete_folder(output, ".png"):
        return
    run_checked([
        sys.executable,
        str(REPO / "get_predictions.py"),
        "--weights", str(weight),
        "--images-path", str(folders["images"]),
        "--masks-path", str(folders["masks"]),
        "--a-path", str(folders["FFA_A"]),
        "--av-path", str(folders["FFA_AV"]),
        "--save-path", str(output),
        "--in_channels", str(input_channels),
        "--base_channels", "16",
        "--k", str(iterations),
        "--tta", tta,
        "--gpu_id", str(gpu_id),
    ])
    require_exact_folder(output, ".png")


def ensemble_folders(inputs: list[Path], output: Path) -> None:
    if reuse_complete_folder(output, ".png"):
        return
    run_checked([
        sys.executable,
        str(REPO / "ensemble_predictions.py"),
        "--inputs", *(str(path) for path in inputs),
        "--output", str(output),
    ])
    require_exact_folder(output, ".png")


def boost_fold(source: Path, destination: Path) -> None:
    if reuse_complete_folder(destination, ".png"):
        return
    destination.mkdir(parents=True, exist_ok=True)
    for stem in EXPECTED_STEMS:
        image = load_rgb_path(source / f"{stem}.png").copy()
        image[:, :, 1] = np.max(image, axis=2)
        save_rgb_path(image, destination / f"{stem}.png")


def shift_probability_threshold(channel: np.ndarray, threshold: float) -> np.ndarray:
    probability = channel.astype(np.float32) / 255.0
    numerator = probability * (1.0 - threshold)
    denominator = numerator + (1.0 - probability) * threshold
    shifted = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.rint(np.clip(shifted, 0.0, 1.0) * 255.0).astype(np.uint8)


def parse_task3_path(path: Path) -> dict[str, float]:
    return parse_task3(path.read_bytes(), str(path))


def write_task3(path: Path, values: dict[str, float]) -> None:
    ordered = (
        "CRAE",
        "CRVE",
        "AVR",
        "artery_density",
        "vein_density",
        "artery_fractal_dimension",
        "vein_fractal_dimension",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key} {values[key]:.6f}\n" for key in ordered),
        encoding="utf-8",
    )


def build_final_task3(raw_e: Path, training_labels: Path, output: Path) -> None:
    from build_joint_exploration_submission import empirical_quantile_targets

    scored = (
        "AVR",
        "artery_density",
        "vein_density",
        "artery_fractal_dimension",
        "vein_fractal_dimension",
    )
    train_stems = tuple(f"g_{index:03d}" for index in range(1, 51))
    raw_rows = [parse_task3_path(raw_e / f"{stem}.txt") for stem in EXPECTED_STEMS]
    train_rows = [
        parse_task3_path(training_labels / f"{stem}.txt") for stem in train_stems
    ]
    raw_matrix = {
        key: np.asarray([row[key] for row in raw_rows], dtype=np.float64)
        for key in TASK3_KEYS
    }
    train_matrix = {
        key: np.asarray([row[key] for row in train_rows], dtype=np.float64)
        for key in TASK3_KEYS
    }
    quantile = {}
    for key in scored:
        target = empirical_quantile_targets(raw_matrix[key], train_matrix[key])
        quantile[key] = 0.65 * raw_matrix[key] + 0.35 * target

    artery_bounds = (1.268001, 1.491455)
    vein_bounds = (1.294064, 1.470247)
    output.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(EXPECTED_STEMS):
        # The scored best-field builder read the six-decimal q035 text first,
        # then recomputed CRAE from those serialized AVR/CRVE values.
        crve = round(float(raw_matrix["CRVE"][index]), 6)
        avr = round(float(quantile["AVR"][index]), 6)
        values = {
            "CRVE": float(crve),
            "AVR": float(avr),
            "CRAE": float(avr * crve),
            "artery_density": float(raw_matrix["artery_density"][index]),
            "vein_density": float(quantile["vein_density"][index]),
            "artery_fractal_dimension": float(
                np.clip(
                    raw_matrix["artery_fractal_dimension"][index] - 0.013,
                    *artery_bounds,
                )
            ),
            "vein_fractal_dimension": float(
                np.clip(
                    raw_matrix["vein_fractal_dimension"][index] + 0.012,
                    *vein_bounds,
                )
            ),
        }
        write_task3(output / f"{stem}.txt", values)
    require_exact_folder(output, ".txt")


def compact_metric_equivalent(array: np.ndarray) -> np.ndarray:
    return np.where(
        array >= 128,
        255,
        np.where(array == 0, 0, 1),
    ).astype(np.uint8)


def write_final_zip(task1: Path, task2: Path, task3: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for task, folder in (("Task1", task1), ("Task2", task2)):
            for stem in EXPECTED_STEMS:
                array = compact_metric_equivalent(
                    load_rgb_path(folder / f"{stem}.png")
                )
                archive.writestr(
                    fixed_zip_info(f"{task}/{stem}.png"), encode_png(array)
                )
        for stem in EXPECTED_STEMS:
            archive.writestr(
                fixed_zip_info(f"Task3/{stem}.txt"),
                (task3 / f"{stem}.txt").read_bytes(),
            )


def run_all(args: argparse.Namespace) -> None:
    """Run every selected model and deterministic operation from raw inputs."""
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")
        args.output.unlink()
    if args.force and args.work_root.is_dir():
        shutil.rmtree(args.work_root)
    elif args.work_root.is_dir():
        print(f"Resuming from completed stages in {args.work_root}", flush=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    verify_weights(args.weights_root)
    folders = require_validation_data(args.data_root)
    training_labels = args.data_root / "training" / "biomarker"
    expected_training = {f"g_{index:03d}.txt" for index in range(1, 51)}
    actual_training = {path.name for path in training_labels.glob("*.txt")}
    if actual_training != expected_training:
        raise ValueError(
            f"{training_labels}: missing={sorted(expected_training - actual_training)}, "
            f"extra={sorted(actual_training - expected_training)}"
        )

    # Task 1 local vessel specialist: CFP-only, five equal folds.
    task1_folds = []
    for fold in range(5):
        output = args.work_root / "task1" / f"fold_{fold}"
        infer_local_model(
            weight=args.weights_root / "task1" / f"fold_{fold}.pth",
            folders=folders,
            output=output,
            input_channels=3,
            iterations=5,
            tta="none",
            gpu_id=args.gpu_id,
        )
        task1_folds.append(output)
    task1_equal = args.work_root / "task1" / "equal_5fold"
    ensemble_folders(task1_folds, task1_equal)

    # Public R2-V2 AV specialist: pinned source/weight, CFP-only, 8-view TTA.
    r2_root = args.weights_root / "external" / "r2v2"
    if args.r2_av_source is not None:
        r2_av = args.r2_av_source
        require_exact_folder(r2_av, ".png")
        print(f"Using independently regenerated PyTorch-2.8 R2-V2 output: {r2_av}")
    else:
        r2_output = args.work_root / "task1" / "r2v2"
        r2_av = r2_output / "av"
        if not reuse_complete_folder(r2_av, ".png"):
            if r2_output.exists():
                shutil.rmtree(r2_output)
            run_checked([
                sys.executable,
                str(r2_root / "source" / "infer.py"),
                "--cfp_path", str(folders["images"]),
                "--masks_path", str(folders["masks"]),
                "--model_type", "av",
                "--weights_path", str(r2_root),
                "--save_path", str(r2_output),
                "--tta",
                "--use-gave-format",
            ])
            require_exact_folder(r2_av, ".png")

    task1_final = args.work_root / "final_soft" / "Task1"
    task1_final.mkdir(parents=True, exist_ok=True)
    for stem in EXPECTED_STEMS:
        r2 = load_rgb_path(r2_av / f"{stem}.png")
        kirby = load_rgb_path(task1_equal / f"{stem}.png")
        final = np.stack(
            (
                r2[:, :, 0],
                shift_probability_threshold(kirby[:, :, 1], 0.40),
                r2[:, :, 2],
            ),
            axis=-1,
        )
        save_rgb_path(final, task1_final / f"{stem}.png")

    # Task 2 selected converged folds. Dc averages raw folds. Db applies
    # G=max(R,G,B) to each fold before averaging.
    dc_folds, boosted_folds = [], []
    for fold in range(3):
        raw = args.work_root / "task2" / f"converged_fold_{fold}"
        infer_local_model(
            weight=args.weights_root / "task2" / f"fold_{fold}.pth",
            folders=folders,
            output=raw,
            input_channels=5,
            iterations=1,
            tta="none",
            gpu_id=args.gpu_id,
        )
        boosted = args.work_root / "task2" / f"boosted_fold_{fold}"
        boost_fold(raw, boosted)
        dc_folds.append(raw)
        boosted_folds.append(boosted)
    dc = args.work_root / "task2" / "Dc"
    db = args.work_root / "task2" / "Db"
    ensemble_folders(dc_folds, dc)
    ensemble_folders(boosted_folds, db)

    from build_experimental_submission import recover_vein_skeleton
    from build_joint_exploration_submission import endpoint_bridge_vein

    endpoint_args = SimpleNamespace(
        task2_core_threshold=0.50,
        task2_min_gap=2.0,
        task2_max_gap=8.0,
        task2_min_component=3,
        task2_min_large_component=20,
        task2_min_alignment=0.25,
        task2_min_mean_vein=0.12,
        task2_min_mean_vessel=0.25,
        task2_max_red_excess=0.18,
        task2_added_confidence=0.60,
        task2_max_added_ratio=0.0075,
        task2_max_bridges_per_image=64,
    )
    task2_final = args.work_root / "final_soft" / "Task2"
    task2_final.mkdir(parents=True, exist_ok=True)
    endpoint_totals = {"selected_bridges": 0, "changed_pixels": 0}
    skeleton_added = 0
    for stem in EXPECTED_STEMS:
        db_image = load_rgb_path(db / f"{stem}.png")
        endpoint, endpoint_stats = endpoint_bridge_vein(db_image, endpoint_args)
        skeleton, skeleton_stats = recover_vein_skeleton(
            db_image,
            core_threshold=0.50,
            support_threshold=0.30,
            dilation_support_threshold=0.20,
            added_confidence=0.60,
        )
        dc_image = load_rgb_path(dc / f"{stem}.png")
        final = np.stack(
            (endpoint[:, :, 0], dc_image[:, :, 1], endpoint[:, :, 2]), axis=-1
        )
        added = (skeleton[:, :, 2] >= 128) & (final[:, :, 2] < 128)
        final[:, :, 2][added] = 255
        save_rgb_path(final, task2_final / f"{stem}.png")
        endpoint_totals["selected_bridges"] += int(
            endpoint_stats["selected_bridges"]
        )
        endpoint_totals["changed_pixels"] += int(endpoint_stats["changed_pixels"])
        skeleton_added += int(added.sum())

    # Task 3 source E: topology fold 0 + baseline folds 1/2, no TTA.
    e_specs = (
        ("topology_fold0", args.weights_root / "task3" / "task2_e_topology_fold0.pth"),
        ("baseline_fold1", args.weights_root / "task3" / "task2_e_baseline_fold1.pth"),
        ("baseline_fold2", args.weights_root / "task3" / "task2_e_baseline_fold2.pth"),
    )
    e_folds = []
    for name, weight in e_specs:
        output = args.work_root / "task3" / name
        infer_local_model(
            weight=weight,
            folders=folders,
            output=output,
            input_channels=5,
            iterations=1,
            tta="none",
            gpu_id=args.gpu_id,
        )
        e_folds.append(output)
    task2_e = args.work_root / "task3" / "task2_E"
    ensemble_folders(e_folds, task2_e)

    optic_disc = args.work_root / "task3" / "optic_disc"
    if not reuse_complete_folder(optic_disc, ".png"):
        run_checked([
            sys.executable,
            str(HERE / "optic_disc_segformer.py"),
            "--images", str(folders["images"]),
            "--model", str(args.weights_root / "external" / "optic_disc"),
            "--output", str(optic_disc),
            "--gpu-id", str(args.gpu_id),
        ])
        require_exact_folder(optic_disc, ".png")

    task3_raw = args.work_root / "task3" / "raw_E"
    if not reuse_complete_folder(task3_raw, ".txt"):
        run_checked([
            sys.executable,
            str(REPO / "get_biomarker_kaggle.py"),
            "--av-dir", str(task2_e),
            "--disc-dir", str(optic_disc),
            "--output-dir", str(task3_raw),
            "--threshold", "127",
            "--start-index", "51",
        ])
        require_exact_folder(task3_raw, ".txt")
    task3_final = args.work_root / "final_soft" / "Task3"
    build_final_task3(task3_raw, training_labels, task3_final)

    write_final_zip(task1_final, task2_final, task3_final, args.output)
    result = validate_submission(args.output)
    result["raw_end_to_end"] = True
    result["r2v2_runtime"] = (
        "pytorch-2.8 companion output generated from raw CFP"
        if args.r2_av_source is not None
        else "current pipeline interpreter (diagnostic fallback)"
    )
    result["task2_endpoint"] = endpoint_totals
    result["task2_veinskel_added"] = skeleton_added
    result["expected_scored_sha256"] = EXPECTED_FINAL_SHA256
    result["matches_scored_sha256"] = result["sha256"] == EXPECTED_FINAL_SHA256
    report = args.output.with_name("run_all_report.json")
    report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["matches_scored_sha256"] and not args.allow_hash_mismatch:
        raise RuntimeError(
            "Raw end-to-end output did not match the scored artifact hash; "
            "pass --allow-hash-mismatch only for diagnostics"
        )


def system_check() -> None:
    import torch

    details = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(details, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("system-check")

    verify = sub.add_parser("verify-weights")
    verify.add_argument("--weights-root", type=Path, required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--expect-final-hash", action="store_true")

    exact = sub.add_parser("reproduce")
    exact.add_argument("--base", type=Path, required=True)
    exact.add_argument("--veinskel", type=Path, required=True)
    exact.add_argument("--output", type=Path, required=True)
    exact.add_argument("--force", action="store_true")

    infer = sub.add_parser("infer-components")
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--weights-root", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--gpu-id", type=int, default=0)

    full = sub.add_parser(
        "run-all",
        help="Run raw official inputs through every selected model to kirby.zip.",
    )
    full.add_argument("--data-root", type=Path, required=True)
    full.add_argument("--weights-root", type=Path, required=True)
    full.add_argument("--work-root", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--gpu-id", type=int, default=0)
    full.add_argument(
        "--r2-av-source",
        type=Path,
        help=(
            "R2-V2 av/ folder regenerated by the pinned PyTorch-2.8 companion "
            "container. Required for byte-identical mixed-environment output."
        ),
    )
    full.add_argument("--force", action="store_true")
    full.add_argument("--allow-hash-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "system-check":
        system_check()
    elif args.command == "verify-weights":
        print(json.dumps(verify_weights(args.weights_root), indent=2))
    elif args.command == "validate":
        result = validate_submission(args.submission)
        if args.expect_final_hash and result["sha256"] != EXPECTED_FINAL_SHA256:
            raise RuntimeError(
                f"Expected scored artifact {EXPECTED_FINAL_SHA256}, got {result['sha256']}"
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "reproduce":
        reproduce(args)
    elif args.command == "infer-components":
        infer_components(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
