"""Test STEP 1: print and optionally save the frozen dataset summary.

Reads ``dataset_manifest.json`` only -- no sample is opened, and no training
dependency is imported, so this is instant.

The summary deliberately reports structure, counts and the time grid, and not
the frozen physical model parameters (mover mass and inertia, actuator mapping
``B``, encoder matrix ``H``): this command is normally run in a terminal that
may be shared, so it keeps those values out of its output.  They remain in the
stored manifest and in every sample's ``/metadata`` group, neither of which
this command prints.

Usage:
    python scripts/inspect_dataset.py --config configs/test.yaml
    python scripts/inspect_dataset.py --config configs/test.yaml \
        --run-dir runs/test_20260918-120000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import apply_data_root  # noqa: E402

SPLIT_LABELS = (("train", "Train"), ("val", "Validation"), ("test", "Test"))


def dataset_summary(root: Path) -> Dict[str, Any]:
    """Collect the test-facing dataset facts from the manifest."""
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No dataset manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fields = manifest.get("fields", {})

    def channels(field: str) -> int:
        entry = fields.get(field)
        shape = entry.get("shape") if isinstance(entry, dict) else None
        return int(shape[-1]) if shape else 0

    time_grid = manifest["time_grid"]
    dt = float(time_grid["dt_s"])
    simulation = manifest.get("simulation", {})
    splits = manifest.get("splits", {})
    problem = manifest.get("recommended_problem") or {}
    return {
        "dataset": root.name,
        "path": str(root),
        "manifest_schema": manifest.get("schema"),
        "hdf5_schema_version": manifest.get("hdf5_schema_version"),
        "complete": manifest.get("complete"),
        "failure_count": manifest.get("failure_count"),
        "splits": {split: len(splits.get(split, [])) for split, _ in SPLIT_LABELS},
        "duration_s": float(manifest.get("generation", {}).get("duration", 0.0)),
        "dt_s": dt,
        "sampling_rate_hz": (1.0 / dt) if dt > 0 else 0.0,
        "time_points": int(time_grid["count"]),
        "preroll_s": float(simulation.get("preroll_s", 0.0)),
        "integration_substeps": int(simulation.get("integration_substeps", 1)),
        "raw_input_channels": {
            "force": channels("force"),
            "encoder_displacement": channels("encoder_displacement"),
            "total": channels("force") + channels("encoder_displacement"),
        },
        "target_channels": {
            "disturbance": channels("disturbance"),
            "total": channels("disturbance"),
        },
        "recommended_problem": problem.get("name"),
        "raw_shape": [int(time_grid["count"]), channels("force") + channels("encoder_displacement")],
        "target_shape": [int(time_grid["count"]), channels("disturbance")],
    }


def format_summary(summary: Dict[str, Any]) -> str:
    raw = summary["raw_input_channels"]
    target = summary["target_channels"]
    splits = summary["splits"]
    lines = [
        "Frozen test dataset",
        f"  Dataset             : {summary['dataset']}",
        f"  Path                : {summary['path']}",
        f"  Manifest schema     : {summary['manifest_schema']}",
        f"  HDF5 schema         : {summary['hdf5_schema_version']}",
        f"  Complete            : {summary['complete']}",
    ]
    lines += [f"  {label:<19s} : {splits[key]}" for key, label in SPLIT_LABELS]
    lines += [
        f"  Duration            : {summary['duration_s']} s",
        f"  Sampling rate       : {summary['sampling_rate_hz']:g} Hz",
        f"  Time points         : {summary['time_points']}",
        f"  Raw input channels  : {raw['total']}"
        f"  (force {raw['force']} + encoder_displacement {raw['encoder_displacement']})",
        f"  Target channels     : {target['total']}  (disturbance {target['disturbance']})",
        f"  Raw task shape      : {summary['raw_shape']} -> {summary['target_shape']}",
        f"  Preroll             : {summary['preroll_s']} s",
        f"  Integration substeps: {summary['integration_substeps']}",
        f"  Recommended problem : {summary['recommended_problem']}",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/test.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="test run directory; writes dataset_summary.json into it",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    apply_data_root(config)

    root = Path(config["data"]["root"])
    summary = dataset_summary(root)
    print(format_summary(summary))
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        path = args.run_dir / "dataset_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return summary


if __name__ == "__main__":
    main()
