"""Acceptance STEP 0: print and save the execution environment record.

Writes ``environment.json`` (machine-readable) and ``environment.txt``
(human-readable) into the acceptance run directory, and prints a concise table
to the terminal.

No credentials are collected: no IP or MAC address, no SSH material, no tokens,
and no environment variables are dumped.  Only versions, model names, sizes and
git revisions are recorded.

The two project-local virtual environments are described separately, because
they are genuinely different: data generation (simulation repo) needs Exudyn
and SciPy, while surrogate training needs PyTorch and is CPU-only on this
server.  Nothing here implies GPU execution of the surrogate.

Usage:
    python scripts/show_environment.py --run-dir runs/acceptance_20260918-120000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import apply_data_root  # noqa: E402

DEFAULT_SIMULATION_REPO = Path("/nishome/charliewang/forge-projects/oled-microstage-simulation")
DEFAULT_CONFIG = Path("configs/acceptance.yaml")


def _run(command: List[str], timeout: int = 20) -> Optional[str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _os_release() -> Dict[str, str]:
    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        release = {}
    return {
        "os": release.get("PRETTY_NAME") or release.get("NAME", platform.system()),
        "os_version": release.get("VERSION_ID", ""),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_gib() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpu() -> Dict[str, Any]:
    """GPU facts from nvidia-smi, or an explicit 'none visible' record."""
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if not query:
        return {"count": 0, "devices": [], "driver": None, "cuda_driver_reported": None}

    devices = []
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            devices.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    driver = devices[0]["driver"] if devices else None

    cuda = None
    header = _run(["nvidia-smi"])
    if header:
        for token in header.replace("|", " ").split():
            if token.startswith("12.") or token.startswith("11."):  # e.g. 12.2
                cuda = token
                break
        for line in header.splitlines():
            if "CUDA Version" in line:
                cuda = line.split("CUDA Version:")[1].split()[0].strip()
                break
    return {
        "count": len(devices),
        "devices": devices,
        "driver": driver,
        "cuda_driver_reported": cuda,
    }


def _surrogate_environment() -> Dict[str, Any]:
    """Versions of the interpreter running this script."""
    import numpy  # noqa: PLC0415

    entry: Dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": numpy.__version__,
    }
    try:
        import torch  # noqa: PLC0415

        entry.update(
            {
                "torch": torch.__version__,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "torch_device_count": int(torch.cuda.device_count()),
            }
        )
    except ImportError:
        entry["torch"] = "NOT INSTALLED"
    for name in ("h5py", "matplotlib"):
        try:
            module = __import__(name)
            entry[name] = getattr(module, "__version__", "UNKNOWN")
        except ImportError:
            entry[name] = "NOT INSTALLED"
    return entry


def _simulation_environment(repo: Path) -> Dict[str, Any]:
    """Versions of the simulation repository's own virtual environment."""
    python = repo / ".venv" / "bin" / "python"
    if not python.is_file():
        return {"python_executable": str(python), "status": "NOT FOUND"}
    probe = (
        "import json,sys\n"
        "out={'python':sys.version.split()[0]}\n"
        "for n in ('numpy','scipy','h5py','exudyn'):\n"
        "    try:\n"
        "        m=__import__(n); out[n]=getattr(m,'__version__','UNKNOWN')\n"
        "    except Exception:\n"
        "        out[n]='NOT INSTALLED'\n"
        "print(json.dumps(out))\n"
    )
    raw = _run([str(python), "-c", probe])
    entry: Dict[str, Any] = {"python_executable": str(python), "repo": str(repo)}
    if not raw:
        entry["status"] = "PROBE FAILED"
        return entry
    try:
        entry.update(json.loads(raw.splitlines()[-1]))
    except (ValueError, IndexError):
        entry["status"] = "PROBE FAILED"
    return entry


def _git_head(repo: Path) -> Optional[str]:
    if not (repo / ".git").exists():
        return None
    return _run(["git", "-C", str(repo), "rev-parse", "HEAD"])


def collect(config_path: Path, simulation_repo: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    release = _os_release()
    return {
        "system": {
            **release,
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "total_ram_gib": _total_ram_gib(),
        },
        "gpu": _gpu(),
        "surrogate_environment": _surrogate_environment(),
        "simulation_environment": _simulation_environment(simulation_repo),
        "code": {
            "surrogate_repo": str(PROJECT_ROOT),
            "surrogate_head": _git_head(PROJECT_ROOT),
            "simulation_repo": str(simulation_repo),
            "simulation_head": _git_head(simulation_repo),
            "active_config": str(config_path),
            "config_name": config_path.name,
            "canonical_dataset": str(config["data"]["root"]),
            "canonical_device": str(config["device"]),
        },
    }


def format_report(env: Dict[str, Any]) -> str:
    system = env["system"]
    gpu = env["gpu"]
    surrogate = env["surrogate_environment"]
    simulation = env["simulation_environment"]
    code = env["code"]
    lines = [
        "Execution environment",
        "",
        "SYSTEM",
        f"  OS                  : {system['os']}",
        f"  Kernel              : {system['kernel']}",
        f"  Architecture        : {system['architecture']}",
        f"  CPU                 : {system['cpu_model']}",
        f"  Logical CPUs        : {system['logical_cpus']}",
        f"  RAM                 : {system['total_ram_gib']} GiB",
        "",
        "GPU",
        f"  Count               : {gpu['count']}",
    ]
    for index, device in enumerate(gpu["devices"]):
        lines.append(f"  Device {index}            : {device['name']}, {device['memory']}")
    lines += [
        f"  NVIDIA driver       : {gpu['driver']}",
        f"  CUDA (driver)       : {gpu['cuda_driver_reported']}",
        "",
        "SURROGATE ENVIRONMENT (this process)",
        f"  Python              : {surrogate.get('python')}",
        f"  PyTorch             : {surrogate.get('torch')}",
        f"  torch CUDA available: {surrogate.get('torch_cuda_available')}",
        f"  torch CUDA version  : {surrogate.get('torch_cuda_version')}",
        f"  NumPy               : {surrogate.get('numpy')}",
        f"  h5py                : {surrogate.get('h5py')}",
        f"  matplotlib          : {surrogate.get('matplotlib')}",
        "",
        "SIMULATION ENVIRONMENT (data generation)",
        f"  Python              : {simulation.get('python', simulation.get('status', '?'))}",
        f"  NumPy               : {simulation.get('numpy', '-')}",
        f"  SciPy               : {simulation.get('scipy', '-')}",
        f"  h5py                : {simulation.get('h5py', '-')}",
        f"  Exudyn              : {simulation.get('exudyn', '-')}",
        "",
        "CODE",
        f"  Surrogate repo      : {code['surrogate_repo']}",
        f"  Surrogate HEAD      : {code['surrogate_head']}",
        f"  Simulation repo     : {code['simulation_repo']}",
        f"  Simulation HEAD     : {code['simulation_head']}",
        f"  Active config       : {code['active_config']}",
        f"  Canonical dataset   : {code['canonical_dataset']}",
        f"  Canonical device    : {code['canonical_device']}",
        "",
        "NOTE: the surrogate is trained on CPU; the GPUs above are reported as",
        "present hardware and are not used by the acceptance pipeline.",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--simulation-repo", type=Path, default=DEFAULT_SIMULATION_REPO)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    apply_data_root(config)
    env = collect(args.config, args.simulation_repo, config)

    report = format_report(env)
    print(report)

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = config["experiment"].get("run_dir")
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "environment.json").write_text(
            json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "environment.txt").write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {run_dir / 'environment.json'}")
        print(f"wrote {run_dir / 'environment.txt'}")
    return env


if __name__ == "__main__":
    main()
