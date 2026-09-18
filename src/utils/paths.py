"""Data-root resolution: explicit paths first, environment variables, defaults.

The dataset lives outside the repository (read-only). Resolution order
for a run:

1. An explicit path (``--data-root`` CLI flag or a non-null ``data.root``
   in the configuration) is used verbatim.
2. ``$DATA_ROOT/<DATASET_DIR>`` — project-specific override variable.
3. ``~/data/<DATASET_DIR>`` — final default, emitted with an explicit
   warning (once per process) so silent fallbacks cannot hide.

No existence check is performed here: dataset access reports missing
paths naturally, keeping resolution independent of data state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import MutableMapping, Optional, Union

PathLike = Union[str, Path]

# Default dataset directory beneath the resolved data root.  The canonical
# test dataset lives at /nishome/charliewang/data/<DATASET_DIR> on the
# test server and is named from the project, not from a version number:
# version information is carried by the manifest itself.
DATASET_DIR = "oled_microstage_inverse_disturbance"

_warned_fallback = False


def _warn_fallback(root: Path) -> None:
    global _warned_fallback
    if _warned_fallback:
        return
    _warned_fallback = True
    print(
        f"[paths] WARNING: $DATA_ROOT is not set; using default data "
        f"root {root} (dataset: {DATASET_DIR})",
        file=sys.stderr,
    )


def resolve_data_root(explicit: Optional[PathLike] = None) -> Path:
    """Resolve the dataset root directory.

    ``explicit`` may be an absolute path, a ``~``-prefixed path, or a
    relative path (resolved against the current working directory).
    Empty strings are treated as unset. Returns a concrete ``Path``
    without checking that it exists.
    """
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser().resolve() / DATASET_DIR
    default = Path.home() / "data" / DATASET_DIR
    _warn_fallback(default)
    return default


def apply_data_root(config: MutableMapping) -> None:
    """Write the resolved root back into ``config["data"]["root"]`` in place.

    ``config["data"]`` must exist. The resolved value is the string form of
    the path returned by :func:`resolve_data_root`, so saved run configs and
    checkpoints are self-contained.
    """
    data = config["data"]
    data["root"] = str(resolve_data_root(data.get("root")))
