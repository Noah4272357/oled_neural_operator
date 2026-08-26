#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}

cd "$project_root"
exec "$python_bin" scripts/train.py "$@"
