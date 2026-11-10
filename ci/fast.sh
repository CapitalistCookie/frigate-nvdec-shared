#!/bin/bash
# Fast checks, no GPU and no Frigate needed (python3, PyYAML, pytest). Runs in a few seconds.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m py_compile hub.py convert.py tools/parity.py tools/fpstest.py tools/stalltest.py
bash -n start-container.sh
python3 -m pytest -q -p no:cacheprovider tests
echo "FAST OK"
