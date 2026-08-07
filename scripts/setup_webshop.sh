#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${WEBSHOP_DATASET:-small}
cd "$ROOT_DIR/agent_system/environments/env_package/webshop/webshop"
bash ./setup.sh -d "$DATASET"
