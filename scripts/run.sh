#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BENCHMARK=${BENCHMARK:-}

case "$BENCHMARK" in
  alfworld) exec "$ROOT_DIR/scripts/train_alfworld.sh" "$@" ;;
  webshop) exec "$ROOT_DIR/scripts/train_webshop.sh" "$@" ;;
  *)
    printf 'Set BENCHMARK to alfworld or webshop.\n' >&2
    exit 2
    ;;
esac
