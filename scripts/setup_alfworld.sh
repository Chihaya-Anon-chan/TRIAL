#!/usr/bin/env bash
set -euo pipefail

python -m pip install "gymnasium==0.29.1" "stable-baselines3==2.6.0" alfworld
alfworld-download -f
