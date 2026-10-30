#!/usr/bin/env bash
# Add a trained policy to the library:  jetson/add_policy.sh training/logs/rsl_rl/balance_bot_standup/<run> <name>
# Needs <run>/exported/policy.onnx and policy.pt (written by training/scripts/rsl_rl/play.py). policy.onnx runs on the
# Jetson, policy.pt in the PC sim test (test/run_sim_test.sh). Then pick it with POLICY_NAME=<name>.
set -e
RUN=${1:?usage: $0 <training run dir> <name>}
NAME=${2:?usage: $0 <training run dir> <name>}
SRC="$RUN/exported/policy.onnx"
[ -f "$SRC" ] && [ -f "$RUN/exported/policy.pt" ] || { echo "$SRC not found: export it first (training/scripts/rsl_rl/play.py --checkpoint $RUN/model_<N>.pt)"; exit 1; }
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "name: letters, digits, _ . - only"; exit 1; }
DEST="$(dirname "$0")/policies/$NAME"
[ -e "$DEST" ] && { echo "$DEST already exists"; exit 1; }
mkdir -p "$DEST"
cp "$SRC" "$DEST/policy.onnx"
cp "$RUN/exported/policy.pt" "$DEST/policy.pt"
echo "added $DEST (from $RUN)"
echo "add a line for it to $(dirname "$0")/policies/README.md, then use POLICY_NAME=$NAME"
