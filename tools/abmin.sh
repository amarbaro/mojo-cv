#!/usr/bin/env bash
# abmin.sh OLD_BIN NEW_BIN ROUNDS OP OUT IN... [k=v]: alternate OLD and NEW rounds times, reps=150 each, print the min TIME of each.
# Longer than abbench.py per side, so a short quiet window anywhere in the run shows up in the minimum.
set -euo pipefail
old=$1 new=$2 n=$3; shift 3
bo=1e9 bn=1e9
for _ in $(seq "$n"); do
  to=$("$old" "$@" reps=150 | awk '/TIME/{print $2}'); tn=$("$new" "$@" reps=150 | awk '/TIME/{print $2}')
  bo=$(awk -v a="$bo" -v b="$to" 'BEGIN{print (b<a)?b:a}'); bn=$(awk -v a="$bn" -v b="$tn" 'BEGIN{print (b<a)?b:a}')
done
echo "old $bo ms  new $bn ms"
