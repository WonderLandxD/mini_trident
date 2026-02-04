#!/usr/bin/env bash
set -euo pipefail

##### parameters #####
list_json="" # path to the JSON list of slides
job_dir="" # directory to store outputs
gpus=(0 1 2 3) # GPU ids
splits=4  # number of splits, which is the same as the number of GPUs. 
save_patches_type=tar # type of patches to save, `tar` or `jpg` or `none` (no patches)


##### split the list of slides into multiple parts #####
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
split_dir="${job_dir}/_splits"
mkdir -p "$split_dir"

python - <<'PY' "$list_json" "$splits" "$split_dir"
import json, math, sys
from pathlib import Path

list_json = Path(sys.argv[1])
splits = int(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

items = json.loads(list_json.read_text(encoding="utf-8"))
chunk = math.ceil(len(items) / splits) if items else 1
for i in range(splits):
    out = out_dir / f"part_{i:03d}.json"
    out.write_text(json.dumps(items[i*chunk:(i+1)*chunk], ensure_ascii=False), encoding="utf-8")
PY

##### get the list of parts #####
shopt -s nullglob
parts=("$split_dir"/part_*.json)
shopt -u nullglob

##### run the segmentation on each part in the tmux session #####
for i in "${!parts[@]}"; do
  part="${parts[$i]}"
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  session="segment_${i}_gpu${gpu}"
  cmd="cd \"$root_dir\"; python list_segment_slide.py --list_json \"$part\" --job_dir \"$job_dir\" --gpu $gpu --save_patches_type \"$save_patches_type\" --verbose"
  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "$cmd" C-m
  echo "tmux: $session -> GPU $gpu, save_patches_type: $save_patches_type"
done
