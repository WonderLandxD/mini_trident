#!/usr/bin/env bash
set -euo pipefail

##### parameters #####
list_json="" # path to the JSON list of slides
job_dir="" # directory to store outputs
gpus=(0 1 2 3 4 5 6) # GPU ids
splits=7  # number of splits, which is the same as the number of GPUs.
encoders=(uni_v1 uni_v2 virchow_1 virchow_2) # patch encoder names, loop over each in one tmux
precision="fp16" # fp32 | fp16 | bf16
batch_size=32 # batch size for feature extraction
feat_num_workers=10 # dataloader workers for features


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

##### run the feature extraction on each part in tmux #####
for i in "${!parts[@]}"; do
  part="${parts[$i]}"
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  session="extract_features_${i}_gpu${gpu}"
  tmux new-session -d -s "$session"
  enc_cmds=""
  for enc in "${encoders[@]}"; do
    enc_cmds="${enc_cmds}python list_extract_patch_features.py --list_json \"$part\" --job_dir \"$job_dir\" --gpu $gpu --encoder \"$enc\" --precision \"$precision\" --batch_size $batch_size --feat_num_workers $feat_num_workers --verbose; "
  done
  cmd="cd \"$root_dir\"; ${enc_cmds}"
  tmux send-keys -t "$session" "$cmd" C-m
  echo "tmux: $session -> GPU $gpu, encoders: ${encoders[*]}, batch_size: $batch_size"
done