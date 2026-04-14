#!/usr/bin/env bash
set -euo pipefail

##### parameters #####
list_json="./xxxxx.json" # path to the JSON list of slides, you should modify this to your own path
job_dir="./mini_trident_datasets/xxxxx" # directory to store outputs, you should modify this to your own path
gpus=(0 1 2 3 4 5 6 7) # GPU ids
splits=8  # number of splits, which is the same as the number of GPUs. 
save_patches_type=tar # type of patches to save, `tar` or `jpg` or `none` (no patches)
encoder="conch_v1_5" # patch encoder name       # Remain: conch_v1_5, virchow_v1, ctranspath
precision="fp16" # fp32 | fp16 | bf16
feat_num_workers=8 # dataloader workers for features
patch_size=512
seg_conf_thresh="${seg_conf_thresh:-0.9}"
min_tissue_proportion="${min_tissue_proportion:-0.9}"
mag="${mag:-20}" # magnification level (default 20x; override by env var mag=40, etc.)
coords_mode="${coords_mode:-tissue}" # tissue | full (full keeps all patches)


# ### for TissueNet dataset, the mpp needs to be set to 0.5 because these wsis don't have the mpp information in the metadata. mpp=0.5 means 20x. mpp=0.25 means 40x. We use 0.5 with 20x magnification to ensure the tile images are from the second lowest level.
# ### For other datasets, the mpp should be set to the value in the metadata.

# ### For BCNB dataset and other datasets with jpg format, we also need to set the mpp because these wsis don't have the mpp information in the metadata. 
# For example, if you want to cut tile patches with its original size, you should set mpp which is aligned with the mag. Like, mpp=0.5 (20x) corresponds to mag=20 or mpp=0.25 (40x) corresponds to mag=40.
# mpp=0.5

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

##### run the segmentation + feature extraction on each part in tmux #####
for i in "${!parts[@]}"; do
  part="${parts[$i]}"
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  session="segment_and_extract_${i}_gpu${gpu}"
  cmd="cd \"$root_dir\"; python list_segment_slide_and_extract_patch_features.py --list_json \"$part\" --job_dir \"$job_dir\" --gpu $gpu --save_patches_type \"$save_patches_type\" --encoder \"$encoder\" --precision \"$precision\" --feat_num_workers $feat_num_workers --mag \"$mag\" --coords_mode \"$coords_mode\" --verbose --patch_size \"$patch_size\" --seg_conf_thresh \"$seg_conf_thresh\" --min_tissue_proportion \"$min_tissue_proportion\"" # --mpp \"$mpp\" if this is needed
  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "$cmd" C-m
  echo "tmux: $session -> GPU $gpu, encoder: $encoder, save_patches_type: $save_patches_type, mag: $mag, coords_mode: $coords_mode, patch_size: $patch_size, seg_conf_thresh: $seg_conf_thresh, min_tissue_proportion: $min_tissue_proportion"
done
