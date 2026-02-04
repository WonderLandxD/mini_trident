# mini_trident

A minimal version of **[Tident](https://github.com/mahmoodlab/TRIDENT)**. This project focuses on tissue segmentation and patch extraction, removing other components to keep the codebase clean and easy to use.

## Usage

#### 1. Use `segment_slide.py` to process a single whole slide image:

**Command:**
```bash
python segment_slide.py \
  --slide_path /path/to/slide.svs \
  --job_dir /path/to/output \
  --segmenter grandqc \
  --seg_conf_thresh 0.9 \
  --gpu 0 \
  --mag 20 \
  --patch_size 256 \
  --overlap 0 \
  --remove_holes \
  --remove_artifacts \
  --remove_penmarks \
  --save_patches_type tar
```

**Key Arguments:**
- `--slide_path`: Path to the WSI file (.svs, .tif, etc.)
- `--job_dir`: Output directory for all results
- `--segmenter`: Choose `hest` or `grandqc` (default: `grandqc`)
- `--seg_conf_thresh`: Confidence threshold (default: 0.9, try 0.5 for more tissue)
- `--gpu`: GPU index to use
- `--mag`: Magnification level (5, 10, 20, or 40)
- `--patch_size`: Patch size in pixels
- `--overlap`: Overlap between patches in pixels
- `--save_patches_type`: `tar`, `jpg`, or `none` (tar: WebDataset tar file, jpg: individual JPEG patches, none: no patches)

#### 2. Use `list_segment_slide.py` to process a list of whole slide images:

**Command:**
```bash
python list_segment_slide.py \
  --list_json /path/to/slides.json \
  --job_dir /path/to/output \
  --gpu 0 \
  --segmenter grandqc \
  --seg_conf_thresh 0.9 \
  --mag 20 \
  --patch_size 256 \
  --overlap 0 \
  --save_patches_type tar \
  --verbose
```

A json file list of whole slide images should be like this:

**JSON Format:**
```json
[
  {
    "slide_path": "/path/to/slide1.svs",
    "label": "optional_label"
  },
  {
    "slide_path": "/path/to/slide2.svs",
    "label": "optional_label"
  }
]
```

This script processes slides sequentially. If a slide's visualization already exists, it will be skipped.

#### 3. Parallel Processing with Tmux (`run_list_segment_tmux.sh`)

Use `run_list_segment_tmux.sh` to split a JSON list of slides into multiple parts and process them in parallel across different GPUs using tmux sessions:

**Step 1: Edit the script parameters in `run_list_segment_tmux.sh`:**
```bash
# Edit run_list_segment_tmux.sh and set:
list_json="/path/to/slides.json"  # Path to JSON list
job_dir="/path/to/output"          # Output directory
gpus=(0 1 2 3)                     # GPU IDs to use
splits=4                            # Number of splits (should match number of GPUs)
save_patches_type=tar               # tar, jpg, or none
```

**Step 2: Run the script:**
```bash
bash run_list_segment_tmux.sh
```

**How it works:**
1. The script splits the JSON list into `splits` parts
2. Each part is assigned to a tmux session running on a different GPU
3. Each session runs `list_segment_slide.py` on its assigned part
4. You can monitor progress with `tmux ls` and attach to sessions with `tmux attach -t session_name`

**Monitor tmux sessions:**
```bash
# List all sessions
tmux ls

# Attach to a specific session
tmux attach -t segment_0_gpu0

# Detach from session: Press Ctrl+B, then D
```

## Output Structure

After processing, the output directory will contain:

```
job_dir/
├── thumbnails/                    # Slide thumbnails
│   └── <slide_name>.jpg
├── contours_geojson/              # Tissue contours (GeoJSON)
│   └── <slide_name>.geojson
├── contours/                      # Contours overlaid on thumbnails
│   └── <slide_name>.jpg
└── <mag>x_<patch_size>px_<overlap>px_overlap/
    ├── patches/                   # Patch coordinates (HDF5)
    │   └── <slide_name>_patches.h5
    ├── patches_webdataset/        # WebDataset tar files (if --save_patches_type tar)
    │   └── <slide_name>-000000.tar
    ├── patches_jpg/               # Individual JPEG patches (if --save_patches_type jpg)
    │   └── ...
    └── visualization/             # Patch coordinate visualizations
        └── <slide_name>.jpg
```

## WebDataset Format

When using `--save_patches_type tar`, patches are saved as WebDataset tar files. Each tar contains paired samples:

- `<slide>-000000-x1234-y5678.jpg` — patch image (JPEG)
- `<slide>-000000-x1234-y5678.json` — metadata:
  ```json
  {
    "slide": "<slide_name>",
    "x": 1234,
    "y": 5678,
    "patch_size": 256,
    "target_mag": 20,
    "overlap": 0
  }
  ```

**Loading WebDataset patches:**
```python
import webdataset as wds
import json
from PIL import Image
from io import BytesIO

url = "/path/to/job_dir/20x_256px_0px_overlap/patches_webdataset/slide-000000.tar"

def decode_sample(sample):
    img = Image.open(BytesIO(sample["jpg"])).convert("RGB")
    meta = json.loads(sample["json"].decode("utf-8"))
    return img, meta

ds = (
    wds.WebDataset(url)
    .decode()
    .to_tuple("jpg", "json")
    .map(lambda img_bytes, json_bytes: decode_sample({"jpg": img_bytes, "json": json_bytes}))
)

for img, meta in ds:
    print(meta["slide"], meta["x"], meta["y"], img.size)
```

**Combining multiple slides:**
```python
urls = "/path/to/job_dir/**/*.tar"

ds = (
    wds.WebDataset(urls, shardshuffle=True)
    .decode()
    .to_tuple("jpg", "json")
    .map(lambda img_bytes, json_bytes: decode_sample({"jpg": img_bytes, "json": json_bytes}))
    .shuffle(1000)
    .batched(32)
)
```

## Acknowledgments

We thank the authors and developers from MahmoodLab for developing the original TRIDENT repository.
