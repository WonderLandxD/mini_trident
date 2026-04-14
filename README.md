# mini_trident

A minimal version of **[Tident](https://github.com/mahmoodlab/TRIDENT)**. This project focuses on tissue segmentation and patch extraction from Whole Slide Images (WSIs), removing other components to keep the codebase clean and easy to use.

**Supported inputs:** OpenSlide-backed formats (.svs, .tif, etc.), `.sdpc`, and standard images (.png, .jpg, .jpeg, .bmp, .webp, .tif).

- For standard images or when slide metadata has no MPP, pass `--mpp` (e.g. `--mpp 0.5` for 20×).
- Reading `.sdpc` requires `opensdpc` (see error message in `tissue_segmentation/wsi.py`).

## Usage

#### 1) Single slide: tissue segmentation + coords + (optional) patches

Use `segment_slide.py`.

**Command:**
```bash
python segment_slide.py \
  --slide_path /path/to/slide.svs \
  --job_dir /path/to/output \
  --segmenter grandqc \
  --seg_conf_thresh 0.9 \
  --gpu 0 \
  --mag 20 \
  --patch_size 224 \
  --overlap 0 \
  --remove_holes \
  --remove_artifacts \
  --remove_penmarks \
  --min_tissue_proportion 0.9 \
  --save_patches_type tar
# For PNG/JPEG or when slide metadata lacks MPP, add: --mpp 0.5
```

**Key Arguments:**
- `--slide_path`: Path to the WSI file (.svs, .tif, etc.) or standard image (.png, .jpg, etc.; requires `--mpp`)
- `--job_dir`: Output directory for all results
- `--segmenter`: Choose `hest` or `grandqc` (default: `grandqc`)
- `--seg_conf_thresh`: Confidence threshold (default: 0.9, try 0.5 for more tissue)
- `--gpu`: GPU index to use
- `--mag`: Magnification level (5, 10, 20, or 40)
- `--patch_size`: Patch size in pixels
- `--overlap`: Overlap between patches in pixels
- `--mpp`: Override MPP in µm/px (e.g. `0.25` for 40×, `0.5` for 20×). Use when metadata is missing or for PNG/JPEG images.
- `--custom_mpp_keys`: Extra metadata keys to try when reading MPP from slide properties.
- `--save_patches_type`: `tar`, `jpg`, or `none` (tar: WebDataset tar file, jpg: individual JPEG patches, none: no patches)
- `--min_tissue_proportion`: keep patch only if tissue proportion \(\ge\) threshold (0~1)
- `--remove_holes`: Remove holes inside tissue regions (treat holes as background)
- `--remove_artifacts`: Run GrandQC artifact removal after tissue segmentation
- `--remove_penmarks`: Run penmark-only cleanup (overridden if `--remove_artifacts` is set)
- `--verbose`: Show progress bar

#### 2) Single slide: segmentation + coords + (optional) patches + features

Use `segment_slide_and_extract_patch_features.py`.

**Command:**
```bash
python segment_slide_and_extract_patch_features.py \
  --slide_path /path/to/slide.svs \
  --job_dir /path/to/output \
  --gpu 0 \
  --encoder uni_v2 \
  --precision fp16 \
  --feat_num_workers 4 \
  --mag 20 \
  --patch_size 224 \
  --overlap 0 \
  --coords_mode tissue \
  --min_tissue_proportion 0.9 \
  --save_patches_type tar \
  --verbose
```

**Notes:**
- `--coords_mode tissue`: segment tissue first, then tile only tissue region
- `--coords_mode full`: skip tissue mask, tile the whole slide canvas

#### 3) JSON list format

All list-based scripts take a JSON array. Minimal format:

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

#### 4) List: segmentation + coords + (optional) patches

Use `list_segment_slide.py`.

**Command:**
```bash
python list_segment_slide.py \
  --list_json /path/to/slides.json \
  --job_dir /path/to/output \
  --gpu 0 \
  --segmenter grandqc \
  --seg_conf_thresh 0.9 \
  --mag 20 \
  --patch_size 224 \
  --overlap 0 \
  --min_tissue_proportion 0.9 \
  --save_patches_type tar \
  --verbose
```

Skip logic: skip when coords HDF5 exists AND requested patch outputs are complete (tar exists / jpg count matches / `none`).

#### 5) List: segmentation + coords + (optional) patches + features

Use `list_segment_slide_and_extract_patch_features.py`.

**Command:**
```bash
python list_segment_slide_and_extract_patch_features.py \
  --list_json /path/to/slides.json \
  --job_dir /path/to/output \
  --gpu 0 \
  --encoder uni_v2 \
  --precision fp16 \
  --mag 20 \
  --patch_size 224 \
  --overlap 0 \
  --coords_mode tissue \
  --min_tissue_proportion 0.9 \
  --save_patches_type tar \
  --verbose
```

Skip logic: skip when feature file exists (`.../patch_features/<encoder>/<slide>.pth`).

#### 6) Extract features from existing WebDataset tars

If you already have `patches_webdataset/`, you can extract features without re-segmentation:

- `extract_patch_features.py`: single slide directory
- `list_extract_patch_features.py`: a JSON list (uses `slide_path` / `slide_name` / `label` to locate slide folder)

Example (list):

```bash
python list_extract_patch_features.py \
  --list_json /path/to/slides.json \
  --job_dir /path/to/output \
  --gpu 0 \
  --encoder uni_v2 \
  --precision fp16 \
  --batch_size 32 \
  --feat_num_workers 8 \
  --verbose
```

#### 7) Parallel processing (tmux)

Scripts:

- `run_list_segment_tmux.sh`: list segmentation (+ optional patches)
- `run_list_extract_patch_features_tmux.sh`: list feature extraction from existing webdataset tars
- `run_list_segment_and_extract_patch_features_tmux.sh`: list segmentation + feature extraction
- `run_loop_list_segment_and_extract_patch_features_tmux.sh`: loop encoders (seg + feat)
- `run_loop_list_extract_patch_features_tmux.sh`: loop encoders (feat only)

Edit parameters inside the script (paths, GPUs, splits, encoder, etc.), then run:

```bash
bash run_list_segment_tmux.sh
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
    ├── patches_webdataset/        # if --save_patches_type tar
    │   └── <slide_name>/
    │       ├── <slide_name>-000000.tar
    │       └── <slide_name>-000001.tar
    ├── patches_jpg/               # if --save_patches_type jpg
    │   └── <slide_name>/
    │       └── <slide_name>-000000-x1234-y5678.jpg
    ├── patch_features/            # Patch features (PTH)
    │   └── <encoder>/<slide_name>.pth
    └── visualization/             # Patch coordinate visualizations
        └── <slide_name>.jpg
```

## Patch Features Format

Patch features are stored as `.pth` files with:
```python
{
  "feats": Tensor[N, D],
  "coords": Tensor[N, 2],
  "model_name": str,
  "patch_size_level0": int
}
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
    "patch_size": 224,
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

url = "/path/to/job_dir/20x_224px_0px_overlap/patches_webdataset/slide/slide-000000.tar"

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
