import argparse
import json
import os
from glob import glob
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

try:
    import webdataset as wds
except ImportError:
    raise ImportError("webdataset is required. Install with: pip install webdataset")

from patch_encoder_models.model_registry import create_model, list_models


def _parse_json_field(json_bytes):
    """Parse json from bytes, str, or dict."""
    if isinstance(json_bytes, dict):
        return json_bytes
    if isinstance(json_bytes, bytes):
        return json.loads(json_bytes.decode("utf-8"))
    if isinstance(json_bytes, str):
        return json.loads(json_bytes)
    raise TypeError(f"Unsupported json field type: {type(json_bytes)}")


def _samples_from_webdataset(dataset):
    """Yield (img_bytes, meta) from webdataset. Supports two tar formats:
    - Standard: each sample has keys 'jpg' and 'json'.
    - Filename-as-key: one sample has many keys like 'base.jpg', 'base.json'; pair by basename.
    """
    for sample in dataset:
        keys = list(sample.keys())
        if "jpg" in keys and "json" in keys:
            # Standard format (one patch per sample)
            meta = _parse_json_field(sample["json"])
            yield sample["jpg"], meta
        else:
            # Filename-as-key: keys are e.g. 'ibl-000000-x3136-y19712.jpg', '...json'
            jpg_keys = sorted(k for k in keys if k.endswith(".jpg"))
            for jpg_key in jpg_keys:
                base = jpg_key[:-4]  # strip .jpg
                json_key = base + ".json"
                if json_key not in sample:
                    continue
                img_bytes = sample[jpg_key]
                meta = _parse_json_field(sample[json_key])
                yield img_bytes, meta


class WebDatasetDataset(Dataset):
    def __init__(self, tar_paths, transform):
        self.tar_paths = sorted(tar_paths)
        self.transform = transform
        # Pre-load all samples
        self.samples = []
        for tar_path in self.tar_paths:
            dataset = wds.WebDataset(tar_path).decode()
            for img_bytes, meta in _samples_from_webdataset(dataset):
                self.samples.append((img_bytes, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, meta = self.samples[idx]
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, (meta["x"], meta["y"])


def parse_args():
    parser = argparse.ArgumentParser(description="Extract patch features from webdataset for a list of slides.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--list_json", type=str, required=True, help="Path to JSON list with slide names")
    parser.add_argument("--job_dir", type=str, required=True, help="Directory containing webdataset patches")
    parser.add_argument("--encoder", type=str, required=True, choices=list_models(), help="Patch encoder model name")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                        help="Autocast precision for feature extraction")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--feat_num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.list_json, "r") as f:
        slides = json.load(f)
    total = len(slides)

    # Find webdataset directory structure
    patches_webdataset_dir = None
    for root, dirs, files in os.walk(args.job_dir):
        if "patches_webdataset" in root:
            patches_webdataset_dir = root
            break
    if patches_webdataset_dir is None:
        raise ValueError(f"Could not find patches_webdataset directory in {args.job_dir}")
    
    # Determine parent directory for features
    parent_dir = os.path.dirname(patches_webdataset_dir)
    features_base_dir = os.path.join(parent_dir, "patch_features", args.encoder)

    # Load model once
    model = create_model(args.encoder)
    model.eval()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model.to(device)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32
    if device.type != "cuda":
        dtype = torch.float32

    for idx, item in enumerate(slides, start=1):
        # Get slide name from JSON (could be "slide_name" or "slide_path")
        slide_name = item.get("slide_name") or item.get("slide_path")
        if slide_name:
            slide_name = os.path.splitext(os.path.basename(slide_name))[0]
        else:
            # Try to infer from directory structure
            slide_name = item.get("label") or str(item.get("id", idx))
        
        # Handle label-based directory structure
        label = item.get("label")
        if label:
            # If label exists, webdataset is under job_dir/label/<mag>x_.../patches_webdataset/
            label_job_dir = os.path.join(args.job_dir, label)
            patches_webdataset_dir_for_slide = None
            for root, dirs, files in os.walk(label_job_dir):
                if "patches_webdataset" in root:
                    patches_webdataset_dir_for_slide = root
                    break
            if patches_webdataset_dir_for_slide is None:
                print(f"[{idx}/{total}] [SKIP] patches_webdataset not found under {label_job_dir}")
                continue
            slide_dir = os.path.join(patches_webdataset_dir_for_slide, slide_name)
            parent_dir_for_slide = os.path.dirname(patches_webdataset_dir_for_slide)
            features_dir_for_slide = os.path.join(parent_dir_for_slide, "patch_features", args.encoder)
        else:
            # No label, use the global patches_webdataset_dir
            slide_dir = os.path.join(patches_webdataset_dir, slide_name)
            features_dir_for_slide = features_base_dir
        
        if not os.path.exists(slide_dir):
            print(f"[{idx}/{total}] [SKIP] Slide directory not found: {slide_dir}")
            continue
        
        tar_paths = sorted(glob(os.path.join(slide_dir, "*.tar")))
        if not tar_paths:
            print(f"[{idx}/{total}] [SKIP] No tar files found in {slide_dir}")
            continue
        
        features_path = os.path.join(features_dir_for_slide, f"{slide_name}.pth")
        if os.path.exists(features_path):
            print(f"[{idx}/{total}] [SKIP] {slide_name} -> {features_path}")
            continue
        
        print(f"[{idx}/{total}] Processing {slide_name}")
        
        # Load webdataset
        dataset = WebDatasetDataset(tar_paths, model.img_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.feat_num_workers,
            pin_memory=True,
        )
        
        # Extract metadata
        if len(dataset) > 0:
            patch_size = dataset.samples[0][1].get("patch_size", 224)
            patch_size_lv0 = patch_size
        else:
            patch_size = 224
            patch_size_lv0 = 224
        
        # Extract features
        os.makedirs(features_dir_for_slide, exist_ok=True)
        feats_list = []
        coords_list = []
        total_feats = len(dataset)
        offset = 0
        
        for batch_idx, (imgs, coords_batch) in enumerate(dataloader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            device_type = device.type
            autocast_enabled = device_type == "cuda" and dtype != torch.float32
            with torch.inference_mode(), torch.autocast(
                device_type=device_type, dtype=dtype, enabled=autocast_enabled
            ):
                feats = model(imgs)
                if feats.ndim != 2:
                    raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")
            feats_list.append(feats.float().cpu())
            # DataLoader default collate yields (tensor(xs), tensor(ys)); some datasets yield list of (x,y,...)
            if isinstance(coords_batch, (tuple, list)) and len(coords_batch) == 2 and torch.is_tensor(coords_batch[0]):
                coords_list.extend(
                    [(int(x), int(y)) for x, y in zip(coords_batch[0].tolist(), coords_batch[1].tolist())]
                )
            else:
                coords_list.extend([(int(c[0]), int(c[1])) for c in coords_batch])
            offset += feats.shape[0]
            if args.verbose and (batch_idx == 1 or batch_idx % 50 == 0 or offset == total_feats):
                print(f"\r[Feat] {offset}/{total_feats} patches", end="", flush=True)
        if args.verbose:
            print()
        
        feats_tensor = torch.cat(feats_list, dim=0)
        coords_tensor = torch.tensor(coords_list, dtype=torch.int32)
        torch.save(
            {
                "feats": feats_tensor,
                "coords": coords_tensor,
                "model_name": args.encoder,
                "patch_size_level0": patch_size_lv0,
            },
            features_path,
        )
        
        print(f"[{idx}/{total}] Features saved to {features_path}")


if __name__ == "__main__":
    main()
