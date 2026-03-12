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


class WebDatasetDataset(Dataset):
    def __init__(self, tar_paths, transform):
        self.tar_paths = sorted(tar_paths)
        self.transform = transform
        # Pre-load all samples
        self.samples = []
        for tar_path in self.tar_paths:
            dataset = wds.WebDataset(tar_path).decode().to_tuple("jpg", "json")
            for img_bytes, json_bytes in dataset:
                meta = json.loads(json_bytes.decode("utf-8"))
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
    parser = argparse.ArgumentParser(description="Extract patch features from webdataset.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--job_dir", type=str, required=True, help="Directory containing webdataset patches")
    parser.add_argument("--slide_name", type=str, required=True, help="Slide name (directory name in patches_webdataset)")
    parser.add_argument("--encoder", type=str, required=True, choices=list_models(), help="Patch encoder model name")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                        help="Autocast precision for feature extraction")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--feat_num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find webdataset directory structure
    for root, dirs, files in os.walk(args.job_dir):
        if "patches_webdataset" in root:
            patches_webdataset_dir = root
            break
    else:
        raise ValueError(f"Could not find patches_webdataset directory in {args.job_dir}")
    
    slide_dir = os.path.join(patches_webdataset_dir, args.slide_name)
    if not os.path.exists(slide_dir):
        raise ValueError(f"Slide directory not found: {slide_dir}")
    
    tar_paths = sorted(glob(os.path.join(slide_dir, "*.tar")))
    if not tar_paths:
        raise ValueError(f"No tar files found in {slide_dir}")
    
    # Determine save directory from job_dir structure
    # Assume structure: job_dir/<mag>x_<patch_size>px_<overlap>px_overlap/patches_webdataset/...
    parent_dir = os.path.dirname(patches_webdataset_dir)
    features_dir = os.path.join(parent_dir, "patch_features", args.encoder)
    features_path = os.path.join(features_dir, f"{args.slide_name}.pth")
    
    if os.path.exists(features_path):
        print(f"[SKIP] Features exist: {features_path}")
        return
    
    # Load model
    model = create_model(args.encoder)
    model.eval()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model.to(device)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32
    if device.type != "cuda":
        dtype = torch.float32
    
    # Load webdataset
    dataset = WebDatasetDataset(tar_paths, model.img_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.feat_num_workers,
        pin_memory=True,
    )
    
    # Extract metadata from first sample
    if len(dataset) > 0:
        patch_size = dataset.samples[0][1].get("patch_size", 224)
        patch_size_lv0 = patch_size
    else:
        patch_size = 224
        patch_size_lv0 = 224
    
    # Extract features
    os.makedirs(features_dir, exist_ok=True)
    feats_list = []
    coords_list = []
    total = len(dataset)
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
        coords_list.extend([(int(x), int(y)) for x, y in coords_batch])
        offset += feats.shape[0]
        if args.verbose and (batch_idx == 1 or batch_idx % 50 == 0 or offset == total):
            print(f"\r[Feat] {offset}/{total} patches", end="", flush=True)
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
    
    print(f"Features saved to {features_path}")


if __name__ == "__main__":
    main()
