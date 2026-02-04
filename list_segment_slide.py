import argparse
import json
import os
from glob import glob

from tissue_segmentation import load_wsi, segmentation_model_factory
from tissue_segmentation.io_utils import read_coords


def parse_args():
    parser = argparse.ArgumentParser(description="Run tissue segmentation on a list of WSIs.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use for processing tasks")
    parser.add_argument("--list_json", type=str, required=True, help="Path to the JSON list of slides")
    parser.add_argument("--job_dir", type=str, required=True, help="Directory to store outputs")
    parser.add_argument("--mag", type=int, choices=[5, 10, 20, 40], default=20,
                        help="Magnification at which downstream patching would run (kept for CLI parity).")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size (kept for CLI parity).")
    parser.add_argument('--segmenter', type=str, default='grandqc',
                        choices=['hest', 'grandqc'],
                        help='Type of tissue vs background segmenter. Options are HEST or GrandQC. GrandQC is recommended for most cases because it is lightweight and fast.')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.9,
                        help='Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.9. Try 0.5 as 2nd option.')
    parser.add_argument('--remove_holes', action='store_true', default=False,
                        help='Remove holes inside tissue regions (treat holes as background).')
    parser.add_argument('--remove_artifacts', action='store_true', default=False,
                        help='Run GrandQC artifact removal after tissue segmentation.')
    parser.add_argument('--remove_penmarks', action='store_true', default=False,
                        help='Run penmark-only cleanup (overridden if --remove_artifacts is set).')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                        help='Custom keys used to store the resolution as MPP in slide metadata.')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Absolute overlap for patching in pixels (kept for CLI parity).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for segmentation inference.')
    parser.add_argument("--verbose", action="store_true", help="Show patch-level progress bar")
    parser.add_argument('--save_patches_type', type=str, default='none',
                        choices=['tar', 'jpg', 'none'],
                        help='Save patches as tar, jpg, or none.')
    parser.add_argument('--min_tissue_proportion', type=float, default=0.9,
                        help='Minimum tissue proportion for patch extraction. If the tissue proportion is less than this value, the patch will not be saved. Between 0. and 1.0. Default is 0.9.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.job_dir, exist_ok=True)
    with open(args.list_json, "r") as f:
        slides = json.load(f)
    total = len(slides)

    # Load segmentation models once and reuse for all slides
    segmentation_model = segmentation_model_factory(
        model_name=args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
    )
    artifact_remover_model = None
    if args.remove_artifacts or args.remove_penmarks:
        artifact_remover_model = segmentation_model_factory(
            'grandqc_artifact',
            remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
        )

    for idx, item in enumerate(slides, start=1):
        slide_path = item["slide_path"]
        label = item.get("label")
        job_dir = os.path.join(args.job_dir, label) if label else args.job_dir
        os.makedirs(job_dir, exist_ok=True)
        save_coords_dir = os.path.join(
            job_dir,
            f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
        )
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        coords_path = os.path.join(save_coords_dir, "patches", f"{slide_name}_patches.h5")
        patches_root = os.path.join(
            save_coords_dir,
            "patches_webdataset" if args.save_patches_type == "tar" else "patches_jpg",
        )
        if args.save_patches_type == "tar":
            patches_done = len(glob(os.path.join(patches_root, slide_name, "*.tar"))) > 0
        elif args.save_patches_type == "jpg":
            if os.path.exists(coords_path):
                _, coords = read_coords(coords_path)
                expected = len(coords)
                patches_done = len(glob(os.path.join(patches_root, slide_name, "*.jpg"))) == expected
            else:
                patches_done = False
        else:
            patches_done = True
        if os.path.exists(coords_path) and patches_done:
            print(f"[{idx}/{total}] [SKIP] {slide_path} -> {coords_path}")
            continue
        print(f"[{idx}/{total}] {slide_path}")

        slide = load_wsi(slide_path=slide_path, custom_mpp_keys=args.custom_mpp_keys)

        ##### Step 1: Segment the tissue with the segmentation model (saved in GeoJSON format) #####
        result = slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=job_dir,
            device=f"cuda:{args.gpu}",
            holes_are_tissue=not args.remove_holes,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        if artifact_remover_model is not None:
            result = slide.segment_tissue(
                segmentation_model=artifact_remover_model,
                target_mag=artifact_remover_model.target_mag,
                holes_are_tissue=False,
                job_dir=job_dir
            )

        ##### Step 2: Extract the tissue coordinates (saved in HDF5 format) #####
        coords_path = slide.extract_tissue_coords(
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords_dir,
            overlap=args.overlap,
            min_tissue_proportion=args.min_tissue_proportion,
        )
        

        ##### Step 3: Save the tissue tile images (saved in WebDataset or JPG format) #####
        _, coords = read_coords(coords_path)
        coords_to_keep = [tuple(map(int, xy)) for xy in coords]
        slide.save_patches(
            coords_to_keep=coords_to_keep,
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords_dir,
            overlap=args.overlap,
            min_tissue_proportion=args.min_tissue_proportion,
            save_patches_type=args.save_patches_type,
            save_patches_verbose=args.verbose,
        )

        ##### Step 4: Visualize the tissue coordinates (saved in JPG format) #####
        viz_dir = os.path.join(save_coords_dir, "visualization")
        viz_path = slide.visualize_coords(coords_path, viz_dir)
        print(f"[{idx}/{total}] coords: {coords_path}, viz: {viz_path}")
        print(f"[{idx}/{total}] done -> {result}")


if __name__ == "__main__":
    main()
