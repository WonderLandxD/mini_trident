from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from .io_utils import get_dir, get_weights_path, has_internet_connection


class SegmentationModel(torch.nn.Module):
    """
    Lightweight wrapper around a semantic segmentation model.
    Handles common setup (freezing params, preprocessing transforms) and exposes
    a consistent forward interface.
    """

    _has_internet = has_internet_connection()

    def __init__(self, freeze: bool = True, confidence_thresh: float = 0.5, **build_kwargs: Dict[str, Any]):
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh

        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[nn.Module, transforms.Compose]:
        """Construct the model and its eval-time transforms."""
        raise NotImplementedError


class HESTSegmenter(SegmentationModel):
    """DeeplabV3 tissue/background segmenter."""

    def _build(self) -> Tuple[nn.Module, transforms.Compose]:
        from torchvision.models.segmentation import deeplabv3_resnet50

        model_ckpt_name = "deeplabv3_seg_v4.ckpt"
        weights_path = get_weights_path("hest")

        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    "Internet connection not available and checkpoint not found locally.\n"
                    f"Please manually download {model_ckpt_name} from:\n"
                    "https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    "and add its path to local_ckpts.json."
                )

            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = {
            k.replace("model.", ""): v for k, v in checkpoint.get("state_dict", {}).items() if "aux" not in k
        }
        model.load_state_dict(state_dict)

        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 10

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4, (
            f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        )
        logits = self.model(image)["out"]
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(torch.uint8)
        return predictions


class JpegCompressionTransform:
    def __init__(self, quality: int = 80):
        self.quality = quality

    def __call__(self, image):
        import cv2
        import numpy as np
        from PIL import Image

        arr = np.array(image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, arr = cv2.imencode(".jpg", arr, encode_param)
        arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return Image.fromarray(arr)


class GrandQCArtifactSegmenter(SegmentationModel):
    """
    Artifact/penmark removal model from GrandQC (https://www.nature.com/articles/s41467-024-54769-y).
    """

    _class_mapping = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",
        7: "Background",
    }

    def _build(self, remove_penmarks_only: bool = False):
        import segmentation_models_pytorch as smp

        self.remove_penmarks_only = remove_penmarks_only
        model_ckpt_name = "GrandQC_MPP1_state_dict.pth"
        encoder_name = "timm-efficientnet-b0"
        weights_path = get_weights_path("grandqc_artifact")

        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=8,
            activation=None,
        )

        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    "Internet connection not available and checkpoint not found locally.\n"
                    f"Please manually download {model_ckpt_name} from:\n"
                    "https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    "and add its path to local_ckpts.json."
                )

            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 10

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)
        if self.remove_penmarks_only:
            predictions = torch.where((predicted_classes == 4) | (predicted_classes == 7), 0, 1)
        else:
            predictions = torch.where(predicted_classes > 1, 0, 1)
        return predictions.to(torch.uint8)


class GrandQCSegmenter(SegmentationModel):
    """GrandQC tissue detector."""

    def _build(self):
        import segmentation_models_pytorch as smp

        model_ckpt_name = "Tissue_Detection_MPP10.pth"
        encoder_name = "timm-efficientnet-b0"
        weights_path = get_weights_path("grandqc")

        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    "Internet connection not available and checkpoint not found locally.\n"
                    f"Please manually download {model_ckpt_name} from:\n"
                    "https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    "and add its path to local_ckpts.json."
                )

            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=2,
            activation=None,
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 1

        eval_transforms = transforms.Compose(
            [
                JpegCompressionTransform(quality=80),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)
        predictions = (max_probs >= self.confidence_thresh) * (1 - predicted_classes)
        return predictions.to(torch.uint8)


def segmentation_model_factory(
    model_name: str,
    confidence_thresh: float = 0.5,
    freeze: bool = True,
    **build_kwargs,
) -> SegmentationModel:
    """Factory for the supported tissue segmentation models."""
    if model_name == "hest":
        return HESTSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    if model_name == "grandqc":
        return GrandQCSegmenter(freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs)
    if model_name == "grandqc_artifact":
        return GrandQCArtifactSegmenter(freeze=freeze, **build_kwargs)
    raise ValueError(f"Model type {model_name} not supported")
