import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: PathOrchestra: A comprehensive foundation model for computational pathology with over 100 diverse clinical-grade tasks
paper link: https://www.nature.com/articles/s41746-025-02027-w
model weights: https://huggingface.co/AI4Pathology/PathOrchestra
"""


class PathOrchestra(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from timm.data.transforms_factory import create_transform
        from timm.data import resolve_data_config

        self.backbone = timm.create_model(
            "hf-hub:AI4Pathology/PathOrchestra",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.img_transform = create_transform(
            **resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone)
        )

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 1:],
        }
