import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Benchmarking Self-Supervised Learning on Diverse Pathology Datasets
paper link: https://arxiv.org/abs/2212.04690
model weights: https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino
"""


class Lunit_P8(nn.Module):
    def __init__(self):
        super().__init__()

        import timm

        self.backbone = timm.create_model(
            "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
            pretrained=True,
            dynamic_img_size=True,
        )
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.img_transform = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 1:],
        }
