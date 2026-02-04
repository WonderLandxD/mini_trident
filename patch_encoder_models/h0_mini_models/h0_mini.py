import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Distilling foundation models for robust and efficient models in digital pathology
paper link: https://arxiv.org/abs/2501.16239
model weights: https://huggingface.co/bioptimus/H0-mini
"""


class H0_Mini(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from timm.data.transforms_factory import create_transform
        from timm.data import resolve_data_config
        from timm.layers import SwiGLUPacked

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
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
            "patch": tokens[:, 5:],
            "reg": tokens[:, 1:5],
        }
