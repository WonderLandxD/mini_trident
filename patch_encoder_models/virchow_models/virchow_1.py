import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Virchow
paper link:
model weights: hf-hub:paige-ai/Virchow
"""


class Virchow_1(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from timm.data.transforms_factory import create_transform
        from timm.data import resolve_data_config
        from timm.layers import SwiGLUPacked

        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.img_transform = create_transform(
            **resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone)
        )

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 1:],
        }
