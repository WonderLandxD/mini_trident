import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Towards a general-purpose foundation model for computational pathology
paper link: https://www.nature.com/articles/s41591-024-02857-3
model weights: https://huggingface.co/MahmoodLab/UNI
"""

class UNI_v1(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from timm.data.transforms_factory import create_transform
        from timm.data import resolve_data_config
        self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.img_transform = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output
    
    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 1:],
        }