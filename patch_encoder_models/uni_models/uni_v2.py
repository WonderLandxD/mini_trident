import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Towards a general-purpose foundation model for computational pathology
paper link: https://www.nature.com/articles/s41591-024-02857-3
model weights: https://huggingface.co/MahmoodLab/UNI2-h
"""

class UNI_v2(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from timm.data.transforms_factory import create_transform
        from timm.data import resolve_data_config
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.img_transform = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output
    
    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 9:],
            "reg": tokens[:, 1:9],
        }