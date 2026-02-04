import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: StainNet: Scaling Self-Supervised Foundation Models on Immunohistochemistry and Special Stains for Computational Pathology 
paper link: https://arxiv.org/abs/2512.10326
model weights: https://huggingface.co/JWonderLand/StainNet-Base
"""

class StainNet_Base(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        import torchvision.transforms as transforms
        self.backbone = timm.create_model('hf_hub:JWonderLand/StainNet-Base', pretrained=True)
        self.img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output
    
    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        tokens = self.backbone.forward_features(x)
        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, 1:],
        }