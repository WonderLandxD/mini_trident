import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: A whole-slide foundation model for digital pathology from real-world data
paper link: https://www.nature.com/articles/s41586-024-07441-w
model weights: https://huggingface.co/prov-gigapath/prov-gigapath
"""


class ProvGigaPath(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        import torchvision.transforms as transforms

        self.backbone = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
        )
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
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
