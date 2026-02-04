import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: H-optimus-0
paper link: https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0
model weights: https://huggingface.co/bioptimus/H-optimus-0
"""


class H_optimus_0(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        import torchvision.transforms as transforms

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
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
