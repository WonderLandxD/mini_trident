import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: A generalizable pathology foundation model using a unified knowledge distillation pretraining framework
paper link: https://www.nature.com/articles/s41551-025-01488-4
model weights: https://huggingface.co/majiabo/GPFM
"""

class GPFM(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        import torchvision.transforms as transforms
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="majiabo/GPFM", filename="GPFM.pth")
        backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, img_size=224, init_values=1.0e-05)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        backbone.load_state_dict(state_dict, strict=True)
        self.backbone = backbone
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
            "patch": tokens[:, 5:],
            "reg": tokens[:, 1:5],
        }