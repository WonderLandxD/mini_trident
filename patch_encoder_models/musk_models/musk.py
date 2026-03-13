import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: A Vision-Language Foundation Model for Precision Oncology
paper link: https://www.nature.com/articles/s41586-024-08378-w
model weights: https://huggingface.co/xiangjx/musk
"""


class MUSK(nn.Module):
    def __init__(self):
        super().__init__()

        from timm.models import create_model
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from musk_code import utils, modeling
        import torchvision

        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')

        self.backbone = model
        
        self.img_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(384, interpolation=3, antialias=True),
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            ])

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(
            image=x,
            with_head=False,
            out_norm=True,
            ms_aug=True # 2048-dim
        )[0]

        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        raise NotImplementedError("MUSK model does not support image token extraction")


if __name__ == "__main__":
    from PIL import Image
    model = MUSK().to("cuda")
    img = Image.new("RGB", (224, 224), color=(128, 128, 128)) # 虚假的图片，224x224
    x = model.img_transform(img).unsqueeze(0).to("cuda")
    output = model(x)
    print(output.shape)