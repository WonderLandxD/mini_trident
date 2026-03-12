import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class PLIP(nn.Module):
    def __init__(self):
        super().__init__()

        checkpoint_path = "vinid/plip"
        self.backbone = CLIPModel.from_pretrained(checkpoint_path)
        self.processor = CLIPProcessor.from_pretrained(checkpoint_path)
        self.img_transform = self._image_preprocess

    def _image_preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone.get_image_features(pixel_values=x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        raise NotImplementedError("PLIP model does not support image token extraction")


if __name__ == "__main__":
    model = PLIP().to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")
    output = model(x)
    print(output.shape)