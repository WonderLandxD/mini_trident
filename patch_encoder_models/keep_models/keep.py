import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: Knowledge-enhanced pretraining for vision-language pathology foundation model on cancer diagnosis
paper link: https://www.cell.com/cancer-cell/fulltext/S1535-6108(26)00058-9
model weights: https://huggingface.co/Astaxanthin/KEEP
"""


class KEEP(nn.Module):
    def __init__(self):
        super().__init__()

        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as transforms

        self.backbone = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)

        self.img_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone.encode_image(x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W) 
        raise NotImplementedError("KEEP model does not support image token extraction")


if __name__ == "__main__":
    model = KEEP().to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")
    output = model(x)
    print(output.shape)