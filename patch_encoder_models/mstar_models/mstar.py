import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Title: A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model
paper link: https://www.nature.com/articles/s41467-025-66220-x
model weights: https://huggingface.co/Wangyh/mSTAR
"""


class MSTAR(nn.Module):
    def __init__(self):
        super().__init__()

        import timm
        from torchvision import transforms

        self.backbone = timm.create_model(
                        'hf-hub:Wangyh/mSTAR',
                        pretrained=True,
                        init_values=1e-5, dynamic_img_size=True
                        )
        self.img_transform = transforms.Compose(
                            [
                                transforms.Resize(224),
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


if __name__ == "__main__":
    model = MSTAR().to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")
    output = model(x)
    print(output.shape)