import torch
import torch.nn as nn
import torch.nn.functional as F



class Conch_v1(nn.Module):
    def __init__(self):
        super().__init__()

        from .open_clip_custom import create_model_from_pretrained
        self.backbone, self.img_transform = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path="hf_hub:MahmoodLab/conch",
        )

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone.encode_image(x, proj_contrast=False, normalize=False)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        raise NotImplementedError("Conch model does not support image token extraction")


if __name__ == "__main__":
    model = Conch_v1().to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")
    output = model(x)
    print(output.shape)
