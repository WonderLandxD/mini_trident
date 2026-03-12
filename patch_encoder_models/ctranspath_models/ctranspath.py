import torch
import torch.nn as nn
import torchvision.transforms as transforms


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return tuple(x)
        raise ValueError(f"Expected length 2, got {len(x)}")
    return (x, x)


class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[1], (
            f"Input image size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CTransPath(nn.Module):
    def __init__(self):
        super().__init__()

        import timm_ctp
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            repo_id="JWonderLand/CHIEF_unofficial",
            filename="CHIEF_CTransPath.pth",
        )

        self.backbone = timm_ctp.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=ConvStem,
            pretrained=False,
        )
        self.backbone.head = nn.Identity()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict["model"], strict=True)

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        raise NotImplementedError("CTransPath model does not support image token extraction")
