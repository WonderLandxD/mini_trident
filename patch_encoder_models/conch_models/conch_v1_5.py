import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Conch_v1_5(nn.Module):
    def __init__(self):
        super().__init__()

        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, low_cpu_mem_usage=False)
        conch_v1_5, eval_transform = titan.return_conch()
        self.backbone = conch_v1_5
        self.img_transform = eval_transform

    def forward(self, x):  # x: tensor of shape (B, C, H, W)
        output = self.backbone(x)
        return output

    def get_tokens(self, x):  # x: tensor of shape (B, C, H, W)
        raise NotImplementedError("Conch model does not support image token extraction")



if __name__ == "__main__":
    model = Conch_v1_5().to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")
    output = model(x)
    print(output.shape)