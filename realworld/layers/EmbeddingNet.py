import torch.nn as nn
from layers.MLP import MLP
from einops import rearrange


class Embedding_Net(nn.Module):
    def __init__(self, patch_size, input_len, out_len, emb_dim) -> None:
        super().__init__()
        self.patch_size = patch_size if patch_size <= input_len else input_len
        self.stride = self.patch_size // 2
        self.out_len = out_len

        self.num_patches = int((input_len - self.patch_size) / self.stride + 1)

        self.net1 = MLP(1, in_dim=self.patch_size, out_dim=emb_dim)
        self.net2 = MLP(1, emb_dim * self.num_patches, out_dim=self.out_len)

    def forward(self, x):
        B, L, M = x.shape
        if self.num_patches != 1:
            x = rearrange(x, 'b l m -> b m l')
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, 'b m n p -> (b m) n p')
        else:
            x = rearrange(x, 'b l m -> (b m) 1 l')
        x = self.net1(x)
        outputs = self.net2(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b  l m', b=B)
        return outputs
