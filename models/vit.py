import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from math import sqrt
from dinopl.modules import init

__all__ = [
    #"ViT",
    "vit_tiny",
    "vit_small",
    "vit_medium",
    "vit_base",
]


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if embed_dim % num_heads != 0:
            raise ValueError('Embedding dimension must be divisible by number of heads.')

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor):
        B, L, E = x.shape
        #qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, E // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, L, 3, E).permute(2, 0, 1, 3)       # 3, B, L, E
        qkv = qkv.reshape(3, B, L, self.num_heads, E // self.num_heads) # 3, B, L, H, E'
        q, k, v = qkv[0], qkv[1], qkv[2]                                # B, L, H, E'

        attn = (q @ k.transpose(-2, -1)) * self.scale                   # B, L, H, H
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, E)     # Interleave squence dim with embeddings?
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.act(x)         # original DINO doesn't have this activation
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, mlp_hidden_dim:int, qkv_bias=False, attn_drop:float=0., proj_drop:float=0., mlp_drop:float=0.):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        #self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, add_bias_kv=qkv_bias, dropout=attn_drop, batch_first=True)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, drop=mlp_drop)

    def forward(self, x:torch.Tensor):
        # layernorm, self-attention and residual connection
        residual = x 
        x = self.layer_norm1(x)
        x, attn = self.attention(x)
        #x, attn = self.attention(key=x, query=x, value=x)  # nn.MultiHeadAttention
        # x = self.drop_path(x)         # original DINO has DropPath module
        x = x + residual

        # layernorm, mlp and residual connection
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        # x = self.drop_path(x)         # original DINO has DropPath module
        x = x + residual
        return x

class PatchEmbedder(nn.Module):
    def __init__(self, img_chans:int, img_size:Tuple[int, int], patch_size:int, embed_dim:int) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        if len(img_size) != 2 or not isinstance(img_size[0], int) or not isinstance(img_size[1], int):
            raise ValueError('Image size must be an int or a tuple thereof.')

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(img_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        x = self.proj(x)            # B, E, H/patch_size, W/patch_size
        x = x.flatten(-2, -1)       # B, E, L
        x = x.transpose(-2, -1)     # B, L, E
        return x

class ViT(nn.Module):
    def __init__(self, 
            embedder:PatchEmbedder, 
            num_heads:int, 
            num_layers:int, 
            mlp_width_ratio:int = 4, 
            num_classes:int = None, 
            qkv_bias:bool = True,
            dropout:float = 0., 
        ):
        super().__init__()

        # prepare patch embedder 
        #self.embedder = PatchEmbedder(img_chans, img_size, patch_size, embed_dim)
        self.embedder = embedder
        self.num_patches = self.embedder.num_patches
        self.embed_dim = self.embedder.embed_dim

        # prepare CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, self.embed_dim))

        # prepare transformer blocks and final layer norm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim =  mlp_width_ratio * self.embed_dim
        layers = self.num_layers * [Block(self.embed_dim, self.num_heads, self.mlp_hidden_dim, qkv_bias=qkv_bias)]
        self.transformer = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # make fully-connected classifier     
        self.fc = nn.Identity()
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self, method='default', generator:torch.Generator=None):
        # initialize CLS token and positional embeddings
        if method == 'default':
            init.normal_(self.cls_token, generator=generator)
            init.normal_(self.pos_embed, generator=generator)
        if method == 'trunc_normal':
            init.trunc_normal_(self.cls_token, std=.02, generator=generator)
            init.trunc_normal_(self.pos_embed, std=.02, generator=generator)

        # initialize modules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'default':
                    #m.reset_parameters() is equal to:
                    bound = 1 / sqrt(m.in_features)
                    init.uniform_(m.weight, -bound, bound, generator=generator)
                    if m.bias is not None:
                        init.uniform_(m.bias, -bound, bound, generator=generator)

                if method == 'trunc_normal':
                    init.trunc_normal_(m.weight, std=.02, generator=generator)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

            if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        x = self.embedder(x) # (B, C, H, W) -> (B, L, E)

        # add CLS token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # transform sequence of tokens
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        # get CLS token and classify
        cls = x[:,0]
        x = self.fc(cls)

        return x
    

def vit_tiny(img_chans:int, img_size:int, patch_size:int=16, **kwargs) -> ViT:
    """Tiny ViT from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`.

    Args:
        img_chans: Number of channels in input image.
        img_size: Square size of channels in input image.
        patch_size: Patch size for patchwise embedding.
        **kwargs: parameters passed to the ``models.vit.ViT`` base class.
    """

    embedder = PatchEmbedder(img_chans, img_size, patch_size, embed_dim=192)
    return ViT(embedder, num_heads=3, num_layers=12, **kwargs)



def vit_small(img_chans:int, img_size:int, patch_size:int=16, **kwargs) -> ViT:
    """Small ViT from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`.

    Args:
        img_chans: Number of channels in input image.
        img_size: Square size of channels in input image.
        patch_size: Patch size for patchwise embedding.
        **kwargs: parameters passed to the ``models.vit.ViT`` base class.
    """

    embedder = PatchEmbedder(img_chans, img_size, patch_size, embed_dim=384)
    return ViT(embedder, num_heads=6, num_layers=12, **kwargs)


def vit_medium(img_chans:int, img_size:int, patch_size:int=16, **kwargs) -> ViT:
    """Medium ViT from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`.

    Args:
        img_chans: Number of channels in input image.
        img_size: Square size of channels in input image.
        patch_size: Patch size for patchwise embedding.
        **kwargs: parameters passed to the ``models.vit.ViT`` base class.
    """

    embedder = PatchEmbedder(img_chans, img_size, patch_size, embed_dim=512)
    return ViT(embedder, num_heads=8, num_layers=12, **kwargs)


def vit_base(img_chans:int, img_size:int, patch_size:int=16, **kwargs) -> ViT:
    """Base ViT from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`.

    Args:
        img_chans: Number of channels in input image.
        img_size: Square size of channels in input image.
        patch_size: Patch size for patchwise embedding.
        **kwargs: parameters passed to the ``models.vit.ViT`` base class.
    """

    embedder = PatchEmbedder(img_chans, img_size, patch_size, embed_dim=768)
    return ViT(embedder, num_heads=12, num_layers=12, **kwargs)
