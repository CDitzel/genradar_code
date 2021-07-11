import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torchvision.models import resnet50
from torchsummary import summary

from ..utils import auxiliary
from .network_utils import *

logger = logging.getLogger(__name__)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, lEmb, seq_length):
        super().__init__()
        self.pe = nn.Embedding(max_position_embeddings, lEmb)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class FixedPositionalEncoding(nn.Module):
    def __init__(self, lEmb, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, lEmb)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, lEmb, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / lEmb)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = (
            torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            + offset
        )
        sinusoid_inp = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class LinearEmbedding(nn.Module):
    def __init__(self, lEmb, patch_size, in_channels):
        super().__init__()

        self.lEmb = lEmb
        self.patch_size = patch_size

        unrolled_patch_len = in_channels * patch_size ** 2
        self.proj = nn.Linear(unrolled_patch_len, lEmb)

    def forward(self, x):
        ph = pw = self.patch_size
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        return self.proj(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, lEmb, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.proj = ConvLayer(in_channels, lEmb, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c")


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def Mlp(lEmb, rMlp=4, mlp_drop=0.0):
    return nn.Sequential(
        nn.Linear(lEmb, lEmb * rMlp),
        nn.GELU(),
        # GEGLU(),
        # nn.Dropout(mlp_drop),
        nn.Linear(lEmb * rMlp, lEmb),
        nn.Dropout(mlp_drop),
    )


class Attention(nn.Module):
    def __init__(self, lEmb, nHead=8, dHead=64, att_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.nHead = nHead
        self.scale = dHead ** -0.5
        self.to_qkv = nn.Linear(lEmb, (nHead * dHead * 3))

        self.att_drop = nn.Dropout(att_drop)
        self.unify_heads = nn.Linear((nHead * dHead), lEmb)
        self.proj_drop = nn.Dropout(proj_drop)

        self.att_maps = None

        # self.register_buffer("mask", torch.tril(torch.ones(512, 512)) .view(1, 1, 512, 512))

    def forward(self, x, ctx=None, mask=True):
        b, n, _, h = *x.shape, self.nHead
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (h d qkv) -> qkv b h n d", h=h, qkv=3)
        # test each token against every other for every head separately
        att = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        mask = torch.tril(torch.ones(n, n)).to(att)
        att = att.masked_fill(mask == 0, float("-inf"))
        # if n > 256 and n < 260:
            # print(att.shape)
            # print(att[0, 0, 255:266])
            # exit()
        att = F.softmax(att, dim=-1)
        self.att_maps = att.detach().masked_fill(mask == 0, 0.0)
        # del mask
        att = self.att_drop(att)
        out = torch.einsum("bhij,bhjd->bhid", att, v)

        # condense separate heads into emb dimension for succeeding mlp
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.unify_heads(out)
        out = self.proj_drop(out)
        return out


class Layer(nn.Module):
    def __init__(self, lEmb, nHead, dHead, rMlp, attn_drop, proj_drop, mlp_drop, drop_path):
        super().__init__()
        self.ln1 = nn.LayerNorm(lEmb)
        self.attn = Attention(lEmb, nHead, dHead, attn_drop, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(lEmb)
        self.mlp = Mlp(lEmb, rMlp, mlp_drop)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class Transforming(nn.Module):
    def __init__(
        self,
        lEmb,
        nHead,
        dHead,
        recycle_embs,
        nToks,
        nPred,
        nLayer,
        rMlp=4,
        maxToks=512,
        emb_drop=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.recycle_embs = recycle_embs

        if not self.recycle_embs:
            self.init_emb = nn.Embedding(nToks, lEmb)
            print('vocabulary', self.init_emb.weight.shape)
        print('predicting num tokens', nPred)
        # exit()
        self.pos_emb = nn.Parameter(torch.zeros(1, maxToks, lEmb)) # leading 1: broadcastable
        # self.pos_emb = FixedPositionalEncoding(lEmb, maxToks)
        # self.pos_emb = FixedPositionalEmbedding(lEmb)
        self.drop_emb = nn.Dropout(emb_drop)
        self.maxToks = maxToks

        # stochastic depth decay rule
        dpr = [dr.item() for dr in torch.linspace(0, drop_path_rate, nLayer)]
        print('Drop path rate', dpr)
        self.drop_path = [DropPath(dr) if dr > 0.0 else nn.Identity() for dr in dpr]

        self.layers = nn.Sequential(
            *[
                Layer(lEmb, nHead, dHead, rMlp, attn_drop, proj_drop, mlp_drop, dpr[l])
                for l in range(nLayer)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lEmb),
            nn.Linear(in_features=lEmb, out_features=nPred, bias=False),
        )

        self.apply(self._init_weights)
        logger.info(f"num of trafo params: {sum(p.numel() for p in self.parameters()):e}")

    def __call__(self, x):
        if not self.recycle_embs:
            x = self.init_emb(x)  # each index maps to a (learnable) vector

        t = x.shape[1]
        assert t <= self.maxToks, "Cannot forward, model block size is exhausted."
        pos_emb = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # print('vit x', x.shape)
        # print('vit pos_emb', pos_emb.shape)
        x = self.drop_emb(x + pos_emb)
        # print(x.shape)
        # print(self.pos_emb(x).shape)

        # inp = x + self.pos_emb(x)

        # inp = self.pos_emb(x)

        x = self.layers(x)
        return self.mlp_head(x)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Transformer(nn.ModuleDict):
    def __init__(
        self,
        _type: str,
        lEmb: int,
        # nHead,
        # dHead,
        # rMlp,
        # dropout,
        # norm,
        # activation,
        nLayers: int,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = False,
        **kw,
    ):
        # stochastic depth decay rule
        dpr = [dr.item() for dr in torch.linspace(0, drop_path_rate, nLayers)]
        self.drop_path = [DropPath(dr) if dr > 0.0 else nn.Identity() for dr in dpr]

        super().__init__(
            {
                f"Layer {l}": nn.ModuleDict(
                    {
                        "self_att": Residual(PreNorm(lEmb, Attention(lEmb, **kw))),
                        "cros_att": Residual(
                            PreNorm(lEmb, Attention(lEmb, **kw))
                            if _type == "dec"
                            else nn.Identity()
                        ),
                        "feed_fwd": Residual(PreNorm(lEmb, Mlp(lEmb, **kw))),
                    }
                )
                for l in range(nLayers)
            }
        )

    def forward(self, x, ctx=None, mask=None):
        for layer, drop_path in zip(self.values(), self.drop_path):
            x = drop_path(layer["self_att"](x))
            x = drop_path(layer["cros_att"](x, ctx) if ctx is not None else x)
            x = drop_path(layer["feed_fwd"](x))
        return x

    @classmethod
    def encoder(cls, **kw):
        return cls("enc", **kw)

    @classmethod
    def decoder(cls, **kw):
        return cls("dec", **kw)


# @auxiliary.verbose_output
class ViT(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        patch_size,
        in_channels=3,
        lEmb=512,
        init_embedding=PatchEmbedding,
        transformer=Transformer.encoder,
        nLayers=6,
        qkv_bias=False,
        nHead=8,
        dHead=64,
        rMlp=4,
        activation="gelu",
        pos_enc_type="fixed",
        dropout=0.1,
        emb_dropout=0.1,
        drop_path_rate=0.0,
        nClasses=2,
        pool="cls",  # mean
    ):
        super().__init__()

        num_patches = (input_size // patch_size) ** 2
        self.init_embedding = init_embedding(lEmb, patch_size, in_channels)
        self.transformer = transformer(
            lEmb=lEmb,
            nHead=nHead,
            dHead=dHead,
            nLayers=nLayers,
            rMlp=rMlp,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            activation=activation,
            qkv_bias=qkv_bias,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, lEmb))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, lEmb))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, lEmb))
        # self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.to_latent = nn.Identity()
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lEmb),
            nn.Linear(in_features=lEmb, out_features=lEmb * rMlp),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(in_features=lEmb * rMlp, out_features=nClasses),
        )
        # self.mlp_head = nn.Sequential(
        # nn.LayerNorm(dim),
        # nn.Linear(dim, num_classes)
        # )

    def forward(self, x):
        x = self.init_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, : (n + 1)]
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


# @auxiliary.verbose_output
class DualTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_size=256,
        patch_size=16,
        lEmb=512,
        nLayers=6,
        qkv_bias=False,
        nHead=8,
        dHead=64,
        rMlp=4,
        activation="gelu",
        pos_enc_type="fixed",
        dropout=0.1,
        emb_dropout=0.1,
        drop_path_rate=0.0,
        nClasses=2,
        pool="cls",  # mean
        init_tiny_ff=True,
    ):
        super().__init__()

        num_patches = (input_size // patch_size) ** 2
        self.rad_init_embedding = PatchEmbedding(1, lEmb, patch_size)
        self.cam_init_embedding = PatchEmbedding(3, lEmb, patch_size)

        self.rad_trafo = Transformer.encoder(
            lEmb=lEmb,
            nHead=nHead,
            dHead=dHead,
            nLayers=nLayers,
            rMlp=rMlp,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            activation=activation,
            qkv_bias=qkv_bias,
        )
        self.cam_trafo = Transformer.encoder(
            lEmb=lEmb,
            nHead=nHead,
            dHead=dHead,
            nLayers=nLayers,
            rMlp=rMlp,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            activation=activation,
            qkv_bias=qkv_bias,
        )
        self.rad_cls_token = nn.Parameter(torch.randn(1, 1, lEmb))
        self.cam_cls_token = nn.Parameter(torch.randn(1, 1, lEmb))

        self.rad_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, lEmb))
        self.cam_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, lEmb))

        self.rad_emb_dropout = nn.Dropout(emb_dropout)
        self.cam_emb_dropout = nn.Dropout(emb_dropout)

        self.pool = pool

        self.ff_dist = nn.Linear(1, 2)

        if init_tiny_ff:
            # need to initialize the tiny fc
            self.ff_dist.weight.data[0] = -0.7090
            self.ff_dist.weight.data[1] = 0.7090
            self.ff_dist.bias.data[0] = 1.2186
            self.ff_dist.bias.data[1] = -1.2186

        self.mlp_head = nn.Sequential(nn.LayerNorm(lEmb), nn.Linear(lEmb, nClasses))

    def forward(self, rad, cam, **kw):
        rad = self.rad_init_embedding(rad)
        cam = self.cam_init_embedding(cam)
        b, n, _ = rad.shape
        rad_cls_tokens = repeat(self.rad_cls_token, "() n d -> b n d", b=b)
        cam_cls_tokens = repeat(self.cam_cls_token, "() n d -> b n d", b=b)

        rad = torch.cat((rad_cls_tokens, rad), dim=1)
        cam = torch.cat((cam_cls_tokens, cam), dim=1)

        rad += self.rad_pos_embed[:, : (n + 1)]
        cam += self.cam_pos_embed[:, : (n + 1)]

        rad = self.rad_emb_dropout(rad)
        cam = self.cam_emb_dropout(cam)

        rad = self.rad_trafo(rad)
        cam = self.cam_trafo(cam)

        if self.pool == "mean":
            rad = reduce(rad, "b n d -> b d", "mean")
            cam = reduce(cam, "b n d -> b d", "mean")

        else:  # only rely on cls token, cf. ViT
            rad = rad[:, 0]
            cam = cam[:, 0]

        # rad = F.normalize(rad, p=2, dim=1)  # L2 normalization
        # cam = F.normalize(cam, p=2, dim=1)  # L2 normalization

        # distance = F.mse_loss(rad, cam, reduction='none').mean(1).unsqueeze(1)
        # out = self.ff_dist(distance)

        distance = F.mse_loss(rad, cam, reduction="none")
        out = self.mlp_head(distance)

        return out
        # x = self.to_latent(x)


if __name__ == "__main__":
    dummy = torch.ones(1, 3, 256, 256)
    v = ViT(input_size=dummy.shape[-1], patch_size=32, nLayers=6).cuda()
    # out = v(dummy)
    # auxiliary.print_num_params(v)
    # summary(v.cuda(), (1, 256, 256))
    # exit()
    start = time.time()

    for iter in range(100):
        print(iter)
        sample = torch.randn(1, 3, 256, 256).cuda()
        v(sample)

    stop = time.time()
    print("Duration", stop - start)

    # func = register_forward_hook(
    # lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
    # )

    # vapply(func)
