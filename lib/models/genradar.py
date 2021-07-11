from math import log2, sqrt
from enum import Enum
import os
import functools
from abc import ABCMeta, abstractmethod
import time

from torchsummary import summary
import torch
from torch import nn, einsum, optim
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import distributions
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical, Categorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

from .network_utils import *
# from .minGPT import GPT
from ..utils import auxiliary
from .resnet import *
from .vit import FixedPositionalEncoding, FixedPositionalEmbedding, Transforming
import lib
from lib import load_config, instantiate_or_load, eval_decorator, top_k_ratio, set_requires_grad, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
torch.set_printoptions(profile="full", linewidth=500, precision=10),

def sample_gumbel(n,k):
    unif = torch.distributions.Uniform(0,1).sample((n,k))
    g = -torch.log(-torch.log(unif))
    return g


def sample_gumbel_softmax(pi, n, temperature):
    k = len(pi)
    g = sample_gumbel(n, k)
    h = (g + torch.log(pi))/temperature
    h_max = h.max(dim=1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    #     print(pi, torch.log(pi), intmdt)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
    return g

def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

def gumbel_softmax_sample(shape, temperature, eps=1e-20):
    # unif = torch.distributions.Uniform(0,1).sample((n,k))

    u = torch.rand(*shape, device=logits.get_device())
    g = -torch.log(-torch.log(u + eps) + eps)
    x = logits + g
    return F.softmax(x / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y

    n_classes = y.shape[-1]
    z = torch.argmax(y, dim=-1)
    z = F.one_hot(z, n_classes)
    z = (z - y).detach() + y
    return z

@instantiate_or_load
class SoftDiscretizer(nn.Module):
    """
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(
        self,
        dLatent,
        nTokens,
        dTokens,
        temperature,
        kl_weight,
        act=None,
        norm=None,
        soft_idx=False,
        straight_through=False,
        categorical=False,
        init_emb=False
    ):
        super().__init__()
        self.nTokens = nTokens
        self.dTokens = dTokens
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.straight_through = straight_through
        self.categorical = categorical
        self.soft_idx = soft_idx

        # self.pre_proj = ConvLayer(dLatent, dTokens, 1)
        self.pre_proj = ConvLayer(dLatent, nTokens, 1)
        self.embedding = nn.Embedding(nTokens, dTokens)
        self.post_proj = ConvLayer(dTokens, dLatent, 1)

        if init_emb:
            self.embedding.weight.data.uniform_(-1.0 / nTokens, 1.0 / nTokens)

    def embed(self, constituents):
        constituents = self.embedding(constituents)
        b, n, d = constituents.shape
        h = w = int(sqrt(n))
        constituents = rearrange(constituents, "b (h w) d -> b d h w", h=h, w=w)
        return self.post_proj(constituents)

    def forward(self, z, only_quantize=False, soft_idx=False):
        z = self.pre_proj(z)
        B, C, H, W = z.size()
        ratio_weight = H * W

        if self.categorical:

            """ bshall hack
            z = z.permute(0, 2, 3, 1)
            z_flat = z.reshape(-1, self.dTokens)

            distances = (
                z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
            )
            distances = distances.view(B, H, W, -1)
            dist = RelaxedOneHotCategorical(self.temperature(self.training), logits=-distances)

            if self.training:
                samples = dist.rsample()
            else:
                samples = torch.argmax(dist.probs, dim=-1)
                samples = F.one_hot(samples, self.nTokens).float()
                # samples = samples
            froz_dist = RelaxedOneHotCategorical(self.temperature.v1, logits=-distances)
            froz_samples = froz_dist.rsample()
            froz_tokens = froz_samples.argmax(dim= -1)
            cold_dist = Categorical(logits= -distances)
            """
            z = rearrange(z, "b c h w -> b h w c") # prob dists parameterizes over last dim
            if self.training:
                # dist = ExpRelaxedCategorical(
                    # temperature=self.temperature(self.training), logits=z)
                dist = RelaxedOneHotCategorical(
                    temperature=self.temperature(self.training), logits=z)
                samples = dist.rsample()
            else:
                dist = OneHotCategorical(logits=z)
                # if soft_idx: # can yield vastly different tokens than argmax below
                samples = dist.sample() # one-hot tensors
                # else:
                # check = dist.probs.argmax(dim=-1)
                # samples = F.one_hot(check, self.nTokens).float()

            froz_dist = RelaxedOneHotCategorical(self.temperature.v1, logits=z)
            froz_samples = froz_dist.rsample()
            froz_tokens = froz_samples.argmax(dim= -1)
            #"""

            # cold_dist = RelaxedOneHotCategorical(temperature=self.temperature.v1, logits=z)
            # cold_dist = Categorical(logits=dist.probs)
            cold_dist = Categorical(logits=z)
            cold_tokens = cold_dist.sample()

            soft_tokens = samples.argmax(dim= -1)
            hard_tokens = dist.probs.argmax(dim=-1)
            # cold_tokens = cold_samples.argmax(dim= -1)
            if only_quantize:
                # return tokens.flatten(start_dim=1)
                return (tokens.flatten(start_dim=1) for tokens in (soft_tokens, hard_tokens, cold_tokens, froz_tokens))


            samples = rearrange(samples, "b h w c -> b c h w")

            kl = dist.probs * (dist.logits + math.log(self.nTokens))
            kl[(dist.probs == 0).expand_as(kl)] = 0
            kl = kl.sum(dim=(1, 2, 3)).mean()
            kl_loss = self.kl_weight(self.training) * kl * ratio_weight

        else:
            # z \ = z.sum(dim=1)
            # z = z / torch.sum(z, dim=1, keepdim=True)
            # hard= True: take argmax, but differentiate w.r.t. soft sample y
            # hard = self.straight_through if self.training else True
            hard = self.straight_through
            # if self.training:
            samples = F.gumbel_softmax(z, self.temperature(self.training), hard=hard, dim= 1)
            # print(self.temperature.v1)
            # print(self.temperature(self.training))
            # exit()
            # samples taken with the final temperature
            froz_samples = F.gumbel_softmax(z, self.temperature.v1, hard=hard, dim= 1)
            froz_tokens = froz_samples.argmax(dim=1)
            # print(samples.shape)

            # one_hot_categorical = OneHotCategorical(logits=rearrange(z, "b c h w -> b h w c"))
            one_hot_categorical = Categorical(logits=rearrange(z, "b c h w -> b h w c"))
            # print(one_hot_categorical.logits[0, 0, 0])
            cold_tokens = one_hot_categorical.sample()
            # print(cold_tokens[0, 0, 0])

            # cold_tokens = cold_tokens.argmax(dim= -1)

            # print(cold_tokens.shape)
            # print(cold_tokens[0, 0, 0])
            # print(cold_tokens.shape)
            # print(cold_tokens[0, 0, 0])
            # exit()

            # samples = sample_gumbel_softmax(z, self.temperature(self.training))
            # cold_samples = sample_gumbel_softmax(z, self.temperature.v1)

            soft_tokens = samples.argmax(dim= 1)# if soft_idx else z.argmax(dim= 1)
            hard_tokens = z.argmax(dim= 1)


            # print(torch.amin(torch.amax(z.softmax(1), dim=1).flatten()))
            # print(torch.amax(torch.amax(z.softmax(1), dim=1).flatten()))
            # exit()
            # hard_tokens = torch.rot90(hard_tokens, 1, [-2, -1])

            if only_quantize:
                # return tokens.flatten(start_dim=1)
                return (tokens.flatten(start_dim=1) for tokens in (soft_tokens, hard_tokens, cold_tokens, froz_tokens))

            # The KL divergence measures how the encoder distribution, $q(z)$, differs
            # from the uniform prior distribution $p(z)$. This divergence serves as a
            # regularization term, meant to force the distribution as close to a
            # uniform distribution, without losing too much reconstruction ability. In
            # the perfection situation, this will not reach zero, as that would
            # indicate a random distribution, and an decoder that has learned
            # nothing. DONT EXPECT 0 but STABLE VALUE

            logits = F.log_softmax(z, dim=1)
            probs = torch.exp(logits)  # numerically more stable than immediate softmax

            neg_entropy = torch.mean(torch.sum(probs * logits, dim=1))

            # kl = torch.mean(torch.sum(probs * (logits + math.log(self.nTokens)), dim=1))
            # neg_entropy = torch.mean(torch.sum(probs * logits, dim=(1, 2, 3)))
            kl = torch.mean(torch.sum(probs * (logits + math.log(self.nTokens)), dim=(1, 2, 3)))
            kl_loss = self.kl_weight(self.training) * kl * ratio_weight

            # neg_entropy = torch.sum(probs * (logits + math.log(self.nTokens)), dim=(1, 2, 3))
            # kl_loss = self.kl_weight(self.training) * torch.mean(neg_entropy) * ratio_weight


        z_q = einsum("b n h w, n d -> b d h w", samples, self.embedding.weight)
        z_q = self.post_proj(z_q)
        # return z_q, kl_loss, (tokens.flatten(start_dim=0) for tokens in
                              # (soft_tokens, hard_tokens, cold_tokens)), -neg_entropy
        return z_q, kl_loss, (tokens.flatten(start_dim=0) for tokens in
                              (soft_tokens, hard_tokens, cold_tokens))


@instantiate_or_load
class DiscreteVAE(nn.Module):
    def __init__(self, io_chn, nLayers, dLatent, nResBlocks, discretizer, inp_size=256, **kw):
        super().__init__()
        # self.loss = loss
        self.inp_size = inp_size
        self.discretizer = discretizer
        self.encoder, self.decoder = simple_enc_dec(io_chn, dLatent, nLayers, nResBlocks, **kw)
        self.nLayers = len(self.encoder) - nResBlocks

    def __call__(self, x, soft_idx):
        # x = self.loss.inmap(x)
        logits = self.encoder(x)
        # z_q, kl_loss, threefold_tokens, neg_ent = self.discretizer(logits, soft_idx=soft_idx)
        z_q, kl_loss, threefold_tokens = self.discretizer(logits, soft_idx=soft_idx)
        perplx, cluster_use = [], []
        for i, tok in enumerate(threefold_tokens):
            stats = self.perplexity(tok)
            perplx.append(stats[0])
            cluster_use.append(stats[1])

        rec = self.decoder(z_q)
        # return kl_loss, rec, perplx, cluster_use, neg_ent
        return kl_loss, rec, perplx, cluster_use

    @torch.no_grad()
    @eval_decorator
    def discretize(self, x, soft_idx):
        # x = self.loss.inmap(x)
        logits = self.encoder(x)
        return self.discretizer(logits, only_quantize=True, soft_idx=soft_idx)

    @torch.no_grad()
    @eval_decorator
    def constitute(self, constituents):
        constituents = self.discretizer.embed(constituents)
        return self.decoder(constituents)

    @torch.no_grad()
    @eval_decorator
    def perplexity(self, tokens):
        # when perplexity == num_embeddings then all clusters are used exactly equally
        token_dist = torch.bincount(tokens) / len(tokens)
        perplexity = torch.exp(-torch.sum(token_dist * torch.log(token_dist + 1e-10)))
        cluster_use = torch.sum(token_dist > 0)

        return perplexity, cluster_use

    def last(self):
        return self.decoder[-1][0].weight
        # return self.decoder[-1]["c"].weight

@instantiate_or_load
class GenRadar(nn.Module):
    def __init__(
        self,
        cam_vae,
        rad_vae,
        only_cam_labels,
        sep_embs,
        recycle_embs,
        scale_loss,
        pkeep=1.0,
        lEmb=512,
        nLayer=6,
        nHead=8,
        dHead=64,
        rMlp=4,
        emb_drop=0.,
        attn_drop=0.,
        proj_drop=0.,
        mlp_drop=0.,
        drop_path_rate=0.,
        **kwargs
    ):
        super().__init__()

        self.rad_vae = set_requires_grad(rad_vae, requires_grad=False).eval()
        self.cam_vae = set_requires_grad(cam_vae, requires_grad=False).eval()

        self.rad_nToks = rad_vae.discretizer.nTokens
        self.cam_nToks = cam_vae.discretizer.nTokens
        print('rad token', self.rad_nToks)
        print('cam token', self.cam_nToks)

        self.total_nToks = self.rad_nToks + self.cam_nToks

        self.smooth_ce = LabelSmoothingCrossEntropy()
        self.soft_ce = SoftTargetCrossEntropy()

        self.pkeep = pkeep
        self.only_cam_labels = only_cam_labels
        self.sep_embs = sep_embs
        self.recycle_embs = recycle_embs
        self.scale_loss = scale_loss

        self.rad_seq_len = (self.rad_vae.inp_size // (2 ** self.rad_vae.nLayers)) ** 2
        self.cam_seq_len = (self.cam_vae.inp_size // (2 ** self.cam_vae.nLayers)) ** 2
        self.total_seq_len = self.rad_seq_len + self.cam_seq_len

        if self.recycle_embs:
            self.rad_emb = self.rad_vae.discretizer.embedding
            self.cam_emb = self.cam_vae.discretizer.embedding

        self.transformer = Transforming(
            lEmb=lEmb,
            nHead=nHead,
            dHead=dHead,
            recycle_embs=recycle_embs,
            nToks=self.total_nToks if sep_embs else max(self.rad_nToks, self.cam_nToks),
            nPred=self.total_nToks if not only_cam_labels else self.cam_nToks,
            maxToks=self.total_seq_len,
            nLayer=nLayer,
            rMlp=rMlp,
            emb_drop=emb_drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path_rate=drop_path_rate)

        if not self.only_cam_labels:
            seq_range = torch.arange(self.total_seq_len)
            logits_range = torch.arange(self.total_nToks)
            seq_range = rearrange(seq_range, "n -> () n ()")
            logits_range = rearrange(logits_range, "d -> () () d")

            logits_mask = (
                (seq_range >= self.rad_seq_len) & (logits_range < self.rad_nToks)
            ) | ((seq_range < self.rad_seq_len) & (logits_range >= self.rad_nToks))

            self.register_buffer("logits_mask", logits_mask)

    def forward(self, rad, cam, rad_s, cam_s, infer=False, noise_cond=False, last_iter=False):

        if noise_cond == 'image':
            rad = (rad.max() - rad.min()) * torch.rand_like(rad) + rad.min()

        rad_soft_t, rad_hard_t, rad_cold_t, rad_froz_t  = self.rad_vae.discretize(rad, rad_s)
        cam_soft_t, cam_hard_t, cam_cold_t, cam_froz_t = self.cam_vae.discretize(cam, cam_s)

        if cam_s == 'soft':
            cam_targets = cam_soft_t.detach().clone()
            cam_toks = cam_soft_t
        elif cam_s == 'hard':
            cam_targets = cam_hard_t.detach().clone()
            cam_toks = cam_hard_t
        elif cam_s == 'cold':
            cam_targets = cam_cold_t.detach().clone()
            cam_toks = cam_cold_t
        elif cam_s == 'froz':
            cam_targets = cam_froz_t.detach().clone()
            cam_toks = cam_froz_t

        if rad_s == 'soft':
            rad_targets = rad_soft_t.detach().clone()
            rad_toks = rad_soft_t
        elif rad_s == 'hard':
            rad_targets = rad_hard_t.detach().clone()
            rad_toks = rad_hard_t
        elif rad_s == 'cold':
            rad_targets = rad_cold_t.detach().clone()
            rad_toks = rad_cold_t
        elif rad_s == 'froz':
            rad_targets = rad_froz_t.detach().clone()
            rad_toks = rad_froz_t

        # print(rad_toks.view(-1, 16, 16))
        # time.sleep(0.01)

        if noise_cond == 'toks':
            rad_toks = torch.randint_like(rad_toks, self.rad_nToks)

        cam_faith = torch.where(cam_hard_t == cam_toks, 1, 0).float().mean().item()
        rad_faith = torch.where(rad_hard_t == rad_toks, 1, 0).float().mean().item()

        if infer:
            return rad_toks, cam_toks, rad_faith, cam_faith,

        if last_iter:
            rad_dec = self.rad_vae.constitute(rad_toks)
            cam_dec = self.cam_vae.constitute(cam_toks)

        # disturbing training by replacing cam toks with random indices
        # with a probability of (1-pkeep) for every token individually
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(
                self.pkeep*torch.ones(cam_toks.shape, device=cam_toks.device)).int()
            rnd_toks = torch.randint_like(cam_toks, self.cam_nToks)
            cam_toks = mask*cam_toks+(1-mask)*rnd_toks

        if self.sep_embs:
            cam_toks += self.rad_nToks
        if self.recycle_embs:
            rad_toks = self.rad_emb(rad_toks)
            cam_toks = self.cam_emb(cam_toks)

        total_toks = torch.cat([rad_toks, cam_toks], dim=1)

        out = self.transformer(total_toks[:, :-1]) # [r1 r2 r3 r4 | c1 c2 c3]
        # out = self.transformer(total_toks) # [r1 r2 r3 r4 | c1 c2 c3]

        if self.only_cam_labels:
            targets = cam_targets                # [c1 c2 c3 c4]
            out = out[:, self.rad_seq_len - 1 :] # [r4|c1 c2 c3]
            # out = out[:, self.rad_seq_len :] # [r4|c1 c2 c3]
        else:
            max_neg_value = -torch.finfo(out.dtype).max # softmax in CE-loss zeroes these vals
            out.masked_fill_(self.logits_mask[:, 1:], max_neg_value)
            offset_cam_targets = cam_targets + self.rad_nToks
            targets = torch.cat([rad_targets[:, 1:], offset_cam_targets], dim=1) # [r2 r3 r4 | c1 c2 c3 c4]

        out = rearrange(out, 'b n c -> b c n')  # 2D-CrossEntropy
        if not self.only_cam_labels:
            b, c, n = out.shape
            max_mass_rad = out.argmax(dim= -2)[:, :255]
            max_mass_cam = out.argmax(dim= -2)[:, 255:]
            first_rad_tok = repeat(torch.zeros(1, 1), "() n -> b n", b=b).to(max_mass_rad)
            max_mass_rad = torch.cat((first_rad_tok, max_mass_rad), dim=1)
        else:
            max_mass_cam = out.argmax(dim= -2)
            max_mass_rad = torch.zeros_like(max_mass_cam)

        if last_iter:
            max_mass_pred_cam = self.cam_vae.constitute(max_mass_cam -
                                                        self.rad_nToks if not
                                                        self.only_cam_labels else
                                                        max_mass_cam)

            if not self.only_cam_labels:
                max_mass_pred_rad = self.rad_vae.constitute(max_mass_rad)
            else:
                max_mass_pred_rad = torch.zeros_like(max_mass_pred_cam).to(max_mass_pred_cam)

        cam_modes = max_mass_cam - self.rad_nToks if not self.only_cam_labels else max_mass_cam

        sim = nn.CosineSimilarity(dim=1, eps=1e-6)(max_mass_cam -
                                                   self.rad_nToks if not
                                                   self.only_cam_labels else
                                                   max_mass_cam.float(), cam_toks.float()).mean()

        if self.scale_loss != 1 and not self.only_cam_labels:
            rad_loss = F.cross_entropy(out[:, :, :self.rad_seq_len], targets[:, :self.rad_seq_len])
            cam_loss = F.cross_entropy(out[:, :, self.rad_seq_len:], targets[:, self.rad_seq_len:])
            loss = (rad_loss + self.scale_loss * cam_loss) / (self.scale_loss + 1)
        else:
            loss = F.cross_entropy(out, targets)
            # loss = self.smooth_ce(out.transpose(-2, -1), targets)
            # loss = self.soft_ce(out.transpose(-2, -1), targets)
        if last_iter:
            return loss, rad_faith, cam_faith, max_mass_pred_rad, max_mass_pred_cam, rad_dec, cam_dec, sim, cam_modes
        return loss, rad_faith, cam_faith, None, None, None, None, sim, cam_modes

    @torch.no_grad()
    @eval_decorator
    def synthesize(self, rad, cam, temperature=1.0, thld=1.0, num_samples=1,
                   rad_s='hard', cam_s='hard', noise_cond=False, cam_stepwise=False):
        assert (
            not self.transformer.training
        ), "disable transformer dropout for infereence etc."

        # print(f'Sampling camera toks from a multinomial with ' +
              # f'{max(int((1 - thld) * self.cam_nToks), 1)} realizations')

        rad_toks, cam_toks, rad_faith, cam_faith = self(rad, cam, rad_s, cam_s, infer=True, noise_cond=noise_cond)

        if self.recycle_embs:
            total_toks = self.rad_emb(rad_toks)
        else:
            total_toks = rad_toks

        b = rad.shape[0]
        # total_toks = cam_toks[:, :1]
        # cam_toks_pred = total_toks

        # cam_toks_pred = torch.IntTensor().to(rad_toks)
        cam_toks_pred = torch.zeros_like(cam_toks, dtype=torch.long)
        # print(cam_toks_pred.shape)
        # if cam_stepwise:
        cam_prd_stepwise = torch.zeros([b, self.cam_seq_len, *cam.shape[-3:]])
        # cam_prd_stepwise = torch.randint(low=0, high=self.cam_nToks, size=(b, self.cam_seq_len, *cam.shape[-3:]))
        # cam_prd_stepwise = torch.empty((b, self.cam_seq_len, *cam.shape[-3:]))
        # print(cam_prd_stepwise.shape)
        for tok in range(self.cam_seq_len):
            print(f"Predicting cam token: {tok+1}/{self.cam_seq_len}", end='\r')
            logits = self.transformer(total_toks)
            cam_logits = logits[:, -1]

            logits_rad_argmax = logits[:, :self.rad_seq_len - 1, :self.rad_nToks].argmax(dim= -1)
            logits_rad_first_tok = repeat(torch.zeros(1, 1), "() n -> b n", b=b).to(logits_rad_argmax)
            logits_rad_argmax = torch.cat((logits_rad_first_tok,logits_rad_argmax), dim=1)

            if not self.only_cam_labels:
                logits_mask = self.logits_mask[:, -1]
                max_neg_value = -torch.finfo(cam_logits.dtype).max
                cam_logits.masked_fill_(logits_mask, max_neg_value)
            preselected = top_k_ratio(cam_logits, thld)
            preselected_probs = F.softmax(preselected / temperature, dim=-1)
            sample = torch.multinomial(preselected_probs, num_samples=num_samples, replacement=False)

            if not self.only_cam_labels:
                sample -= self.rad_nToks
                # sample = torch.randint(self.cam_nToks, (4, 1)).to(sample)

            cam_toks_pred[:, tok] = sample.squeeze()

            if self.sep_embs:
                sample += self.rad_nToks
                # sample = torch.randint(self.rad_nToks, self.rad_nToks + self.cam_nToks, (4, 1)).to(sample)
            if self.recycle_embs:
                sample = self.cam_emb(sample)

            total_toks = torch.cat([total_toks, sample], dim=1)
            if cam_stepwise:
                cam_prd_stepwise[:, tok] = self.cam_vae.constitute(cam_toks_pred)

        # if not self.only_cam_labels:
            # cam_toks_pred -= self.rad_nToks
        # rad_toks = torch.arange(256).to(rad_toks).unsqueeze(0)
        # cam_toks = torch.arange(256).to(cam_toks).unsqueeze(0)

        # rad_toks = torch.ones(256, 256).to(rad_toks)
        # cam_toks = torch.ones(256, 256).to(cam_toks)
        # for t in range(256):
            # rad_toks[t] = t * torch.ones(256).to(rad_toks).unsqueeze(0)
            # cam_toks[t] = t * torch.ones(256).to(cam_toks).unsqueeze(0)

        cam_prd = self.cam_vae.constitute(cam_toks_pred)
        rad_prd = self.rad_vae.constitute(logits_rad_argmax)
        correct_constitutes = (cam_toks == cam_toks_pred).float().mean()
        sim = nn.CosineSimilarity(dim=1, eps=1e-6)(cam_toks_pred.float(), cam_toks.float()).mean()
        cam_dec = self.cam_vae.constitute(cam_toks)
        rad_dec = self.rad_vae.constitute(rad_toks)
        return torch.rot90(rad_prd, 1, [-2, -1]), torch.rot90(rad_dec, 1, [-2, -1]), cam_prd, cam_dec, correct_constitutes, sim, rad_faith, cam_faith, cam_prd_stepwise


"""
@instantiate_or_load
class LucidRadar(nn.Module):
    def __init__(
            self,
            rad_vae,
            cam_vae,
            only_cam_labels,
            separate_embs,
            recycle_emb=False,
            transformer="Vanilla",
            pkeep=1.0,
            lEmb=512,
            nLayer=6,
            nHead=8,
            dHead=64,
            rMlp=4,
            emb_drop=0.,
            attn_drop=0.,
            proj_drop=0.,
            mlp_drop=0.,
            drop_path_rate=0.,
            reversible=False,
            attn_dropout=0.0,
            ff_mult=4,
            ff_dropout=0,
            attn_types=None,
            axial_pos_emb=False,
            fixed_pos_emb=True
    ):
        super().__init__()

        self.rad_vae = set_requires_grad(rad_vae, requires_grad=False).eval()
        self.cam_vae = set_requires_grad(cam_vae, requires_grad=False).eval()

        self.rad_nToks = rad_vae.discretizer.nToks
        self.cam_nToks = cam_vae.discretizer.nToks
        self.rad_dToks = rad_vae.discretizer.dToks
        self.cam_dToks = cam_vae.discretizer.dToks

        print('rad token', self.rad_nToks)
        print('cam token', self.cam_nToks)

        self.total_nToks = self.rad_nToks + self.cam_nToks

        self.axial_pos_emb = axial_pos_emb
        self.pkeep = pkeep
        self.recycle_emb = recycle_emb
        self.only_cam_labels = only_cam_labels
        self.separate_embs = separate_embs
        self.fixed_pos_emb = fixed_pos_emb

        self.rad_seq_len = (self.rad_vae.inp_size // (2 ** self.rad_vae.nLayers)) ** 2
        self.cam_seq_len = (self.cam_vae.inp_size // (2 ** self.cam_vae.nLayers)) ** 2
        self.total_seq_len = self.rad_seq_len + self.cam_seq_len

        vocab_size = max(self.rad_nToks, self.cam_nToks)
        if transformer == "Vanilla":
            self.transformer = Transforming(
                lEmb=lEmb,
                nHead=nHead,
                dHead=dHead,
                # nToks=vocab_size if only_cam_labels else self.total_nToks,
                nToks=self.total_nToks if self.separate_embs else vocab_size,
                nPred=self.cam_nToks if self.only_cam_labels else self.total_nToks,
                nLayer=nLayer,
                rMlp=rMlp,
                emb_drop=emb_drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_drop=mlp_drop,
                drop_path_rate=drop_path_rate)


        # elif transformer == "Phil":
        # lEmb = self.rad_dToks if recycle_emb else lEmb
        # self.transformer = Transformer(
        # dim=lEmb,
        # causal=True,
        # seq_len=self.total_seq_len,
        # depth=nLayer,
        # heads=nHead,
        # dim_head=dHead,
        # reversible=reversible,
        # attn_dropout=attn_dropout,
        # ff_dropout=ff_dropout,
        # attn_types= attn_types,
        # image_fmap_size = int(sqrt(self.rad_seq_len)),
        # ff_mult=ff_mult,
        # sparse_attn = False
        # )

            # if not fixed_pos_emb:
            # if axial_pos_emb:
            # self.rad_pos_emb = AxialPositionalEmbedding(
            # lEmb, axial_shape=(256, 256)
            # )
            # self.cam_pos_emb = AxialPositionalEmbedding(
            # lEmb, axial_shape=(256, 256)
            # )
            # else:
            # self.rad_pos_emb = nn.Embedding(self.rad_seq_len, lEmb)
            # self.cam_pos_emb = nn.Embedding(self.cam_seq_len, lEmb)
            # #self.rad_pos_emb = nn.Parameter(torch.zeros(1, self.rad_seq_len, lEmb))
            # #self.cam_pos_emb = nn.Parameter(torch.zeros(1, self.cam_seq_len, lEmb))
            # #self.rad_pos_emb = nn.Parameter(torch.randn(1, self.rad_seq_len, lEmb))
            # #self.cam_pos_emb = nn.Parameter(torch.randn(1, self.cam_seq_len, lEmb))
            # else:
            # self.rad_pos_emb = FixedPositionalEncoding(lEmb, self.rad_seq_len)
            # self.cam_pos_emb = FixedPositionalEncoding(lEmb, self.cam_seq_len)

        # #if isinstance(self.transformer, Transformer):
        # if recycle_emb: # if codebooks are used as trafo embeddings, they are frozen
        # self.rad_emb = self.rad_vae.discretizer.embedding
        # self.cam_emb = self.cam_vae.discretizer.embedding
        # else:
        # self.rad_emb = nn.Embedding(self.rad_nToks, lEmb)
        # self.cam_emb = nn.Embedding(self.cam_nToks, lEmb)

            # self.to_logits = nn.Sequential(
            # nn.LayerNorm(lEmb),
            # nn.Linear(
            # lEmb, self.cam_nToks if only_cam_labels else self.total_nToks
            # ),
            # )

        if not self.only_cam_labels:
            seq_range = torch.arange(self.total_seq_len)
            logits_range = torch.arange(self.total_nToks)

            # self.rad_seq_len = 2
            # self.cam_seq_len = 2
            # self.rad_nToks = 4
            # self.cam_nToks = 4
            # self.total_seq_len = self.rad_seq_len + self.cam_seq_len
            # self.total_nToks = self.rad_nToks + self.cam_nToks
            # seq_range = torch.arange(self.total_seq_len)
            # logits_range = torch.arange(self.total_nToks)

            seq_range = rearrange(seq_range, "n -> () n ()")
            logits_range = rearrange(logits_range, "d -> () () d")

            logits_mask = (
                (seq_range >= self.rad_seq_len) & (logits_range < self.rad_nToks)
            ) | ((seq_range < self.rad_seq_len) & (logits_range >= self.rad_nToks))

            self.register_buffer("logits_mask", logits_mask)
            # print(logits_mask)
            # print(self.logits_mask[:, 1:].shape)
            # print(self.logits_mask[:, 1:])
            # exit()

    def forward(self, rad, cam, soft_idx=False):
        rad_toks = self.rad_vae.discretize(rad, soft_idx)
        cam_toks = self.cam_vae.discretize(cam, soft_idx)

        # r = set(rad_toks[0].tolist())
        # c = set(cam_toks[0].tolist())
        # common = r.intersection(c)
        # print(common)
        # exit()

        cam_targets = cam_toks.detach().clone()

        # disturbing training by replacing cam toks with random indices
        # with a probability of (1-pkeep) for every token individually
        # if self.training and self.pkeep < 1.0:
        # mask = torch.bernoulli(
        # self.pkeep*torch.ones(cam_toks.shape,
        # device=cam_toks.device)).int()
        # rnd_indices = torch.randint_like(cam_toks, self.cam_nToks)
        # cam_toks = mask*cam_toks+(1-mask)*rnd_indices

        if isinstance(self.transformer, Transformer):
            ...
            # with torch.set_grad_enabled(not self.recycle_emb):
            # rad_emb = self.rad_emb(rad_toks)
            # cam_emb = self.cam_emb(cam_toks)
            # if self.axial_pos_emb:
            # rad_emb += self.rad_pos_emb(rad_emb)
            # cam_emb += self.cam_pos_emb(cam_emb)
            # elif self.fixed_pos_emb:
            # rad_emb = self.rad_pos_emb(rad_emb)
            # cam_emb = self.cam_pos_emb(cam_emb)
            # else:
            # rad_emb += self.cam_pos_emb.weight
            # cam_emb += self.rad_pos_emb.weight
            # #rad_emb += self.rad_pos_emb
            # #cam_emb += self.cam_pos_emb
            # total_toks = torch.cat([rad_emb, cam_emb], dim=1)
            # out = self.transformer(total_toks[:, :-1])
            # out = self.to_logits(out)
        else:
            if self.separate_embs:
                cam_toks += self.rad_nToks
                total_toks = torch.cat([rad_toks, cam_toks], dim=1)
            else:
                total_toks = torch.cat([rad_toks, cam_toks], dim=1)
                out = self.transformer(total_toks[:, :-1])

        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        if self.only_cam_labels:
            targets = cam_targets
            out_cam = out[:, self.rad_seq_len - 1 :]
        else:
            max_neg_value = -torch.finfo(out.dtype).max
            out.masked_fill_(self.logits_mask[:, 1:], max_neg_value)
            # offset_cam_targets = cam_toks + self.rad_nToks
            offset_cam_targets = cam_targets + self.rad_nToks
            targets = torch.cat([rad_toks[:, 1:], offset_cam_targets], dim=1)
            # targets = torch.cat([rad_toks, offset_cam_targets], dim=1)
            out_cam = out

        return out_cam.transpose(1, 2), targets  # 2D-CrossEntropy

    @torch.no_grad()
    @eval_decorator
    def synthesize(
            self, rad, cam, temperature=1.0, thld=1.0, num_samples=1, soft_idx=False
    ):
        assert (
            not self.transformer.training
        ), "disable transformer dropout for infereence etc."

        print(f'Sampling camera toks from a multinomial with ' +
              f'{max(int((1 - thld) * self.cam_nToks), 1)} realizations')

        rad_toks = self.rad_vae.discretize(rad, soft_idx)
        cam_toks = self.cam_vae.discretize(cam, soft_idx)

        cam_toks_pred = torch.IntTensor().to(rad_toks)

        if isinstance(self.transformer, Transformer):
            ...
            # total_toks = self.rad_emb(rad_toks)
        else:
            total_toks = rad_toks

        for i in range(self.cam_seq_len):
            print(f"Predicting camera token: {i+1}/{self.cam_seq_len}", end='\r')
            if isinstance(self.transformer, Transformer):
                ...
                # if self.axial_pos_emb:
                # total_toks[:, : self.rad_seq_len] += self.rad_pos_emb(
                # total_toks[:, : self.rad_seq_len]
                # )
                # total_toks[:, self.rad_seq_len :] += self.cam_pos_emb(
                # total_toks[:, self.rad_seq_len :]
                # )
                # elif self.fixed_pos_emb:
                # total_toks[:, : self.rad_seq_len] = self.rad_pos_emb(total_toks[:, : self.rad_seq_len])
                # total_toks[:, self.rad_seq_len :] = self.cam_pos_emb(
                # total_toks[:, self.rad_seq_len :]
                # )
                # else:
                # #total_toks[:, : self.rad_seq_len] += self.rad_pos_emb[:, : self.rad_seq_len]
                # #total_toks[:, self.rad_seq_len :] += self.cam_pos_emb[:, :i]
                # total_toks[:, : self.rad_seq_len] += self.cam_pos_emb.weight[: self.rad_seq_len]
                # total_toks[:, self.rad_seq_len :] += self.rad_pos_emb.weight[:i]

                # logits = self.transformer(total_toks)
                # logits = self.to_logits(logits)[:, -1]
                # if not self.only_cam_labels:
                # logits_mask = self.logits_mask[:, -1]
                # max_neg_value = -torch.finfo(logits.dtype).max
                # logits.masked_fill_(logits_mask, max_neg_value)
                # selected_logits = top_k_ratio(logits, thld)
                # #selected_logits = torch.randn_like(selected_logits)
                # probs = F.softmax(selected_logits / temperature, dim=-1)
                # sample = torch.multinomial(probs, num_samples=num_samples)
                # sample -= self.rad_nToks if not self.only_cam_labels else 0
                # cam_emb = self.cam_emb(sample)
                # #cam_emb += self.cam_pos_emb(cam_emb)
                # total_toks = torch.cat([total_toks, cam_emb], dim=1)
            else:
                # print(total_toks.min())
                # print(total_toks.max())
                logits = self.transformer(total_toks)
                logits = logits[:, -1]
                if not self.only_cam_labels:
                    logits_mask = self.logits_mask[:, -1]
                    max_neg_value = -torch.finfo(logits.dtype).max
                    logits.masked_fill_(logits_mask, max_neg_value)
                    selected_logits = top_k_ratio(logits, thld)
                    probs = F.softmax(selected_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, num_samples=num_samples)
                    # print('sample', sample.shape)
                    # print('sample', sample)
                if not self.only_cam_labels:
                    sample -= self.rad_nToks
                    # print('sample', sample.shape)
                    # print('sample', sample)
                total_toks = torch.cat([total_toks, sample], dim=1)

            cam_toks_pred = torch.cat([cam_toks_pred, sample], dim=1)


        # if not self.only_cam_labels:
        # if not self.only_cam_labels and not isinstance(self.transformer, Transformer):
        # cam_toks_pred -= self.rad_nToks

        # rand_dummy = torch.randint(0, self.cam_nToks, cam_toks_pred.size()).cuda()
        # l = []
        # for i in range(8):
        # c = torch.arange(i * self.cam_seq_len, self.cam_seq_len * (i + 1)).expand_as(cam_toks_pred).cuda()
        # print(c.min())
        # print(c.max())
        # l.append(self.cam_vae.constitute(c))
        # print(cam_toks_pred.shape)
        # cam_toks_pred = torch.ones_like(cam_toks_pred) * 2000

        cam_prd = self.cam_vae.constitute(cam_toks_pred)
        # correct_constitutes = torch.sum(cam_toks == cam_toks_pred, dim=1)
        # correct_constitutes = correct_constitutes.float().mean()
        correct_constitutes = (cam_toks == cam_toks_pred).float().mean()
        sim = nn.CosineSimilarity(dim=1, eps=1e-6)(cam_toks_pred.float(), cam_toks.float()).mean()

        # cam_prd = self.cam_vae.constitute(cam_toks_pred - self.rad_nToks)
        cam_dec = self.cam_vae.constitute(cam_toks)
        rad_dec = self.rad_vae.constitute(rad_toks)
        # return *l, correct_constitutes
        return cam_prd, cam_dec, torch.rot90(rad_dec, 1, [-2, -1]), correct_constitutes, sim
"""
