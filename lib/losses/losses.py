import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange

from lib import instantiate_or_load, weights_init

# cf. https://github.com/ardasnck/learning_to_localize_sound_source/blob/master/losses.py
import numpy as np


class Triplet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_v, f_pos, f_neg, dev):
        d_pos = F.pairwise_distance(f_v, f_pos, keepdim=False)
        d_neg = F.pairwise_distance(f_v, f_neg, keepdim=False)
        dist_vector = torch.stack((d_pos, d_neg), 1)
        dist_softmax = F.softmax(dist_vector, dim=-1)
        dist_target = torch.stack(
            (torch.zeros((f_v.size(0))), torch.ones((f_v.size(0)))), 1
        ).to(dev)
        loss = dist_softmax - dist_target
        loss = loss ** 2
        loss = loss.mean()
        return loss

    """ min/max emb vectores L3 model zissermann
   img = self.icn(img)
        img = self.img_pool(img)
        img = img.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]
        img = F.relu(self.img_fc1(img))
        img_emb = F.normalize(self.img_fc2(img), p=2, dim=1)  # L2 normalization

        # audio subnetwork
        aud = self.acn(aud)
        aud = self.aud_pool(aud)
        aud = aud.squeeze(2).squeeze(2)  # [N, 512, 1, 1] to [N, 512]
        aud = F.relu(self.aud_fc1(aud))
        aud_emb = F.normalize(self.aud_fc2(aud), p=2, dim=1)  # L2 normalization

        # fusion network
        euc_dist = ((img_emb - aud_emb) ** 2).sum(dim=1, keepdim=True).sqrt()  # Euclidean distance
        out = self.fc3(euc_dist)
        return out, img_emb, aud_emb
"""


class Contrast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cam_f, rad_f, lab, **kw):
        lab_inv = torch.where(lab.bool(), 0, 1)
        dist = F.pairwise_distance(rad_f, cam_f, keepdim=False)
        loss = (dist.sigmoid() - lab_inv) ** 2
        return loss.mean()


def computeMatchmap(rad, cam):
    return torch.einsum("cij, ckl -> ijkl", rad, cam)


def matchmapSim(M, simtype):
    if simtype == "SISA":
        return M.mean()
    elif simtype == "MISA":
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == "SIMA":
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError


class MarginRanked(nn.Module):
    """ https://github.com/dharwath/DAVEnet-pytorch/blob/master/steps/util.py """

    def __init__(self):
        super().__init__()

    def forward(self, cam_f, rad_f, margin=1, simtype="MISA"):
        bsze = cam_f.size(0)
        assert bsze > 1
        loss = torch.zeros(1, device=cam_f.device, requires_grad=True)

        att_map = torch.zeros(
            (bsze, 15, 15, 15, 15), device=cam_f.device, requires_grad=False
        )

        for i in range(bsze):
            rad_i = i
            cam_i = i
            while rad_i == i:
                rad_i = np.random.randint(0, bsze)
            while cam_i == i:
                cam_i = np.random.randint(0, bsze)
                # print('real idx', i)
                # print('rad_i', rad_i)
                # print('cam_i', cam_i)
            anchorsim = matchmapSim(computeMatchmap(cam_f[i], rad_f[i]), simtype)
            Iimpsim = matchmapSim(computeMatchmap(cam_f[cam_i], rad_f[i]), simtype)
            Aimpsim = matchmapSim(computeMatchmap(cam_f[i], rad_f[rad_i]), simtype)

            A2I_simdif = margin + Iimpsim - anchorsim
            if (A2I_simdif > 0).all():
                loss = loss + A2I_simdif
                I2A_simdif = margin + Aimpsim - anchorsim
            if (I2A_simdif > 0).all():
                loss = loss + I2A_simdif

            att_map[i] = computeMatchmap(cam_f[i], rad_f[i])

        loss = loss / bsze
        # print(att_map.shape)
        M_maxH, _ = att_map.max(1)
        M_maxHW, _ = M_maxH.max(1)
        M_maxHW = M_maxHW.sigmoid()

        return loss, M_maxHW.unsqueeze(1)


def compute_matchmap_similarity_matrix(
        image_outputs, audio_outputs, nframes, simtype="MISA"
):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert image_outputs.dim() == 4
    assert audio_outputs.dim() == 3
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
        for audio_idx in range(n):
            nF = max(1, nframes[audio_idx])
            S[image_idx, audio_idx] = matchmapSim(
                computeMatchmap(
                    image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]
                ),
                simtype,
            )
    return S


def calc_recalls(image_outputs, audio_outputs, nframes, simtype="MISA"):
    """
    Computes recall at 1, 5, and 10 given encoded image and audio outputs.
    """
    S = compute_matchmap_similarity_matrix(
        image_outputs, audio_outputs, nframes, simtype=simtype
    )
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
                # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
            # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
            # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {
        "A_r1": A_r1.avg,
        "A_r5": A_r5.avg,
        "A_r10": A_r10.avg,
        "I_r1": I_r1.avg,
        "I_r5": I_r5.avg,
        "I_r10": I_r10.avg,
    }
    #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls


def validate(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
        audio_model = audio_model.to(device)
        image_model = image_model.to(device)
        # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = []
    A_embeddings = []
    frame_counts = []
    with torch.no_grad():
        for i, (image_input, audio_input, nframes) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # compute output
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input)

            image_output = image_output.to("cpu").detach()
            audio_output = audio_output.to("cpu").detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.div_(pooling_ratio)

            frame_counts.append(nframes.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nframes = torch.cat(frame_counts)

        recalls = calc_recalls(
            image_output, audio_output, nframes, simtype=args.simtype
        )
        A_r10 = recalls["A_r10"]
        I_r10 = recalls["I_r10"]
        A_r5 = recalls["A_r5"]
        I_r5 = recalls["I_r5"]
        A_r1 = recalls["A_r1"]
        I_r1 = recalls["I_r1"]

    print(
        " * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs".format(
            A_r10=A_r10, I_r10=I_r10, N=N_examples
        ),
        flush=True,
    )
    print(
        " * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs".format(
            A_r5=A_r5, I_r5=I_r5, N=N_examples
        ),
        flush=True,
    )
    print(
        " * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs".format(
            A_r1=A_r1, I_r1=I_r1, N=N_examples
        ),
        flush=True,
    )

    return recalls


@instantiate_or_load
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()

        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def __call__(self, prediction, target_is_real=True):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.real_label if target_is_real else self.fake_label
            target_tensor = target_tensor.expand_as(prediction)
            return self.loss(prediction, target_tensor)

        elif self.gan_mode == "wgangp":
            return -prediction.mean() if target_is_real else prediction.mean()


@instantiate_or_load
class dVAECritic(nn.Module):
    """The Discriminator has the job of taking two images, an input image and an
    unknown image (which will be either a target or output image from the
    generator), and deciding if the second image was produced by the generator
    or not. """
    def __init__(
            self, critic, enable_after_iter, weight, com_loss, cri_loss, init_critic
    ):
        super().__init__()
        self.critic = critic if not init_critic else critic.apply(weights_init)
        self.weight = weight
        self.com_loss = com_loss
        self.cri_loss = cri_loss
        self.enable_after_iter = enable_after_iter

    def __call__(self, whatever):
        return whatever
        # raise NotImplementedError

    def start_criticising(self, train_step):
        return train_step > self.enable_after_iter

    # @torch.no_grad()
    def criticise(self, rec, cond):
        verdict_on_fakes = self.critic(
            rec.contiguous()
            if not self.critic.conditions
            else torch.cat([rec.contiguous(), cond], dim=1)
        )
        return self.weight * self.com_loss(verdict_on_fakes, target_is_real=True)

    # @torch.enable_grad()
    def reflect(self, x, rec, cond):
        logits_real = self.critic(
            x.contiguous().detach()
            if not self.critic.conditions
            else torch.cat([x.contiguous().detach(), cond.contiguous().detach()], dim=1)
        )
        logits_fake = self.critic(
            rec.contiguous().detach()
            if not self.critic.conditions
            else torch.cat(
                    [rec.contiguous().detach(), cond.contiguous().detach()], dim=1
            )
        )
        return self.cri_loss(logits_real, logits_fake)


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight) for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
                err = self._forward(preds, targets, weight)
        elif isinstance(preds, torch.distributions.distribution.Distribution):
            if weight is None:
                weight = preds.logits.new_ones(1)
                err = self._forward(preds, targets, weight)
        return err



@instantiate_or_load
class NLLLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    @classmethod
    def inmap(cls, x):
        # x = x - 0.5
        return x - 0.5 # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return ((x + 0.5) * 255).long()
        # return (torch.clamp(x + 0.5, 0, 1) * 255).long()
        # return torch.floor((x + 0.5) * 255).long()
        # return (x + 0.5).long()
        # return x.long()
        # return torch.clamp(x + 0.5, 0, 1)

    def _forward(self, pred_dist, target, weight):
        # print(target.min())
        # print(target.max())
        # target = self.unmap(target)
        target = target.long()
        # print(target.min())
        # print(target.max())
        # exit()
        # print(pred_dist.logits.shape)
        logp = pred_dist.log_prob(target)
        return -logp.sum((1, 2, 3)).mean()


@instantiate_or_load
class L1Loss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


@instantiate_or_load
class L2Loss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


@instantiate_or_load
class Huber(BaseLoss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred, target, weight):
        return F.smooth_l1_loss(pred, target)


@instantiate_or_load
class BCELoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

@instantiate_or_load
class BCELogitsLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)




class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_real, logits_fake):
        loss_real = torch.mean(
            F.relu(1.0 - logits_real)
        )  # push towards >= 1 for loss = 0
        loss_fake = torch.mean(
            F.relu(1.0 + logits_fake)
        )  # push towards <= -1 for loss = 0
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss


class VanillaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
        )
        return d_loss


# https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/timm/loss/cross_entropy.py
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        print(logprobs.shape)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        print(target.unsqueeze(1).shape)
        print(nll_loss.shape)
        exit()
        nll_loss = nll_loss.squeeze(1)
        # bis hierhin normaler ce-loss
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/jsd.py
class JsdCrossEntropy(nn.Module):
    """Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += (
            self.alpha
            * sum(
                [
                    F.kl_div(logp_mixture, p_split, reduction="batchmean")
                    for p_split in probs
                ]
            )
            / len(probs)
        )
        return loss



"""
VQVAE losses, used for the reconstruction term in the ELBO
"""
# -----------------------------------------------------------------------------

class LogitLaplaceLoss(nn.Module):
    """ the Logit Laplace distribution log likelihood from OpenAI's DALL-E paper """
    logit_laplace_eps = 0.1

    @classmethod
    def inmap(cls, x):
        # map [0,1] range to [eps, 1-eps]
        return (1 - 2 * cls.logit_laplace_eps)/ 255 * x + cls.logit_laplace_eps

    @classmethod
    def unmap(cls, x):
        # inverse map, from [eps, 1-eps] to [0,1], with clamping
        return torch.clamp((x - cls.logit_laplace_eps)
                           / (1 - 2 * cls.logit_laplace_eps), 0, 1)

    @classmethod
    def logit(cls, x):
        return torch.log(x / (1 - x) + 1e-10)
        # return torch.log(x / (1 - x))
        # return torch.log(x)
        # return torch.sigmoid(x / (1 - x))
        # return x / (1 - x)

    def forward(self, x, mu_logb):
        num_samples = x.numel() // x.shape[0]

        # print(x.min())
        # print(x.max())
        # x = LogitLaplaceLoss.inmap(x)
        log_b = mu_logb[:, 3:]
        b = torch.exp(log_b)
        mu = mu_logb[:, :3]
        # print('x', x.min())
        # print('x', x.max())
        # print('b', b.min())
        # print('b', b.max())
        # print('mu', mu.min())
        # print('m', mu.max())
        # print('log_b', log_b.min())
        # print('log_b', log_b.max())

        # pre = -num_samples * log_b - torch.log(2 * x * (1 - x) + 1e-10)
        pre = -log_b - torch.log(2 * x * (1 - x))
        # print('c', c.min())
        # print('c', c.max())
        # pre = -log_b + c
        # pre = 2 * log_b * x * (1 - x)

        pre = torch.sum(pre, dim=(1, 2, 3))
        # print('pre', pre.min())
        # print('pre', pre.max())

        fraction = torch.sum(torch.abs(LogitLaplaceLoss.logit(x) - mu) / b, dim=(1, 2, 3))
        # print('fraction', fraction.min())
        # print('fraction', fraction.max())

        total = pre - fraction
        # print('total', total.min())
        # print('total', total.max())

        # total = torch.sum(total, dim=(1, 2, 3))

        val = torch.mean(total)
        assert val > 0, f'negative value {val}'
        # print(val)
        # exit()
        return val


@instantiate_or_load
class NormalLoss(nn.Module):
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """
    # data_variance = 0.06327039811675479 # cifar-10 data variance, from deepmind sonnet code
    def __init__(self, var):
        super().__init__()
        # self.data_variance = 0.06327039811675479
        self.unit_variance = torch.FloatTensor(var).view(-1, 1, 1) # calculated a priori

    @classmethod
    def inmap(cls, x):
        return x - 0.5 # map [0,1] range to [-0.5, 0.5]

    # @classmethod
    def unmap(self, x):
        return torch.clamp(x + 0.5, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu)**2).mean() / (2 * cls.data_variance) #+ math.log(math.sqrt(2 * math.pi * cls.data_variance))

    def __call__(self, x, mu):
        # return ((x - mu)**2).mean() / (2 * self.unit _variance) #+ math.log(math.sqrt(2 * math.pi * cls.data_variance))
        return ((x -  mu) ** 2 / (2 * self.unit_variance.to(x))).mean()
        # return ((x -  mu) ** 2 / 1.0).mean()
        # return torch.mean(weight * torch.pow(pred - target, 2))
