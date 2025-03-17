import torch
import torch.nn.functional as F
import pdb
from .until_module import CrossEn


def Wasserstein2(mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差
    bs1 = mu1.shape[0]
    bs2 = mu2.shape[0]
    mu1 = torch.stack([mu1]*bs2, dim=1)
    sigma1 = torch.stack([sigma1]*bs2, dim=1)
    mu2 = torch.stack([mu2]*bs1, dim=0)
    sigma2 = torch.stack([sigma2]*bs1, dim=0)
    p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)
    p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)
    return p1+p2, p1

def compute_dis_contrast(txt_mu, txt_sigma, vid_mu, vid_sigma, negative_scale = 1/2000, shift = 4, temp = 0.01):
    loss_fct = CrossEn()
    # vid_mu = vid_mu[:, 0]
    # vid_sigma = torch.exp(vid_logsigma[:, 0])
    # txt_mu = txt_mu[:, 0]
    # txt_sigma = torch.exp(txt_logsigma[:, 0])

    # pl_module.log('con/img_sigma_mean', torch.mean(vid_sigma), on_step=True)
    # pl_module.log('con/txt_sigma_mean', torch.mean(txt_sigma), on_step=True)
    
    bs = vid_mu.shape[0]
    # phase = "train" if pl_module.training else "val"

    # gather
    # allgather = AllGather_multi.apply
    # vid_mu = allgather(vid_mu)
    # txt_mu = allgather(txt_mu)
    # vid_sigma = allgather(vid_sigma)
    # txt_sigma = allgather(txt_sigma)

    W2_distance, mu_distance = Wasserstein2(vid_mu, vid_sigma, txt_mu, txt_sigma)
    similarity = (-negative_scale * W2_distance + shift) / temp

    labels = torch.arange(bs).to(similarity.device)
    loss = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    # loss = loss_fct(similarity) + loss_fct(similarity.transpose(0, 1)
    
    # pl_module.log(f"contrast/{phase}/loss", loss)
    # pl_module.log("temperature", pl_module.temp)

    # ret = {'contrast_loss': loss}

    return loss

def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    nll_loss = nll_loss.view(bsz, num_props)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask) 
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
            })

    return final_loss, loss_dict

    
def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()
    else:
        rank_loss = min_nll_loss.mean()
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = kwargs['alpha_1'] * rank_loss

    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    loss = loss + kwargs['alpha_2'] * div_loss.mean()

    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item()
    }