import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

#from timm.models.layers import DropPath
import pdb
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DisAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim * num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, weight=None):
        #pdb.set_trace()
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, mu_heads_num+logsig_heads_num, n, dim_heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask            
        attn = attn.softmax(dim=-1)
        if weight is not None:
            weight = weight.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, N, -1).reshape(*attn.shape)
            attn = attn * (weight + 1e-10)
            attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C/2))

        mu = x[:,:,0,:]
        logsigma = x[:,:,1,:]
        mu = self.mu_proj(mu)
        mu = self.mu_proj_drop(mu)
        logsigma = self.logsig_proj(logsigma)
        logsigma = self.logsig_proj_drop(logsigma)
        return mu, logsigma, attn


class DisTrans(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = drop_path(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.logsig_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # MLP version
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)
        self.mu_proj_drop = nn.Dropout(drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)
        self.logsig_proj_drop = nn.Dropout(drop)
    def forward(self, x, mask=None, weight=None):
        # Trans Version
       # pdb.set_trace()
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_, mask=mask, weight=weight)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))

        # MLP version 3dim
        # x_ = self.norm1(self.act(self.fc(x)))
        # B, N, C = x_.shape
        # x_ = x_.reshape(B, N, 2, int(C/2))
        # mu = x_[:,:,0,:]
        # logsigma = x_[:,:,1,:]
        # mu = self.mu_proj(mu)
        # mu = self.mu_proj_drop(mu)
        # logsigma = self.logsig_proj(logsigma)
        # logsigma = self.logsig_proj_drop(logsigma)
        # mu = x + self.drop_path(mu)
        # mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        # logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        # attn = None

        # MLP version 2dim
        # x_ = self.norm1(self.act(self.fc(x)))
        # B, C = x_.shape
        # x_ = x_.reshape(B, 2, int(C/2))
        # mu = x_[:,0,:]
        # logsigma = x_[:,1,:]
        # mu = self.mu_proj(mu)
        # mu = self.mu_proj_drop(mu)
        # logsigma = self.logsig_proj(logsigma)
        # logsigma = self.logsig_proj_drop(logsigma)
        # mu = x + self.drop_path(mu)
        # mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        # logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        # attn = None

        #

        return mu, logsigma, attn

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1,1)
        return t

    def forward(self, x):
        #pdb.set_trace()
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim= 2 , keepdim=True)   # bs,V/T,1
        std = (x.var(dim=2, keepdim=True) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)# bs,V/T,1
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)  
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean) / (std + self.eps)
        x = x * gamma + beta

        return x