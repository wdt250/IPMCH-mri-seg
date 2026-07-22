import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# from timm.models.registry import register_model
from timm.models import register_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
 
from torch import Tensor
from typing import Tuple
import numbers
# from timm.models.layers import to_2tuple, trunc_normal_
from timm.layers import to_2tuple, trunc_normal_
from einops import rearrange
import gc
import torch
import torch.nn as nn
from einops import rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models import register_model
from timm.models.vision_transformer import _cfg
 
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k
 
 
class LayerNorm2d(nn.Module):
 
  def __init__(self, 
               channels
               ):
    super().__init__()
    self.ln = nn.LayerNorm(channels)
 
  def forward(self, x):
    x = rearrange(x, "N C H W -> N H W C")
    x = self.ln(x)
    x = rearrange(x, "N H W C -> N C H W")
    return x
 
 
 
def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
 
#def to_4d(x,s,h,w):
#    return rearrange(x, 'b (s h w) c -> b c s h w',s=s,h=h,w=w)
 
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
 
#def to_3d(x):
#    return rearrange(x, 'b c s h w -> b (s h w) c')
 
class Partial:
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.args = args
        self.kwargs = kwargs
 
    def __call__(self, *args_c, **kwargs_c):
        return self.module(*args_c, *self.args, **kwargs_c, **self.kwargs)
 
 
 
class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
 
    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x
 
class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)
 
    def forward(self, x):
 
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')
    
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
 
        assert len(normalized_shape) == 1
 
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
 
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
 
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
 
        assert len(normalized_shape) == 1
 
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
 
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
 
 
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    
#class LayerNorm(nn.Module):
#    def __init__(self, dim, LayerNorm_type):
#        super(LayerNorm, self).__init__()
#        if LayerNorm_type =='BiasFree':
#            self.body = BiasFree_LayerNorm(dim)
#        else:
#            self.body = WithBias_LayerNorm(dim)
#    def forward(self, x):
#        s, h, w = x.shape[-3:]
#        return to_4d(self.body(to_3d(x)),s, h, w)
    
 
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
 
    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
 
        return x
    
class ConvFFN(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 1, 1, 0)
 
    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
 
        return x
    
 
 
class Attention(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
 
        #######################################
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################
 
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x
 
class AttentionLePE(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
 
    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
 
        #######################################
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
 
        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')
 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
 
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################
 
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x
 
 
 
class nchwAttentionLePE(nn.Module):
    """
    Attention with LePE, takes nchw input
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
 
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
 
    def forward(self, x:torch.Tensor):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        B, C, H, W = x.size()
        q, k, v = self.qkv.forward(x).chunk(3, dim=1) # B, C, H, W
 
        attn = q.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2) @ \
               k.view(B, self.num_heads, self.head_dim, H*W)
        attn = torch.softmax(attn*self.scale, dim=-1)
        attn = self.attn_drop(attn)
 
        # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
        output:torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)
 
        output = self.proj_drop(self.proj(output))
 
        return output
 
 
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
 
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
 
        return r_weight, topk_index
 
 
class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight
 
    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )
 
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing
 
        return topk_kv
 
class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
 
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v
 
 
class QKVConv(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Conv2d(dim,  qk_dim + qk_dim + dim, 1, 1, 0)
 
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=1)
        return q, kv
    
 
        
class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5
 
 
        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)
 
        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')
 
        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)
 
        self.auto_pad=auto_pad
 
    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
 
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            
            #assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################
 
 
        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
 
        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 
 
        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
 
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)
 
        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
 
        ############ gather q dependent k/v #################
 
        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors
 
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
 
        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)
 
        out = out + lepe
        # output linear
        out = self.wo(out)
 
        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
 
        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out
        
        
class TransformerMLPWithConv(nn.Module):
 
    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x
    
        
class DeBiLevelRoutingAttentionblcok(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=True, param_size='small'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
 
#############################################################
        if param_size=='tiny':
            if self.dim == 64 :
                self.n_groups = 1
                self.top_k_def = 16   # 2    128
                self.kk = 9
                self.stride_def = 8
                self.expain_ratio = 3
                self.q_size=to_2tuple(56)
 
            if self.dim == 128 :
                self.n_groups = 2
                self.top_k_def = 16   # 4    256
                self.kk = 7
                self.stride_def = 4
                self.expain_ratio = 3
                self.q_size=to_2tuple(28)
 
            if self.dim == 256 :
                self.n_groups = 4
                self.top_k_def = 4   # 8    512
                self.kk = 5
                self.stride_def = 2
                self.expain_ratio = 3
                self.q_size=to_2tuple(14)
 
            if self.dim == 512 :
                self.n_groups = 8
                self.top_k_def = 49   # 8    512
                self.kk = 3
                self.stride_def = 1
                self.expain_ratio = 3
                self.q_size=to_2tuple(7)
#############################################################
        if param_size=='small':
            if self.dim == 64 :
                self.n_groups = 1
                self.top_k_def = 16   # 2    128
                self.kk = 9
                self.stride_def = 8
                self.expain_ratio = 3
                self.q_size=to_2tuple(56)
 
            if self.dim == 128 :
                self.n_groups = 2
                self.top_k_def = 16   # 4    256
                self.kk = 7
                self.stride_def = 4
                self.expain_ratio = 3
                self.q_size=to_2tuple(28)
 
            if self.dim == 256 :
                self.n_groups = 4
                self.top_k_def = 4   # 8    512
                self.kk = 5
                self.stride_def = 2
                self.expain_ratio = 3
                self.q_size=to_2tuple(14)
 
            if self.dim == 512 :
                self.n_groups = 8
                self.top_k_def = 49   # 8    512
                self.kk = 3
                self.stride_def = 1
                self.expain_ratio = 1
                self.q_size=to_2tuple(7)
#############################################################
        if param_size=='base':
            if self.dim == 96 :
                self.n_groups = 1
                self.top_k_def = 16   # 2    128
                self.kk = 9
                self.stride_def = 8
                self.expain_ratio = 3
                self.q_size=to_2tuple(56)
 
            if self.dim == 192 :
                self.n_groups = 2
                self.top_k_def = 16   # 4    256
                self.kk = 7
                self.stride_def = 4
                self.expain_ratio = 3
                self.q_size=to_2tuple(28)
 
            if self.dim == 384 :
                self.n_groups = 3
                self.top_k_def = 4   # 8    512
                self.kk = 5
                self.stride_def = 2
                self.expain_ratio = 3
                self.q_size=to_2tuple(14)
                
            if self.dim == 768 :
                self.n_groups = 6
                self.top_k_def = 49   # 8    512
                self.kk = 3
                self.stride_def = 1
                self.expain_ratio = 3
                self.q_size=to_2tuple(7)
            
 
        self.q_h, self.q_w = self.q_size
        
        self.kv_h, self.kv_w = self.q_h // self.stride_def, self.q_w // self.stride_def
        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.num_heads // self.n_groups
        self.n_group_channels = self.dim // self.n_groups
        
        self.offset_range_factor = -1
        self.head_channels = dim // num_heads
 
        self.n_group_heads = self.num_heads // self.n_groups
 
        #assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5
        
        
        
        
        self.rpe_table = nn.Parameter(
                    torch.zeros(self.num_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
        trunc_normal_(self.rpe_table, std=0.01)
 
 
 
 
        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe1 = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=self.stride_def, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
 
 
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
 
 
        # router
        #assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
 
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)
 
 
 
 
        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            #self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.qkv_conv = QKVConv(self.dim, self.qk_dim)
            #self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            #self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.qkv_conv = QKVConv(self.dim, self.qk_dim)
            #self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
 
 
 
 
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')
 
        self.attn_act = nn.Softmax(dim=-1)
 
        self.auto_pad=auto_pad
 
 
 
 
##########################################################################################
 
        self.proj_q = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_k = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )
        
        self.unifyheads1 = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )
 
        self.conv_offset_q = nn.Sequential(
                        nn.Conv2d(self.n_group_channels, self.n_group_channels, (self.kk,self.kk), (self.stride_def,self.stride_def), (self.kk//2,self.kk//2), groups=self.n_group_channels, bias=False),
                        LayerNormProxy(self.n_group_channels),
                        nn.GELU(),
                        nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False),
                )
 
 
### FFN
 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
 
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
 
        self.mlp =TransformerMLPWithConv(dim, self.expain_ratio, 0.)
 
 
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
 
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
 
        return ref
 
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
 
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
 
        return ref
 
    def forward(self, x, ret_attn_mask=False):
        dtype, device = x.dtype, x.device
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
# NOTE: use padding for semantic segmentation
###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
 
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #
            
        #print("X_in")
        #print(x.shape)
        
###################################################
        #q=self.proj_q_def(x)
        x_res = rearrange(x, "n h w c -> n c h w")
#################qkv projection###################
        
        q,kv = self.qkv_conv(x.permute(0, 3, 1, 2))
        q_bi = rearrange(q, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        
        
        q_pix = rearrange(q_bi, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
 
##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe1 = self.lepe1(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
 
 
#################################################################   Offset Q
        
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_q = self.conv_offset_q(q_off).contiguous() # B * g 2 Sg HWg
        Hk, Wk = offset_q.size(2), offset_q.size(3)
        n_sample = Hk * Wk
 
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset_q = offset_q.tanh().mul(offset_range).mul(self.offset_range_factor)
 
        offset_q = rearrange(offset_q, 'b p h w -> b h w p') # B * g 2 Hg Wg -> B*g Hg Wg 2
        reference = self._get_ref_points(Hk, Wk, N, dtype, device)
 
        if self.offset_range_factor >= 0:
            pos_k = offset_q + reference
        else:
            pos_k = (offset_q + reference).clamp(-1., +1.)
 
        x_sampled_q = F.grid_sample(
            input=x_res.reshape(N * self.n_groups, self.n_group_channels, H, W),
            grid=pos_k[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
 
        q_sampled = x_sampled_q.reshape(N, C, Hk, Wk)
 
 
########　　Bi-LEVEL Gathering
 
        if self.auto_pad:
            q_sampled=q_sampled.permute(0, 2, 3, 1)
            Ng, Hg, Wg, Cg = q_sampled.size()
        
            pad_l = pad_t = 0
            pad_rg = (self.n_win - Wg % self.n_win) % self.n_win
            pad_bg = (self.n_win - Hg % self.n_win) % self.n_win
            q_sampled = F.pad(q_sampled, (0, 0, # dim=-1
                          pad_l, pad_rg, # dim=-2
                          pad_t, pad_bg)) # dim=-3
            _, Hg, Wg, _ = q_sampled.size() # padded size
            
            q_sampled=q_sampled.permute(0, 3, 1, 2)
            
            lepe1 = F.pad(lepe1.permute(0, 2, 3, 1), (0, 0, # dim=-1
                          pad_l, pad_rg, # dim=-2
                          pad_t, pad_bg)) # dim=-3
            lepe1=lepe1.permute(0, 3, 1, 2)
            
            pos_k = F.pad(pos_k, (0, 0, # dim=-1
                          pad_l, pad_rg, # dim=-2
                          pad_t, pad_bg)) # dim=-3
 
            
        queries_def = self.proj_q(q_sampled)  #Linnear projection
 
        queries_def = rearrange(queries_def, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win).contiguous()
 
        q_win, k_win = queries_def.mean([2, 3]), kv[..., 0:(self.qk_dim)].mean([2, 3])
        r_weight, r_idx = self.router(q_win, k_win)
        kv_gather = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c )
 
        k_gather, v_gather = kv_gather.split([self.qk_dim, self.dim], dim=-1)
 
        ###     Bi-level Routing MHA
        k = rearrange(k_gather, 'n p2 k hw (m c) -> (n p2) m c (k hw)', m=self.num_heads)
        v = rearrange(v_gather, 'n p2 k hw (m c) -> (n p2) m (k hw) c', m=self.num_heads)
        q_def = rearrange(queries_def,  'n p2 h w (m c)-> (n p2) m (h w) c',m=self.num_heads)
 
        attn_weight = (q_def * self.scale) @ k
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v
 
        out_def = rearrange(out, '(n j i) m (h w) c -> n (m c) (j h) (i w)', j=self.n_win, i=self.n_win, h=Hg//self.n_win, w=Wg//self.n_win).contiguous()
 
        out_def = out_def + lepe1
 
        out_def = self.unifyheads1(out_def)
        
        out_def = q_sampled + out_def
        
        out_def = out_def + self.mlp(self.norm2(out_def.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) # (N, C, H, W)
 
 
#############################################################################################
 
 
 
 
########　　　Deformable Gathering
#############################################################################################  
 
        out_def = self.norm(out_def.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        k = self.proj_k(out_def)
        v = self.proj_v(out_def)
             
        k_pix_sel = rearrange(k, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        v_pix_sel = rearrange(v, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        q_pix = rearrange(q, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        
        attn = torch.einsum('b c m, b c n -> b m n', q_pix, k_pix_sel) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        ### Bias
        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(N, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, N, dtype, device)
        displacement = (q_grid.reshape(N * self.n_groups, H * W, 2).unsqueeze(2) - pos_k.reshape(N * self.n_groups, Hg*Wg, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
                    input=rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns
        attn_bias = attn_bias.reshape(N * self.num_heads, H * W, Hg*Wg)
        attn = attn + attn_bias
        ### 
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v_pix_sel)
        out = out.reshape(N,C,H,W).contiguous()
        out = self.proj_out(out).permute(0,2,3,1)
 
#############################################################################################
        
        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
 
        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out
 
 
 
 
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')
 
 
class DeBiLevelRoutingAttention(nn.Module):
    def __init__(self, dim,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, 
                       kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, 
                       diff_routing=False, soft_routing=False, mlp_ratio=4, param_size='small',mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=True):
        super().__init__()
        qk_dim = qk_dim or dim
 
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed1 = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
            self.pos_embed2 = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
            
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
 
        self.attn1 = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=32, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
            
        self.attn2 = DeBiLevelRoutingAttentionblcok(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad,param_size=param_size)
 
 
 
    def forward(self, x):
        """
        x: NCHW tensor
        """
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
 
        x = self.attn2(x)
        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x
 
 
class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
 
    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
 
 
 
 
class DSAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, c2, kernel_size=7):  # ch_in, ch_out, kernels
        # print("### USING DSAM FILE: xxx.py 2026-01-07 v3 ###")
        super().__init__()
        # print('DSAM init with c1=c2=', c1, ' ', c2)
        assert c1 == c2
        
        # print('kernel size bf: ', kernel_size)
        self.channel_attention = DeBiLevelRoutingAttention(c1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # print('kernel size af: ', kernel_size)
        # kernel_size = 7
        # self.spatial_attention = SpatialAttention(kernel_size)
 
    def forward(self, x):
        # return self.spatial_attention(self.channel_attention(x))
        return x + self.gamma * self.channel_attention(x)
        # print('DSAM forward with x.shape=', x.shape)
        # y = self.channel_attention(x)
        # # y= x
        # # print('DSAM forward with y.shape=', y.shape)

        # # 1) 确保输出 shape 和 x 完全一致（很多 attention 会输出 NHWC）
        # if y.shape != x.shape:
        #     # 常见情况：y 是 [B,H,W,C]，x 是 [B,C,H,W]
        #     if y.ndim == 4 and y.shape[0] == x.shape[0] and y.shape[-1] == x.shape[1]:
        #         y = y.permute(0, 3, 1, 2).contiguous()

        # # 2) 关键：把 NaN/Inf 变成 0，避免 0*NaN = NaN
        # y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # return x

class DropPath(nn.Module):
    """Stochastic Depth. Very small implementation to avoid external deps."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask


class CoordAtt(nn.Module):
    """Coordinate Attention (CA) - lightweight and localization-friendly."""
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, channels, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)                      # [B,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B,C,W,1]
        y = torch.cat([x_h, x_w], dim=2)          # [B,C,H+W,1]
        y = self.act(self.bn1(self.conv1(y)))     # [B,mip,H+W,1]
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)             # [B,mip,1,W]
        a_h = self.sigmoid(self.conv_h(y_h))      # [B,C,H,1]
        a_w = self.sigmoid(self.conv_w(y_w))      # [B,C,1,W]
        return x * a_h * a_w


class DSAMPlus(nn.Module):
    """
    DSAM++ for OKS: global attention + local detail branch + coord attention + gated fusion.
    Keeps: y = x + gamma * F(x)
    """
    def __init__(
        self,
        c1, c2,
        drop_path=0.05,
        use_ca=True,
        ca_reduction=32,
        use_highpass=True,
        gate_channels=1,
    ):
        super().__init__()
        assert c1 == c2
        c = c1

        # Your original global attention
        self.global_attn = DeBiLevelRoutingAttention(c)

        # Local detail branch (depthwise separable conv)
        self.local = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
        )

        # Coordinate attention (more helpful for keypoint localization than pure channel attention)
        self.ca = CoordAtt(c, reduction=ca_reduction) if use_ca else nn.Identity()

        # Optional high-pass residual (great for faint landmarks in X-ray)
        self.use_highpass = use_highpass
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Gate: decide global vs local
        out_ch = gate_channels  # 1 is usually more stable than C
        self.gate = nn.Sequential(
            nn.Conv2d(c, out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

        self.drop_path = DropPath(drop_path)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # High-pass component (x - blur(x))
        if self.use_highpass:
            hp = x - self.avg_pool(x)
        else:
            hp = 0.0

        g = self.global_attn(x)          # global/context
        l = self.local(x + hp)           # local/detail (bias to edges)

        # Gate (broadcast if 1-channel)
        a = self.gate(x)
        if a.shape[1] == 1:
            a = a.expand_as(g)

        fused = a * g + (1.0 - a) * l
        fused = self.ca(fused)           # coordinate-aware refinement
        fused = self.drop_path(fused)

        return x + self.gamma * fused
    
class PSABlock_DSAM(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.
    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.
    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.
    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
 
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()
 
        #self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.attn = DeBiLevelRoutingAttention(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        # x = x + self.attn(x) if self.add else self.attn(x)
        # x = x + self.ffn(x) if self.add else self.ffn(x)
        # return x
        x = x + self.gamma1*self.attn(self.norm1(x)) if self.add else self.attn(self.norm1(x))
        x = x + self.gamma2*self.ffn(self.norm2(x)) if self.add else self.ffn(self.norm2(x))
        return x
 
class C2PSA_DSAM(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.
    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """
 
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
 
        self.m = nn.Sequential(*(PSABlock_DSAM(self.c) for _ in range(n)))
 
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))