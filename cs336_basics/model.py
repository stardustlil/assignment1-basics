import torch.nn as nn
import math
import torch
from einops import rearrange



class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, device = None, dtype = None):
        super().__init__()
        std = math.sqrt(2 / (in_dim + out_dim))
        bound = 3 * std
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_dim, in_dim , dtype=dtype, device= device) , mean = 0 , std = std , a = -bound , b = bound)
        )

    def forward(self , x:torch.Tensor) -> torch.Tensor:
        return torch.einsum("...a,...ba->...b", x , self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model , device = device, dtype = dtype), mean = 0 , std = 1, a = -3, b = 3)
        )

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device=None, dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        olddtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
        return (self.weight * x / rms).to(olddtype)


def silu(x:torch.Tensor):
    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor):
        return self.w2(silu(self.w1(x))*self.w3(x))

def softmax(x:torch.Tensor , dim:int)->torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max) #数值稳定性
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class RoPE(nn.Module):
    cos:torch.Tensor
    sin:torch.Tensor
    def __init__(self, d_k:int, theta:float, max_seq_len:int, device = None):
        super().__init__()
        inv_freq = theta **(-torch.arange(0, d_k, 2 , device=device).float() / d_k) # (d_k / 2, )
        pos = torch.arange(max_seq_len, device=device).float() #(max_seq_len, )
        angles = pos[:, None] * inv_freq[None, :]  #(max_seq_len, 1) * (1, d_k/2) #(max_seq_len, d_k/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        # x:(..., seq_len, d_k), token_positions:(..., seq_len)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1*cos - x2*sin
        out[..., 1::2] = x1*sin + x2*cos
        return out        


def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask = None) -> torch.Tensor:
    d_k = Q.size(-1)
    # Q(..., queries, d_k) , K(..., keys, d_k), V(..., keys, d_v), mask(..., queries, keys)
    scores = torch.einsum("...qa,...ka->...qk", Q, K)
    scores = scores/math.sqrt(d_k)    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn   = softmax(scores, dim = -1) 
    out = torch.einsum("...qk,...kd->...qd",attn,V)
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        assert num_heads > 0, "num_heads must be positive"
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model,device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model,device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model,device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model,device=device, dtype=dtype)
        self.rope = None
        if max_seq_len is not None:
            assert theta is not None, "theta must be provided when max_seq_len is set"
            self.rope = RoPE(self.head_dim, theta, max_seq_len, device = device)

    def _split_heads(self, x:torch.Tensor) -> torch.Tensor:
        return rearrange(x, '... s (h d) -> ... h s d' , h = self.num_heads)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, '... h s d -> ... s (h d)')
    
    def forward(self, x:torch.Tensor, token_positions = None):
        seq = x.shape[-2]
        q = self._split_heads(self.q_proj(x))   # (b, h, s, d)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k , v, causal)
        out = self._merge_heads(out)
        return self.output_proj(out)


