import sys

sys.path.append('..')


import math
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils.embeddings import TemporalEmbedding, HuMobTemporalEmbedding, UserEmbedding, RelativeFourierPE

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    if tensor is not None:
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    return None

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:  # Ensure weight exists before initializing
            trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:  # Ensure bias exists before initializing
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1.0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.gamma_param = nn.Parameter(torch.full((nhead,), -1.0))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, prev_attn=None, blend_gamma=None, return_attn=False):
        B, L, _ = x.shape

        if not Config.use_attention:
            if L == 0:
                fused = torch.zeros(B, self.nhead, L, L, device=x.device, dtype=x.dtype)
            else:
                fused = torch.full(
                    (B, self.nhead, L, L),
                    1.0 / L,
                    device=x.device,
                    dtype=x.dtype
                )
            if src_key_padding_mask is not None:
                pad_row = src_key_padding_mask.unsqueeze(1).unsqueeze(-1)
                pad_col = src_key_padding_mask.unsqueeze(1).unsqueeze(-2)
                fused = fused.masked_fill(pad_row | pad_col, 0.0)
                fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn_output = torch.zeros_like(x)
            self.last_attn_weights = fused
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self.ffn(x))
            if return_attn:
                return x, fused
            return x

        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if src_mask is not None:
            if src_mask.dim() == 2:
                logits = logits + src_mask.unsqueeze(0).unsqueeze(0)
            else:
                logits = logits + src_mask
        mask = None
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(mask, float('-inf'))

        current_attn = torch.softmax(logits, dim=-1)
        if not Config.use_attention:
            prev_attn = None

        if prev_attn is not None:
            prev = prev_attn
            if prev.dim() == 3:
                prev = prev.unsqueeze(0)
            if prev.shape != current_attn.shape:
                raise ValueError(f"prev_attn shape {prev.shape} does not match current attention shape {current_attn.shape}")

            gamma = torch.sigmoid(self.gamma_param).view(1, self.nhead, 1, 1)
            fused = (1.0 - gamma) * current_attn + gamma * prev
            if mask is not None:
                fused = fused.masked_fill(mask, 0.0)
            fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            fused = current_attn

        attn_weights = F.dropout(fused, p=self.self_attn.dropout if isinstance(self.self_attn.dropout, float) else self.self_attn.dropout.p, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        attn_output = self.self_attn.out_proj(attn_output)

        self.last_attn_weights = fused

        x = self.norm1(x + attn_output)
        # Feedforward
        x = self.norm2(x + self.ffn(x))
        if return_attn:
            return x, fused
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.last_attn_weights = None  # Exposed for downstream use

    def forward(self, x, src_mask=None, src_key_padding_mask=None, prev_attn=None, blend_gamma=None):
        override = prev_attn
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x, attn_weights = layer(x, src_mask, src_key_padding_mask, prev_attn=override, blend_gamma=blend_gamma, return_attn=True)
                self.last_attn_weights = attn_weights
            else:
                x = layer(x, src_mask, src_key_padding_mask, prev_attn=override, blend_gamma=blend_gamma)
                override = layer.last_attn_weights.detach()
        return x


class Model(nn.Module):
    """
    level 1: Spatiotemporal features.
             transformer encoder for location sequence.

    level 2: Semantic / behavioral features.
             transformer encoder for POI sequences and time.

    level 3: TBD.

    """

    def __init__(self, Config, input_dim, hidden_dim, num_layers=3, nheads=8,  maxlen: int = 201):
        super(Model, self).__init__()
        self.nheads = nheads
        self.input_dim = input_dim
        self.loc_emb = nn.Embedding(400001,input_dim)
        # location sequence PE.
        self.pe = LearnablePositionalEncoding(input_dim, 0.1, maxlen)
        # time info.
        self.tempe = TemporalEmbedding(input_dim)
        self.time_down = nn.AvgPool1d(kernel_size=2, stride=2)

        self.m_pe = RelativeFourierPE(input_dim)
        self.m_fuse = nn.Linear(input_dim*2, input_dim)
        # time info. for humob dataset.
        self.tem_humob_pe = HuMobTemporalEmbedding(input_dim)
        # user embedding.
        self.upe = UserEmbedding(embedding_dim=input_dim)

        self.max_length = maxlen

        """
        customised Transformer Encoder.
        """
        self.encoder = TransformerEncoder(d_model=input_dim, nhead=nheads, num_layers=num_layers,
                                          dim_feedforward=hidden_dim)
        """
        transformer
        """

    def forward(self, cell_emb, t, src_padding_mask, adj, indicator, poi_seq, motion, prev_attn):
        B, L, d = cell_emb.shape



        if indicator == 3 and motion is not None:
            motion_pe = self.m_pe(motion)
            # cell_emb  = cell_emb + motion_pe
            cell_emb = torch.cat((cell_emb, motion_pe), dim=-1)
            cell_emb = self.m_fuse(cell_emb)

        # TODO: also try time2vec for time embedding.
        if indicator == 3 and t is not None:
            # print("using temporal embedding...")
            if not Config.humob:
                tpe = self.tempe(t)
                cell_emb += tpe

            else:
                tpe = self.tem_humob_pe(t)
                cell_emb += tpe

        if not Config.long:
            cell_emb = self.pe(cell_emb)
        else:
            pe_long = PositionalEncoding(emb_size= self.input_dim, dropout=0.1, maxlen=3000, learnable=False)
            cell_emb = pe_long(cell_emb)
        src_padding_mask = src_padding_mask.to(cell_emb.device)

        """
        use upsampled attention to the next layer.
        """


        if prev_attn is not None:
            B, n, L_attn, _ = prev_attn.shape
            if L_attn != L:
                prev_attn = F.pad(prev_attn, (0, L - L_attn, 0, L - L_attn)) if L_attn < L else prev_attn[:, :, :L, :L]
        
        traj_emb = self.encoder(cell_emb, None, src_key_padding_mask=src_padding_mask, prev_attn=prev_attn, blend_gamma=None)
        fused_attn = self.encoder.last_attn_weights

        mask = 1 - src_padding_mask.unsqueeze(-1).expand(traj_emb.shape).float() 
        traj_emb = mask * traj_emb
        return traj_emb, fused_attn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(LearnablePositionalEncoding, self).__init__()
        # Instead of fixed position encoding, we use a learnable parameter
        self.pos_embedding = nn.Parameter(torch.zeros(maxlen, emb_size))
        # Initialize the positional encoding parameter
        nn.init.uniform_(self.pos_embedding, -0.1, 0.1)  # You can experiment with different initialization strategies

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: torch.Tensor):
        """
        Args:
            token_embedding: Tensor of shape [batch_size, seq_len, emb_size]
            temporal_embedding: Tensor of shape [batch_size, seq_len, emb_ size]
                - Temporal embeddings aligned with the token embeddings.
        """
        # Assume token_embedding is of shape [batch_size, seq_len, emb_size]
        # We need to expand position encoding to match the batch size
        batch_size, seq_len, _ = token_embedding.size()

        # Expand the position embedding to the whole batch
        pos_embedding = self.pos_embedding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1).to(token_embedding.device)
        # print(pos_embedding.shape, temporal_embedding.shape)
        # Add the learnable position encoding to the input token embeddings along with temporal embeddings
        combined_embeddings = token_embedding + pos_embedding

        return combined_embeddings

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 3000,
        learnable: bool = False
    ):
        """
        Args:
            emb_size:      embedding dimension
            dropout:       dropout probability
            maxlen:        maximum supported sequence length
            learnable:     if True, position embeddings are a learned parameter;
                           otherwise fixed sin/cos are used.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if learnable:
            # [1, maxlen, emb_size] so it broadcasts over batch
            self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size))
            nn.init.uniform_(self.pos_embedding, -0.1, 0.1)
        else:
            # create fixed sin/cos table
            den = torch.exp(torch.arange(0, emb_size, 2) *
                            (-math.log(10000) / emb_size))
            pos = torch.arange(0, maxlen).unsqueeze(1)            # [maxlen,1]
            pe = torch.zeros(maxlen, emb_size)
            pe[:, 0::2] = torch.sin(pos * den)
            pe[:, 1::2] = torch.cos(pos * den)
            pe = pe.unsqueeze(0)                                  # [1,maxlen,emb_size]
            self.register_buffer('pos_embedding', pe)

    def forward(self, token_embedding: torch.Tensor):
        """
        token_embedding: [batch_size, seq_len, emb_size]
        returns:          [batch_size, seq_len, emb_size] + dropout
        """
        seq_len = token_embedding.size(1)
        # slice to the actual length, shape [1, seq_len, emb_size]
        pos = self.pos_embedding[:, :seq_len, :].to(token_embedding.device)
        # broadcast add, then dropout
        return self.dropout(token_embedding + pos)
