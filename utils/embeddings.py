import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
# import datetime
from config import Config

class TemporalEmbedding(nn.Module):
    """
    week, day, hour, ...
    """
    def __init__(self, dim, freq='t'):
        super(TemporalEmbedding, self).__init__()

        second_size = 60
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # Embed = nn.Embedding
        if freq == 't':
            self.second_embed = nn.Embedding(second_size, dim)
            self.minute_embed = nn.Embedding(minute_size, dim)
        self.hour_embed = nn.Embedding(hour_size, dim)
        self.weekday_embed = nn.Embedding(weekday_size, dim)
        self.day_embed = nn.Embedding(day_size, dim)
        self.month_embed = nn.Embedding(month_size, dim)


    def forward(self, x):
        x = x.long()
        # print(x.shape)
        assert x.dim() == 3 and x.size(2) == 6, "Input tensor must be [batch, seq_len, 6]"
        # print(x[:, :, 0:5])
        # Extract and embed each time component
        month_x = self.month_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        hour_x = self.hour_embed(x[:, :, 3])
        minute_x = self.minute_embed(x[:, :, 4]) if self.minute_embed is not None else 0
        # print(1)
        # TODO: maybe add seconds?
        second_x = self.second_embed(x[:, :, 5]) if self.second_embed is not None else 0

        # needs to be added to regular PEs.
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x


class POIEncoder(nn.Module):
    def __init__(self, num_poi_types, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_poi_types, output_dim)

    def forward(self, poi_counts):
        """
        poi_counts: Tensor [B, L, num_poi_types] — sparse multi-hot (or multi-count) encoding
        Returns:
            Tensor [B, L, output_dim]
        """
        # [B, L, num_poi_types] x [num_poi_types, output_dim] → [B, L, output_dim]
        return torch.matmul(poi_counts, self.embedding.weight)

class RelativeFourierPE(nn.Module):
    def __init__(self, base_dim=256, num_freqs=6):
        """
        Fourier-encode relative distance and angle, then fuse with main features.

        Args:
            base_dim (int): Dimension of your main input (e.g., 256)
            num_freqs (int): Number of Fourier frequency bands.
            include_input (bool): Whether to include raw distance/angle.
            fusion (str): "concat" or "add"
        """
        super().__init__()
        self.num_freqs = num_freqs

        # Calculate final PE dimension
        pe_dim =  2 * num_freqs
        self.pe_dim = pe_dim  # for [distance, angle]
        self.pe_proj = nn.Linear(self.pe_dim, base_dim)
        #
        # # Projection layer if fusing with concat
        # if fusion == "concat":
        #     self.fuse_proj = nn.Linear(base_dim + self.pe_dim, base_dim)
        # elif fusion == "add":
        #     self.pe_proj = nn.Linear(self.pe_dim, base_dim)
        # else:
        #     raise ValueError("fusion must be 'concat' or 'add'")

    def fourier_encode(self, x):
        x = x.unsqueeze(-1)  # [B, L, 1]
        freqs = 2 ** torch.arange(self.num_freqs, dtype=torch.float32, device=x.device) * math.pi  # [num_freqs]
        x_proj = x * freqs  # [B, L, num_freqs]

        sin = torch.sin(x_proj)
        cos = torch.cos(x_proj)
        out = torch.cat([sin, cos], dim=-1)  # [B, L, 2*num_freqs]

        if self.include_input:
            out = torch.cat([x, out], dim=-1)  # [B, L, 1 + 2*num_freqs]

        return out

    def forward(self, rel_feat):
        # distance = rel_feat[:, :, 0]
        angle = rel_feat[:, :]

        # pe_dist = self.fourier_encode(distance)  # [B, L, D]
        pe_angle = self.fourier_encode(angle)    # [B, L, D]
        # pe = torch.cat([pe_dist, pe_angle], dim=-1)  # [B, L, 2D]
        pe = self.pe_proj(pe_angle)

        return pe


class UserEmbedding(nn.Module):
    def __init__(self, num_users=100000, embedding_dim=256):
        super().__init__()
        self.num_users = num_users
        self.embedding_dim = embedding_dim

        # Create an embedding table of shape [num_users, embedding_dim].
        # Each row i corresponds to user i.
        self.embedding_layer = nn.Embedding(num_users, embedding_dim)

        # Optional: Initialize embeddings (e.g., xavier_uniform_).
        nn.init.xavier_uniform_(self.embedding_layer.weight)

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:

        return self.embedding_layer(user_ids)

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """
        Retrieve the embedding for a single user ID, returned as a 1D tensor.

        Args:
            user_id (int): The user index (0 <= user_id < num_users).

        Returns:
            torch.Tensor: A tensor of shape [embedding_dim] representing the user embedding.
        """
        user_id_tensor = torch.tensor([user_id], dtype=torch.long)
        return self.embedding_layer(user_id_tensor).squeeze(0)


class HuMobTemporalEmbedding(nn.Module):
    """
    day, hour.
    """
    def __init__(self, dim):
        super(HuMobTemporalEmbedding, self).__init__()

        hour_size = 48
        day_size = 75

        self.hour_embed = nn.Embedding(hour_size, dim)
        self.day_embed = nn.Embedding(day_size, dim)

    def forward(self, x):
        x = x.long()

        # Extract and embed each time component
        day_x = self.day_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])

        # needs to be added to regular PEs.
        return hour_x + day_x




# class LearnableFourierPositionalEncoding(nn.Module):
#     def __init__(self, M: int, F_dim: int, D: int, gamma=1.0):
#         """
#         :param M: each point has a M-dimensional positional values
#         :param F_dim: depth of the Fourier feature dimension
#         :param H_dim: hidden layer dimension
#         :param D: positional encoding dimension
#         :param gamma: parameter to initialize Wr
#         """
#         super().__init__()
#         self.M = M
#         self.F_dim = F_dim
#         self.H_dim = D
#         self.D = D
#         self.gamma = gamma
#
#         # Projection matrix on learned lines (used in eq. 2)
#         self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
#         # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.F_dim, self.H_dim, bias=True),
#             nn.GELU(),
#             nn.Linear(self.H_dim, self.D)
#         )
#
#         self.init_weights()
#
#     def init_weights(self):
#         nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)
#
#     def forward(self, x):
#         """
#         Produce positional encodings from x
#         :param x: tensor of shape [B, N, M] that represents N positions where each position is in the shape of M with batch size B,
#         :return: positional encoding for X [B, N, D]
#         """
#         B, N, M = x.shape
#         # Step 1. Compute Fourier features (eq. 2)
#         projected = self.Wr(x)
#         cosines = torch.cos(projected)
#         sines = torch.sin(projected)
#         F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
#         # Step 2. Compute projected Fourier features (eq. 6)
#         Y = self.mlp(F)
#         # Step 3. Reshape to x's shape
#         pos_enc = Y.reshape((B, N, self.D))
#         return pos_enc
#
#

