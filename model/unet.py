import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class UNet1DEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(UNet1DEncoder, self).__init__()

        self.use_unet_residual = Config.use_unet_residual

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.down1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, Config.l2_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.down2 = nn.Conv1d(Config.l2_dim, Config.l2_dim, kernel_size=3, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.Conv1d(Config.l2_dim, Config.l1_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.norm3 = nn.LayerNorm(Config.l1_dim)

        if self.use_unet_residual:
            self.res1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.res2 = nn.Conv1d(hidden_dim, Config.l2_dim, kernel_size=1)
            self.norm2 = nn.LayerNorm(Config.l2_dim)
        else:
            self.res1 = None
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.res2 = None
            self.norm2 = nn.LayerNorm(Config.l2_dim)

    def forward(self, x):

        # Input x: [B, L, d] → [B, d, L]
        x = x.permute(0, 2, 1)

        x1 = self.enc1(x)               # [B, h, L]
        x1 = F.max_pool1d(x1, kernel_size=3, stride=1, padding=1)
        if self.use_unet_residual:
            x1 = x1 + self.res1(x)
        x1 = self.norm1(x1.permute(0, 2, 1)).permute(0, 2, 1)


        x_down1 = self.down1(x1)
        x2 = self.enc2(x_down1)  # [B, h, L/2]
        x2 = F.max_pool1d(x2, kernel_size=3, stride=1, padding=1)
        if self.use_unet_residual:
            x2 = x2 + self.res2(x_down1)
        x2 = self.norm2(x2.permute(0, 2, 1)).permute(0, 2, 1)
        x3 = self.enc3(self.down2(x2))  # [B, h, L/4]
        x3 = F.max_pool1d(x3, kernel_size=3, stride=1, padding=1)
        x3 = self.norm3(x3.permute(0, 2, 1)).permute(0, 2, 1)

        # Convert back to [B, L, h]
        x1 = x1.permute(0, 2, 1)  # [B, L, h]
        x2 = x2.permute(0, 2, 1)  # [B, L/2, h]
        x3 = x3.permute(0, 2, 1)  # [B, L/4, h]

        return x1, x2, x3



class AttentionUpsampler(nn.Module):
    def __init__(self, nheads, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.scale = 2
        # No need to define deconv layer anymore.

    def forward(self, attn, target_len=None):
        """
        attn: [B, H, L_in, L_in]
        returns: [B, H, L_out, L_out]
        """
        if target_len is None:
            target_len = attn.size(-1) * self.scale

        attn_up = F.interpolate(
            attn,
            size=(target_len, target_len),
            mode='bilinear',
            align_corners=False
        )

        if self.normalize:
            attn_up = attn_up / attn_up.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return attn_up