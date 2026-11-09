from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers import Decoder

from config import Config
from jepa.components import (
    EncoderOutput,
    JEPA_Encoder,
    JEPA_Head,
    JEPA_Hierarchy,
    JEPA_Loss,
    JEPA_Mask,
    Level,
)

##########################################################
#  Level generating functions
##########################################################


class UNet1DEncoder(nn.Module):
    """
    A 3-level 1D Convolutional Network.
    It processes a sequence and extracts features at three different temporal resolutions:
    Level 1: Full sequence length (L)
    Level 2: Half sequence length (L/2)
    Level 3: Quarter sequence length (L/4)
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        # ==========================================
        # HOW THE RESIDUAL CONNECTION WORKS (POINTWISE CONV)
        # ==========================================
        # The goal of a residual connection is to add the original input directly to the
        # processed output (i.e., Output = Processed(x) + x) to help gradients flow.
        #
        # However, the main encoder convolutions change the number of channels
        # (e.g., from `in_channels` to `hidden_dim`). Because the channel counts
        # no longer match, we cannot mathematically add the two tensors together.
        #
        # To solve this, the residual layers (self.res1, self.res2) use a 1x1 convolution
        # (kernel_size=1). This acts as a simple linear projection on every time step
        # independently, without looking at neighboring sequence points. It simply
        # projects the original input into the new channel dimension so the shapes
        # match perfectly, allowing the element-wise addition: x1 = x1 + self.res1(x).
        # ==========================================
        self.use_unet_residual = Config.use_unet_residual

        # --- LEVEL 1: Finest resolution (Length = L) ---
        # ==========================================
        # DECOUPLED FEATURE EXTRACTION & DOWNSAMPLING
        # ==========================================
        # The network explicitly separates feature extraction and spatial downsampling
        # into two different convolutional steps:
        #
        # 1. The Encoder Convs (e.g., self.enc1, self.enc2):
        #    - Function: Feature extraction and channel projection.
        #    - Mechanism: Uses stride=1 and padding=1.
        #    - Result: Changes the channel dimension (e.g., in_channels -> hidden_dim)
        #      but keeps the sequence length (L) exactly the same.
        #
        # 2. The Downsampling Convs (e.g., self.down1, self.down2):
        #    - Function: Timeline compression.
        #    - Mechanism: Uses stride=2.
        #    - Result: Keeps the channel dimension exactly the same, but cuts the
        #      sequence length in half (L -> L/2).
        # ==========================================
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Downsample to transition from Level 1 to Level 2 (stride=2 halves the sequence length)
        self.down1 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
        )

        # --- LEVEL 2: Medium resolution (Length = L/2) ---
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, Config.l2_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Downsample to transition from Level 2 to Level 3
        self.down2 = nn.Conv1d(
            Config.l2_dim, Config.l2_dim, kernel_size=3, stride=2, padding=1
        )

        # --- LEVEL 3: Coarsest resolution (Length = L/4) ---
        self.enc3 = nn.Sequential(
            nn.Conv1d(Config.l2_dim, Config.l1_dim, kernel_size=3, padding=1), nn.ReLU()
        )

        # LayerNorm applied to the feature dimension of the final level
        self.norm3 = nn.LayerNorm(Config.l1_dim)

        # Optional Residual connections: 1x1 convolutions to match channel dimensions
        # so the original input can be added directly to the processed output.
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
        # PyTorch Conv1d expects shape [Batch, Channels, Length]
        # Transformers usually output [Batch, Length, Dimension]
        # So we permute the Length and Dimension axes.
        x = x.permute(0, 2, 1)

        # --- PROCESS LEVEL 1 ---
        x1 = self.enc1(x)  # Shape: [B, hidden_dim, L]
        x1 = F.max_pool1d(x1, kernel_size=3, stride=1, padding=1)
        if self.use_unet_residual:
            x1 = x1 + self.res1(x)  # Add skip connection

        # LayerNorm expects the feature dimension to be last.
        # So we permute to [B, L, hidden_dim], apply norm, and permute back.
        x1 = self.norm1(x1.permute(0, 2, 1)).permute(0, 2, 1)

        # --- PROCESS LEVEL 2 ---
        x_down1 = self.down1(x1)  # Shape: [B, hidden_dim, L/2]
        x2 = self.enc2(x_down1)  # Shape: [B, l2_dim, L/2]
        x2 = F.max_pool1d(x2, kernel_size=3, stride=1, padding=1)
        if self.use_unet_residual:
            x2 = x2 + self.res2(x_down1)
        x2 = self.norm2(x2.permute(0, 2, 1)).permute(0, 2, 1)

        # --- PROCESS LEVEL 3 ---
        x3 = self.enc3(self.down2(x2))  # Shape: [B, l1_dim, L/4]
        x3 = F.max_pool1d(x3, kernel_size=3, stride=1, padding=1)
        x3 = self.norm3(x3.permute(0, 2, 1)).permute(0, 2, 1)

        # Convert all outputs back to Transformer-friendly shapes: [Batch, Length, Channels]
        x1 = x1.permute(0, 2, 1)  # [B, L, hidden_dim]
        x2 = x2.permute(0, 2, 1)  # [B, L/2, l2_dim]
        x3 = x3.permute(0, 2, 1)  # [B, L/4, l1_dim]

        # Returns coarsest (shortest) to finest (longest)
        return x3, x2, x1


class UNetHierarchy(JEPA_Hierarchy):
    """
    Bridges the UNet1DEncoder to the JEPA framework.
    It reverses the UNet outputs to match JEPA's Coarse->Fine requirement
    and calculates proper padding masks for the downsampled sequences.
    """

    def __init__(self, fpn):
        super().__init__()
        self.fpn = fpn  # This is the UNet1DEncoder instance

    def forward(self, x, lengths=None):
        L_in = x.size(1)  # Original sequence length
        levels = []

        for feat in self.fpn(x):
            L_i = feat.size(1)  # Length of the current hierarchical level (e.g., L/4)
            mask = None

            if lengths is not None:
                # ==========================================
                # DOWNSAMPLED PADDING MASK CALCULATION
                # ==========================================
                # 1. Calculate Stride: -(-L_in // L_i) is a fast Python trick for integer
                #    ceiling division. It dynamically deduces the downsampling factor
                #    (e.g., L_in=30 dropping to L_i=15 means the stride was 2).
                # 2. Scale Valid Lengths: We take the original, unpadded sequence lengths
                #    of the batch and scale them down by that exact same stride to find
                #    the new valid data boundaries for this downsampled level.
                # 3. Generate Mask: We create a sequence of indices [0, 1, 2...] and
                #    mark any index >= the new valid length as True (meaning it is padding).
                # ==========================================
                stride = -(-L_in // L_i)

                # Downsample the valid.
                lvl_len = torch.div(lengths + stride - 1, stride, rounding_mode="trunc")

                # Create a boolean mask: True for padded positions, False for valid data.
                # Compares a sequence of numbers [0, 1, 2...] against the valid length limit.
                mask = (
                    torch.arange(L_i, device=feat.device)[None, :] >= lvl_len[:, None]
                )

            # Pack the feature tensor and its corresponding mask into the JEPA Level dataclass
            levels.append(Level(feat, mask))

        return levels


############################################################################################################
# Mask
############################################################################################################


class JEPA_HMTraj_Mask(JEPA_Mask):
    def __init__(
        self, sample_number=4, mask_ratio=(0.2, 0.25), succ_prob=0.5, context_lb=0.85
    ):
        super().__init__()
        self.sample_number = sample_number
        self.mask_ratio = mask_ratio
        self.succ_prob = succ_prob
        self.context_lb = context_lb

    @staticmethod
    def mask_tensor(x, src_padding_mask, mask_ratio, succ_prob, mode):
        """
        Mask a tensor with a given mask ratio, considering the padding mask and
        a probability to apply successive masking.
        This function is optimized to use matrix operations for efficiency.

        Args:
        - x (torch.Tensor): input embedding with shape [B, L, d].
        - src_padding_mask (torch.Tensor): Padding mask with shape [B, L], True for padded positions.
        - mask_ratio (float): The ratio of len to be masked.
        - succ_prob (float): The probability of applying successive masking.
        - mode (str): 'resample', 'waypoints', or 'endpoint'.

        Returns:
        - torch.Tensor: masked tensor.
        - torch.Tensor: boolean mask.
        - list: list of masked indices for each batch item.
        """
        B, L, d = x.shape
        device = x.device

        if mode not in ["resample", "waypoints"]:
            # For modes other than 'resample' or 'waypoints', return unmasked tensor
            mask = torch.zeros((B, L), dtype=torch.bool, device=device)
            return x * mask.unsqueeze(-1).to(x.dtype), mask, None

        non_padded_len = L - src_padding_mask.sum(dim=1)
        num_mask = (non_padded_len * mask_ratio).long()

        # Create a default mask for positions with no masking needed
        final_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        # only process where num_mask > 0
        has_mask = num_mask > 0
        if not has_mask.any():
            return x * final_mask.unsqueeze(-1).to(x.dtype), final_mask, None

        # Decide between successive and random masking for the entire batch
        succ_mask_decision = torch.rand(B, device=device) < succ_prob

        # --- Successive Masking ---
        succ_indices = torch.where(succ_mask_decision & has_mask)[0]
        if len(succ_indices) > 0:
            succ_non_padded_len = non_padded_len[succ_indices]
            succ_num_mask = num_mask[succ_indices]
            max_start = succ_non_padded_len - succ_num_mask
            starts = (
                torch.rand(len(succ_indices), device=device) * max_start.float()
            ).long()

            arange_l = (
                torch.arange(L, device=device)
                .unsqueeze(0)
                .expand(len(succ_indices), -1)
            )
            starts_exp = starts.unsqueeze(1)
            ends_exp = (starts + succ_num_mask).unsqueeze(1)

            successive_mask_for_succ = (arange_l >= starts_exp) & (arange_l < ends_exp)
            final_mask[succ_indices] = successive_mask_for_succ

        # --- Random Masking ---
        rand_indices = torch.where(~succ_mask_decision & has_mask)[0]
        if len(rand_indices) > 0:
            # rand_non_padded_len = non_padded_len[rand_indices]
            rand_num_mask = num_mask[rand_indices]

            # Generate random noise and suppress padded positions
            noise = torch.rand(len(rand_indices), L, device=device)
            noise.masked_fill_(src_padding_mask[rand_indices], -1.0)

            max_k = rand_num_mask.max().item()
            if max_k > 0:
                # Use torch.topk which is more efficient than argsort for finding a subset of indices,
                # especially when k is much smaller than L.
                _, topk_indices = torch.topk(noise, k=max_k, dim=1)

                # Create a mask to select top `num_mask` indices for each item
                arange_k = (
                    torch.arange(max_k, device=device)
                    .unsqueeze(0)
                    .expand(len(rand_indices), -1)
                )
                selection_k_mask = arange_k < rand_num_mask.unsqueeze(1)

                indices_to_mask = topk_indices[selection_k_mask]
                batch_idx_for_scatter = (
                    torch.arange(len(rand_indices), device=device)
                    .unsqueeze(1)
                    .expand(-1, max_k)[selection_k_mask]
                )

                random_mask_for_rand = torch.zeros(
                    len(rand_indices), L, dtype=torch.bool, device=device
                )
                random_mask_for_rand[batch_idx_for_scatter, indices_to_mask] = True
                final_mask[rand_indices] = random_mask_for_rand

        masked_x = x * final_mask.unsqueeze(-1).to(x.dtype)

        # The mask_indices are not used, so we can avoid computing it and just return None.
        return masked_x, final_mask, None
        # return mask_indices     # [B, len]

    def target_view(self, level):
        """The Level the TARGET encoder sees: same input, but never the timestamps."""
        return replace(level, side={**level.side, "t": None})

    @torch.no_grad()
    def generate_target(self, x_emb, src_padding_mask, indicator):
        """
        Generates M masked variations of the target embeddings.
        Expects x_emb to already be processed by the target encoder.
        """
        B, L, d = x_emb.shape
        if src_padding_mask is None:  # no padding information: nothing is padded
            src_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=x_emb.device)

        # Randomly select a mask ratio from the tuple
        idx = torch.randint(low=0, high=len(self.mask_ratio), size=(1,)).item()
        r = self.mask_ratio[idx]

        # Determine mode based on hierarchical level
        mode = "waypoints" if indicator in [0, 1] else "resample"

        # Repeat input for M samples (using self.sample_number, which maps to M)
        M = self.sample_number
        x_emb_repeated = x_emb.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, L, d)
        padding_repeated = (
            src_padding_mask.unsqueeze(0).expand(M, -1, -1).reshape(M * B, L)
        )

        # Get masks for all M*B samples at once
        masked_x, masks, _ = self.mask_tensor(
            x_emb_repeated, padding_repeated, r, self.succ_prob, mode
        )

        # Reshape back to (M, B, L, d) and (M, B, L)
        all_mask_x = masked_x.view(M, B, L, d)
        all_masks = masks.view(M, B, L)

        return all_mask_x, all_masks

    def generate_context(self, level, target_masks, indicator):
        """
        Samples the context sequence: random positions that do NOT overlap with any
        of the M target masks. Returns the Level the CONTEXT encoder will see.

        level:        Level with .x [B, L, d], .padding_mask [B, L] or None, .side dict
        target_masks: [M, B, L] bool, True = position is a target
        """
        x, padding_mask = level.x, level.padding_mask
        B, L, d = x.shape
        device = x.device
        if padding_mask is None:  # no padding information: nothing is padded
            padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        # 1. Determine number of samples for each item in the batch
        ratios = torch.rand((B,), device=device).clamp(min=self.context_lb, max=1.0)
        non_padded_lens = L - padding_mask.sum(dim=1)
        num_samples = (non_padded_lens.float() * ratios).long()

        # 2. Vectorized random sampling to create initial sample_mask
        noise = torch.rand(B, L, device=device)
        noise.masked_fill_(padding_mask, -1.0)  # Padded positions can't be sampled

        max_k = num_samples.max().item()
        sample_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        if max_k > 0:
            # Use topk to find indices for sampling
            _, topk_indices = torch.topk(noise, k=max_k, dim=1)

            # Create a selection mask for variable number of samples
            arange_k = torch.arange(max_k, device=device).unsqueeze(0).expand(B, -1)
            selection_k_mask = arange_k < num_samples.unsqueeze(1)

            indices_to_sample = topk_indices[selection_k_mask]
            batch_idx_for_scatter = (
                torch.arange(B, device=device)
                .unsqueeze(1)
                .expand(-1, max_k)[selection_k_mask]
            )

            sample_mask[batch_idx_for_scatter, indices_to_sample] = True

        # 3. Remove overlaps with ANY of the M target masks
        # target_masks has shape [M, B, L]; drop any context point used in any target mask
        overlap_mask = (sample_mask.unsqueeze(0) & target_masks).any(dim=0)

        # Clone sample_mask to avoid in-place modification issues with autograd
        valid_mask = sample_mask.clone()
        valid_mask[overlap_mask] = False

        # 4. Build the context view: keep only the valid positions; hide timestamps,
        #    zero the adjacency (same policy as before, now owned by the mask)
        adj = level.side.get("adj")
        side = {
            **level.side,
            "t": None,
            "adj": None if adj is None else torch.zeros_like(adj),
        }
        return Level(x * valid_mask.unsqueeze(-1), level.padding_mask, side)


#############################################################################################################
# Encoder
#############################################################################################################


# =============================================================================
# ARCHITECTURAL OVERVIEW: ATTENTION UPSAMPLING & TRANSFORMER STACK
# =============================================================================
# This code defines the core contextual representation engine for the JEPA framework.
# It utilizes a hierarchy of Transformer models, where each model processes a
# sequence at a different resolution (Coarse -> Fine).
#
# The crucial innovation here is the `connect_levels` feature. Instead of processing
# each hierarchical level entirely independently, the architecture forces the models
# to share their learned representations.
#
# How it works:
# 1. The Coarsest model processes a short, highly compressed sequence and generates
#    a square Attention Map (size: L_short x L_short) representing global relationships.
# 2. Before the next (Finer) model processes its longer sequence (L_long), the
#    `AttentionUpsampler` uses Bilinear Interpolation to stretch out the Coarse
#    attention map to (L_long x L_long).
# 3. This stretched attention map is passed into the Finer model as a prior
#    (`prev_attn`), explicitly guiding its self-attention mechanism to focus on
#    the global relationships discovered by the Coarse layer.
#
# This creates a "Chained Transformer Stack", allowing global context to explicitly
# guide fine-grained, localized predictions.
# =============================================================================


class AttentionUpsampler(nn.Module):
    """
    Stretches a square attention map from a lower-resolution hierarchical level
    to fit the dimensions of a higher-resolution level.
    """

    def __init__(self, nheads, normalize=False):
        super().__init__()
        # If true, re-normalizes the attention probabilities so they sum to 1
        self.normalize = normalize
        # The expected scale factor between hierarchical levels (e.g., L/2 -> L means scale=2)
        self.scale = 2

    def forward(
        self, attn: torch.Tensor, target_len: int | None = None
    ) -> torch.Tensor:
        """
        Args:
            attn: The attention map from the previous (coarser) level.
                  Shape: [Batch, Heads, L_in, L_in]
            target_len: The sequence length of the current (finer) level.
                        If not provided, it defaults to L_in * self.scale.

        Returns:
            The upsampled attention map.
            Shape: [Batch, Heads, target_len, target_len]
        """
        # 1. Determine the target size for the new attention map
        if target_len is None:
            target_len = attn.size(-1) * self.scale

        # 2. Perform 2D spatial interpolation.
        # By treating the [L_in, L_in] attention map like an image, bilinear interpolation
        # smoothly stretches the values to fit the new [target_len, target_len] grid.
        attn_up = F.interpolate(
            attn, size=(target_len, target_len), mode="bilinear", align_corners=False
        )

        # 3. Optional Normalization
        # Bilinear interpolation does not guarantee that the rows of the resulting
        # probability matrix will sum to exactly 1.0. This fixes that.
        if self.normalize:
            attn_up = attn_up / attn_up.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return attn_up


class TransformerStack(JEPA_Encoder):
    """
    A unified JEPA Encoder composed of multiple discrete Transformer models.
    Each model is responsible for exactly one hierarchical level.
    The models are chained together: the attention map from level i is upsampled
    and passed as a prior to level i+1.
    """

    def __init__(self, models, upsampler=None, connect_levels=True):
        super().__init__()

        # Validation: If we want to chain levels, we MUST have a way to resize the attention maps
        if connect_levels and upsampler is None:
            raise ValueError("connect_levels=True needs an upsampler")

        # Store the individual Transformer models (one for each hierarchical level)
        self.models = nn.ModuleList(models)
        self.connect_levels = connect_levels
        self.upsampler = upsampler if connect_levels else None

    @property
    def out_dims(self):
        """
        Satisfies the JEPA_Encoder interface by returning the expected
        embedding dimension for each level.
        """
        return [m.input_dim for m in self.models]

    def forward(self, levels):
        """
        Sequentially executes the Transformer stack.

        Args:
            levels: A list of JEPA `Level` dataclasses ordered Coarse -> Fine.
                    Each level contains the sequence embeddings (.x), padding mask,
                    and auxiliary side info (timestamps, adjacency matrices, etc.).

        Returns:
            EncoderOutput: A dataclass containing a list of the final embeddings
                           and a list of the attention maps for each level.
        """
        embs, attns, prev = [], [], None

        # Loop through each hierarchical level and its dedicated Transformer model
        for i, (model, lv) in enumerate(zip(self.models, levels)):
            pad = lv.padding_mask

            # If no padding mask exists in the dataclass, create a default "all False" mask
            if pad is None:
                pad = torch.zeros(lv.x.shape[:2], dtype=torch.bool, device=lv.x.device)

            prev_attn = None

            # If chaining is enabled and we are not on the first (coarsest) level,
            # grab the attention map from the PREVIOUS level and stretch it to fit
            # the CURRENT level's sequence length.
            if self.connect_levels and prev is not None:
                prev_attn = self.upsampler(prev, target_len=lv.x.size(1))

            # Execute the Transformer model for this specific level.
            # We inject the current level's sequence (.x), side info (t, adj, poi_seq, motion),
            # and the upsampled prior attention map (prev_attn).
            emb, attn = model(
                cell_emb=lv.x,
                t=lv.side.get("t"),
                adj=lv.side.get("adj"),
                src_padding_mask=pad,
                indicator=i,
                poi_seq=lv.side.get("poi_seq"),
                motion=lv.side.get("motion"),
                prev_attn=prev_attn,
            )

            # Store outputs
            embs.append(emb)
            attns.append(attn)

            # Save this level's attention map to be upsampled by the next level
            prev = attn

        return EncoderOutput(embs, attns)


#################################################################################################################
# Predictor
#################################################################################################################


class PredictionHead(JEPA_Head):
    """
    One predictor head for one hierarchy level.

    Owns everything that turns a context-encoder output into predictions
    at the masked positions:
      - a learned positional embedding (per position, per level)
      - a learned latent query vector (shared across positions; PE disambiguates)
      - an x_transformers.Decoder that lets latent queries attend to context
      - a LayerNorm and a final projection
    """

    def __init__(self, dim: int, max_len: int, depth: int = 1, heads: int = 8):
        super().__init__()
        self.dim = dim

        self.decoder = Decoder(dim=dim, depth=depth, heads=heads)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)

        self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        self.latent_var = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, context_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        context_out : [B, L, d]     context-encoder output (grad flows here)
        mask        : [M, B, L]     bool, True = position is a target (to predict)

        returns     : [M, B, L, d]  predictions, zeroed at non-target positions
        """
        M, B, L = mask.shape
        d = self.dim

        # Expand context to M copies: [B, L, d] -> [M*B, L, d]
        ctx = context_out.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, L, d)

        # Flatten mask: [M, B, L] -> [M*B, L]
        flat_mask = mask.reshape(M * B, L)  # bool

        # Positional embedding, cropped to L, broadcast to M*B
        pe = self.pe.expand(M * B, -1, -1)[:, :L, :]  # [M*B, L, d]
        masked_pe = pe * flat_mask.unsqueeze(-1).to(pe.dtype)

        # Context tokens + PE (PE only at masked slots)
        ctx_with_pe = ctx + masked_pe

        # Latent queries: same vector everywhere, PE disambiguates position
        z = self.latent_var.expand(M * B, L, -1) + masked_pe  # [M*B, L, d]

        # Predictor: latent half attends to context half
        out = torch.cat([ctx_with_pe, z], dim=1)  # [M*B, 2L, d]
        out = self.decoder(out)
        out = self.norm(out)
        out = out[:, L:, :]  # keep latent half
        out = self.proj_out(out)

        # Zero out non-target positions (matches old behaviour)
        out = out * flat_mask.unsqueeze(-1).to(out.dtype)

        return out.reshape(M, B, L, d)

    def predict_all(self, context_out):
        """Inference: predict every position. [B, L, d] -> [B, L, d]."""
        B, L, _ = context_out.shape
        full = torch.ones(1, B, L, dtype=torch.bool, device=context_out.device)
        return self.forward(context_out, full).squeeze(0)


#######################################################
# Loss
#######################################################


class JEPA_VIC_REG(JEPA_Loss):
    """
    Computes a hybrid loss combining a standard prediction loss (Smooth L1) with
    VICReg (Variance-Invariance-Covariance Regularization).

    VICReg prevents "informational collapse" (where the model outputs the same
    embedding for every input) without needing negative pairs.
    """

    def __init__(self, level_weights=(0.8, 0.15, 0.05), eps=1e-4):
        super().__init__()
        # Weights for the hierarchical levels (Coarse/Level 1 gets 0.8, etc.)
        self.level_weights = level_weights
        # Epsilon for numerical stability to prevent division by zero or log(0)
        self.eps = eps
        # Smooth L1 (Huber) loss for the prediction term.
        # reduction="none" allows us to manually sum/mean specific dimensions later.
        self.pred_loss = nn.SmoothL1Loss(reduction="none")

    def build(self, dims):
        """
        Called once to initialize trainable parameters.
        Creates an "expander" (projection head) for each hierarchical level.
        VICReg applies its regularization in a higher-dimensional projected space,
        so this maps the embedding dimension (d) to (2 * d).
        """
        self.expanders = nn.ModuleList([nn.Linear(d, 2 * d) for d in dims])

    def variance_loss(self, embeddings):
        """
        Variance Regularization: Forces the embeddings to be diverse across the batch/sequence.
        Prevents informational collapse by ensuring variance doesn't drop below 1.
        """
        # Calculate variance along the flattened [M*B*L] dimension (dim=0)
        # Adds eps to prevent exact 0 variance which can cause instability.
        variances = embeddings.var(dim=0) + self.eps

        # Penalize variances that are less than 1.
        # If variance >= 1, (1 - variance) is negative, and ReLU makes the penalty 0.
        return torch.mean(torch.relu(1 - variances))

    def covariance_loss(self, embeddings):
        """
        Covariance Regularization: Decorrelates the feature dimensions from each other.
        Forces the off-diagonal elements of the covariance matrix to be 0, ensuring
        that each dimension of the embedding captures unique information.
        """
        # 1. Center the embeddings by subtracting the mean
        mean = embeddings.mean(dim=0, keepdim=True)
        embeddings_centered = embeddings - mean

        # 2. Calculate the covariance matrix (Shape: [new_d, new_d])
        cov = (embeddings_centered.T @ embeddings_centered) / (embeddings.size(0) - 1)

        # 3. Create a mask that is 0 on the diagonal and 1 everywhere else
        off_diagonal_mask = torch.ones_like(cov) - torch.eye(
            cov.size(0), device=cov.device
        )

        # 4. Square the covariance matrix (so negatives become positive penalties)
        #    and mask out the diagonal. We only penalize non-zero off-diagonals.
        cov_loss = (cov**2 * off_diagonal_mask).mean()
        return cov_loss

    def level_loss(self, out, level):
        """
        Calculates the total loss for a single hierarchical level.
        Total Loss = Prediction Loss + Variance Penalty + Covariance Penalty.
        """
        pred, tgt = out["pred"], out["target"]
        assert pred.shape == tgt.shape

        # 1. Prediction Term (The "Invariance" part of VICReg)
        # Sums the loss across the spatial/feature dimensions, then averages over the batch.
        pred_term = self.pred_loss(pred, tgt).sum(dim=(-1, -2)).mean()

        # 2. Regularization Terms
        # Grab the expander projection head for this specific level
        exp = self.expanders[level]
        d = out["context_emb"].size(-1)

        # Flatten the embeddings from [M, B, L, d] to [M*B*L, d], then project to [M*B*L, 2d]
        z_c = exp(out["context_emb"].reshape(-1, d))
        z_t = exp(out["target_full"].reshape(-1, d))

        # Calculate variance and covariance penalties for both context and target embeddings
        reg = (
            self.variance_loss(z_c)
            + self.variance_loss(z_t)
            + self.covariance_loss(z_c)
            + self.covariance_loss(z_t)
        )

        return pred_term + reg

    def combine(self, losses):
        """
        Aggregates the individual losses from each hierarchical level into a single
        scalar using the predefined level weights (e.g., 0.8 * L1 + 0.15 * L2 + 0.05 * L3).
        """
        return sum(w * l for w, l in zip(self.level_weights, losses))
