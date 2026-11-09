import sys

sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from config import Config
from torch.nn.utils.rnn import pad_sequence
from x_transformers import Decoder

from model.base_model import Model, init_weights
from model.unet import *


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, token_embedding.size(1), :])


class JEPA_base(nn.Module):
    def __init__(self, M=4):
        super(JEPA_base, self).__init__()
        self.M = M

        self.unet = UNet1DEncoder(Config.seq_embedding_dim, Config.seq_embedding_dim)
        self.upsampler = AttentionUpsampler(Config.attn_head, normalize=False)

        self.target_encoder_3 = Model(Config, input_dim=Config.seq_embedding_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.context_encoder_3 = Model(Config, input_dim=Config.seq_embedding_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.context_encoder_2 = Model(Config, input_dim=Config.l2_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.target_encoder_2 = Model(Config, input_dim=Config.l2_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.context_encoder_1 = Model(Config, input_dim=Config.l1_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.target_encoder_1 = Model(Config, input_dim=Config.l1_dim,
                                       hidden_dim=Config.hidden_dim,
                                       num_layers=Config.attn_layer,
                                       nheads=Config.attn_head,
                                       maxlen=Config.max_traj_len+1)

        self.expander_1 = nn.Linear(Config.l1_dim, Config.l1_dim*2)  # Reduce back to scalar
        self.expander_2 = nn.Linear(Config.l2_dim, Config.l2_dim*2)  # Reduce back to scalar
        self.expander_3 = nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim*2)  # Reduce back to scalar

        self.norm_1 = nn.LayerNorm(Config.l1_dim)
        self.norm_2 = nn.LayerNorm(Config.l2_dim)
        self.norm_3 = nn.LayerNorm(Config.seq_embedding_dim)

        self.pe_1 = nn.Parameter(torch.randn(1, Config.max_traj_len, Config.l1_dim))
        self.pe_2 = nn.Parameter(torch.randn(1, Config.max_traj_len, Config.l2_dim))
        self.pe_3 = nn.Parameter(torch.randn(1, Config.max_traj_len, Config.cell_embedding_dim))

        self.latent_var_3 = nn.Parameter(torch.randn(1, 1, Config.cell_embedding_dim))
        self.latent_var_2 = nn.Parameter(torch.randn(1, 1, Config.l2_dim))
        self.latent_var_1 = nn.Parameter(torch.randn(1, 1, Config.l1_dim))
        self.predictor_3 = Decoder(dim=Config.cell_embedding_dim, depth=1, heads=8)
        self.predictor_2 = Decoder(dim=Config.l2_dim, depth=1, heads=8)
        self.predictor_1 = Decoder(dim=Config.l1_dim, depth=1, heads=8)

        self.proj_out_1 = nn.Linear(Config.l1_dim, Config.l1_dim)
        self.proj_out_2 = nn.Linear(Config.l2_dim, Config.l2_dim)
        self.proj_out_3 = nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim)
        self.loss = nn.SmoothL1Loss(reduction='none')
        # self.log_vars = nn.Parameter(torch.zeros(3))

        for m in self.context_encoder_1.modules():
            init_weights(m)
        for m in self.context_encoder_2.modules():
            init_weights(m)
        for m in self.context_encoder_3.modules():
            init_weights(m)
        for m in self.target_encoder_1.modules():
            init_weights(m)
        for m in self.target_encoder_2.modules():
            init_weights(m)
        for m in self.target_encoder_3.modules():
            init_weights(m)
        #
        for m in self.predictor_1.modules():
            init_weights(m)
        for m in self.predictor_2.modules():
            init_weights(m)
        for m in self.predictor_3.modules():
            init_weights(m)


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

        if mode not in ['resample', 'waypoints']:
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
            starts = (torch.rand(len(succ_indices), device=device) * max_start.float()).long()
            
            arange_l = torch.arange(L, device=device).unsqueeze(0).expand(len(succ_indices), -1)
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
                arange_k = torch.arange(max_k, device=device).unsqueeze(0).expand(len(rand_indices), -1)
                selection_k_mask = arange_k < rand_num_mask.unsqueeze(1)
                
                indices_to_mask = topk_indices[selection_k_mask]
                batch_idx_for_scatter = torch.arange(len(rand_indices), device=device).unsqueeze(1).expand(-1, max_k)[selection_k_mask]

                random_mask_for_rand = torch.zeros(len(rand_indices), L, dtype=torch.bool, device=device)
                random_mask_for_rand[batch_idx_for_scatter, indices_to_mask] = True
                final_mask[rand_indices] = random_mask_for_rand

        masked_x = x * final_mask.unsqueeze(-1).to(x.dtype)

        # The mask_indices are not used, so we can avoid computing it and just return None.
        return masked_x, final_mask, None
        # return mask_indices     # [B, len]


    @torch.no_grad()
    def target_seg(self, x, Config):
        """
        All we need is the target input, and the M * B masked segments.
        """
        cell_emb, t, num_points, src_padding_mask, adj, indicator, poi_seq, motion, prev_attn_t = x

        self.target_encoder_1.eval()
        self.target_encoder_2.eval()
        self.target_encoder_3.eval()
        # generate embeddings by target encoder.
        if indicator == 0:
            x_emb, attn_target = self.target_encoder_1(**{
                'cell_emb': cell_emb,
                't': None,
                'src_padding_mask': src_padding_mask,
                'adj': adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_t
            })  # [B, L/4, 4d]
        elif indicator == 1:
            x_emb, attn_target = self.target_encoder_2(**{
                'cell_emb': cell_emb,
                't': None,
                'src_padding_mask': src_padding_mask,
                'adj': adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_t
            })  # [B, L/2, 2d]
        else:
            x_emb, attn_target = self.target_encoder_3(**{
                'cell_emb': cell_emb,
                't': None,
                'src_padding_mask': src_padding_mask,
                'adj': adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_t
            })  # [B, L, d]

        mask_ratio = Config.mask_ratio
        idx = torch.randint(low=0, high=len(mask_ratio), size=(1,))
        r = mask_ratio[idx]

        mode = 'waypoints' if indicator in [0, 1] else 'resample'
        
        B, L, d = x_emb.shape
        # Repeat input for M samples and reshape for batch processing
        x_emb_repeated = x_emb.unsqueeze(0).expand(self.M, -1, -1, -1).reshape(self.M * B, L, d)
        src_padding_mask_repeated = src_padding_mask.unsqueeze(0).expand(self.M, -1, -1).reshape(self.M * B, L)

        # Get masks for all M*B samples at once
        masked_x, masks, _ = self.mask_tensor(x_emb_repeated, src_padding_mask_repeated, r, Config.succ_prob, mode)

        # Reshape back to (M, B, L, d) and (M, B, L)
        all_mask_x = masked_x.view(self.M, B, L, d)
        all_masks = masks.view(self.M, B, L)

        return all_mask_x, all_masks, attn_target, x_emb



    def context_seg(self, x, Config):
        cell_emb, t, src_padding_mask, adj, indicator, motion, all_masks = x
        lb = Config.context_lb  # Lower bound ratio for sampling

        # Determine the number of sequence elements to sample based on the lower bound
        # B, L, d = cell_emb.shape
        if Config.humob:
            B, L, d = cell_emb.shape
            # d = Config.seq_embedding_dim
        else:
            B, L, d = cell_emb.shape

        # --- Vectorized Implementation ---

        # 1. Determine number of samples for each item in the batch
        ratios = torch.rand((B,), device=cell_emb.device).clamp(min=lb, max=1.0)
        non_padded_lens = L - src_padding_mask.sum(dim=1)
        num_samples = (non_padded_lens.float() * ratios).long()

        # 2. Vectorized random sampling to create initial sample_mask
        noise = torch.rand(B, L, device=cell_emb.device)
        noise.masked_fill_(src_padding_mask, -1.0)  # Padded positions can't be sampled

        max_k = num_samples.max().item()
        sample_mask = torch.zeros(B, L, dtype=torch.bool, device=cell_emb.device)

        if max_k > 0:
            # Use topk to find indices for sampling
            _, topk_indices = torch.topk(noise, k=max_k, dim=1)

            # Create a selection mask for variable number of samples
            arange_k = torch.arange(max_k, device=cell_emb.device).unsqueeze(0).expand(B, -1)
            selection_k_mask = arange_k < num_samples.unsqueeze(1)
            
            indices_to_sample = topk_indices[selection_k_mask]
            batch_idx_for_scatter = torch.arange(B, device=cell_emb.device).unsqueeze(1).expand(-1, max_k)[selection_k_mask]
            
            sample_mask[batch_idx_for_scatter, indices_to_sample] = True

        # 3. Remove overlaps with any of the M target masks
        # `all_masks` has shape [M, B, L]. `sample_mask` has shape [B, L]
        overlap_mask = (sample_mask.unsqueeze(0) & all_masks).any(dim=0)
        sample_mask[overlap_mask] = 0
        valid_mask = sample_mask

        # 4. Apply the final valid mask to get the context features
        aug_sampled_emb = cell_emb * valid_mask.unsqueeze(-1)
        
        aug_sampled_motion = None
        if indicator == 3 and motion is not None:
            aug_sampled_motion = motion * valid_mask.unsqueeze(-1)

        aug_sampled_adj = None
        if adj is not None:
            aug_sampled_adj = torch.zeros_like(adj)
            
        # The original loop effectively just copied src_padding_mask to aug_src_padding_mask.
        aug_src_padding_mask = src_padding_mask

        return aug_sampled_emb, t, aug_src_padding_mask, aug_sampled_adj, aug_sampled_motion

    @staticmethod
    def variance_loss(embeddings, eps=1e-4):
        # Flatten to [M*B*L, new_d] to calculate variance across M, B, L
        # flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        variances = embeddings.var(dim=0) + eps
        return torch.mean(torch.relu(1 - variances))

    @staticmethod
    def covariance_loss(embeddings):
        # Flatten to [M*B*L, new_d] to calculate covariance
        # flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        mean = embeddings.mean(dim=0, keepdim=True)
        embeddings_centered = embeddings - mean
        cov = (embeddings_centered.T @ embeddings_centered) / (embeddings.size(0) - 1)
        off_diagonal_mask = torch.ones_like(cov) - torch.eye(cov.size(0), device=cov.device)
        cov_loss = (cov ** 2 * off_diagonal_mask).mean()
        return cov_loss


    def loss_fn_old(self, context_out, targets):
        assert context_out.shape == targets.shape, "context_out and targets must have the same shape"
        M, B, L, d = context_out.shape

        # Calculate MSE loss for each element
        # mse_loss = F.mse_loss(context_out, targets, reduction='none')
        loss = self.loss(context_out, targets)

        loss = loss.sum(dim=(-1,-2))
        loss = loss.mean()

        return loss
    def loss_fn(self, context_out, targets, var_covar):
        assert context_out.shape == targets.shape, "context_out and targets must have the same shape"

        loss = self.loss(context_out, targets)
        loss = loss.sum(dim=(-1,-2))
        loss = loss.mean()
        # print(f"loss: {loss}, var_covar: {var_covar}.")
        return loss + var_covar
        # return loss

    def compute_weighted_loss(self, losses):
        weighted_loss = 0
        for i in range(Config.exec_layer):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * losses[i] + self.log_vars[i]
        return weighted_loss

    def interpret(self, cell_emb, t, num_points, poi_seq, adj, motion, exec_layer):
        # exec_layer = 1
        if Config.humob:
            B, L, d = cell_emb.shape
            # d = Config.seq_embedding_dim
        else:
            B, L, d = cell_emb.shape

        mapping = {0: 4, 1: 2, 2: 1}
        # we only abstract the seq emb at first stage.
        cell_emb_1, cell_emb_2, cell_emb_3 = self.unet(cell_emb)
        # print(cell_emb_3.shape)
        prev_attn_c = None
        emb_list = [cell_emb_3, cell_emb_2, cell_emb_1]
        lengths = [e.size(1) for e in emb_list]

        for i, emb in enumerate(emb_list):
            if i == exec_layer:
                break
            denom = mapping[i]
            temp_num_points = torch.div(num_points + denom - 1, denom, rounding_mode='trunc')
            # temp_num_points = torch.div(num_points, denom, rounding_mode='trunc')
            # print(num_points)
            max_traj_len = temp_num_points.max().item()  # in essense -- trajs1_len[0]
            # print(cell_emb.shape)
            src_padding_mask = torch.arange(max_traj_len, device=cell_emb.device)[None, :] >= temp_num_points[:, None]
            # print(src_padding_mask.shape)
            if i == 0:
                emb, attn_context = self.context_encoder_1(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })
            elif i == 1:
                emb, attn_context = self.context_encoder_2(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })
            else:
                emb, attn_context = self.context_encoder_3(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })

            if i + 1 < min(exec_layer, len(lengths)):
                target_len = lengths[i + 1]
                prev_attn_c = self.upsampler(attn_context, target_len=target_len)
            else:
                prev_attn_c = None


        return emb

    def run_layer(self, cell_emb, t, num_points, adj, poi_seq, motion, indicator, prev_attn_c, prev_attn_t):
        """
        accept conv.-ed features and split into context/target.
        construct target.
        do JEPA.
        return informaiton (attn. weights) to the upper layer.
        """
        if Config.humob:
            B, L, d = cell_emb.shape
            # d = Config.seq_embedding_dim
        else:
            # print(cell_emb.shape)
            B, L, d = cell_emb.shape

        max_traj_len = num_points.max().item()  # in essense -- trajs1_len[0]
        src_padding_mask = torch.arange(max_traj_len, device=cell_emb.device)[None, :] >= num_points[:, None]

        # get context/target. (without time)
        target_emb, all_masks, attn_target, x_emb = self.target_seg(
            (cell_emb, None, num_points, src_padding_mask, adj, indicator, None, motion, prev_attn_t), Config)

        """
        VICReg
        """
        if indicator == 0:
            target_exp = self.expander_1(x_emb.reshape(-1, d))
        elif indicator == 1:
            target_exp = self.expander_2(x_emb.reshape(-1, d))
        else:
            target_exp = self.expander_3(x_emb.reshape(-1, d))
        # target_exp = target_exp.reshape(target_emb.shape)
        var_target = self.variance_loss(target_exp)
        covar_target = self.covariance_loss(target_exp)
        """
        context construction.
        """
        aug_sampled_emb, aug_sampled_t, aug_src_padding_mask, aug_sampled_adj, aug_sampled_motion = self.context_seg(
            (cell_emb, t, src_padding_mask, adj, indicator, motion, all_masks), Config)


        # var_context, covar_context = [], []
        if indicator == 0:
            output, attn_context = self.context_encoder_1(**{
                'cell_emb': aug_sampled_emb,
                't': None,
                'src_padding_mask': aug_src_padding_mask,
                'adj': aug_sampled_adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_c
            })
        elif indicator == 1:
            output, attn_context = self.context_encoder_2(**{
                'cell_emb': aug_sampled_emb,
                't': None,
                'src_padding_mask': aug_src_padding_mask,
                'adj': aug_sampled_adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_c
            })
        else:
            output, attn_context = self.context_encoder_3(**{
                'cell_emb': aug_sampled_emb,
                't': None,
                'src_padding_mask': aug_src_padding_mask,
                'adj': aug_sampled_adj,
                'indicator': indicator,
                'poi_seq': poi_seq,
                'motion': motion,
                'prev_attn': prev_attn_c
            })

        if indicator == 0:
            context_exp = self.expander_1(output.reshape(-1, d))
        elif indicator == 1:
            context_exp = self.expander_2(output.reshape(-1, d))
        else:
            context_exp = self.expander_3(output.reshape(-1, d))
        var_context = self.variance_loss(context_exp)
        covar_context = self.covariance_loss(context_exp)

        var = var_context + var_target
        covar = covar_context + covar_target

        # --- Vectorized Predictive Process ---
        
        # Reshape masks and output for parallel processing
        # mask shape: [M, B, L] -> [M*B, L]
        mask = all_masks.view(self.M * B, L)
        # output shape: [B, L, d] -> [M*B, L, d]
        output_expanded = output.unsqueeze(0).expand(self.M, -1, -1, -1).reshape(self.M * B, L, d)

        # Select and expand positional embeddings based on indicator
        if indicator == 0:
            pe = self.pe_1.expand(self.M * B, -1, -1)
            z_base = self.latent_var_1
            predictor = self.predictor_1
            norm = self.norm_1
            proj_out = self.proj_out_1
        elif indicator == 1:
            pe = self.pe_2.expand(self.M * B, -1, -1)
            z_base = self.latent_var_2
            predictor = self.predictor_2
            norm = self.norm_2
            proj_out = self.proj_out_2
        else:
            pe = self.pe_3.expand(self.M * B, -1, -1)
            z_base = self.latent_var_3
            predictor = self.predictor_3
            norm = self.norm_3
            proj_out = self.proj_out_3

        pe = pe[:, :L, :]
        masked_pe = pe * mask.unsqueeze(-1)
        
        # Apply PE to output
        output_with_pe = output_expanded + masked_pe
        
        # Prepare latent variables `z` with PE
        z = z_base.expand(self.M * B, L, -1)
        z = z + masked_pe
        
        # Concatenate and predict
        out = torch.cat([output_with_pe, z], dim=1)
        out = predictor(out)
        out = norm(out)
        
        # Final projection and masking
        out = out[:, L:, :]
        out = proj_out(out)
        context_out = out * mask.unsqueeze(-1)

        # Reshape to final [M, B, L, d]
        context_out = context_out.view(self.M, B, L, d)
        
        # tgt and context_emb should be replaced to attention weights.
        assert context_out.shape == (self.M, B, L, d)
        assert target_emb.shape == (self.M, B, L, d)
        return context_out, target_emb, attn_target, var+covar, all_masks, attn_context


    def forward(self, cell_emb, t, num_points, adj, poi_seq, motion, exec_layer):
        """
        received 3-layer unet outputs, each output is fed into different JEPA layers.
        """


        losses = []
        mapping = {0: 4, 1: 2, 2: 1}
        # we only abstract the seq emb at first stage.
        cell_emb_1, cell_emb_2, cell_emb_3 = self.unet(cell_emb)
        prev_attn_c, prev_attn_t = None, None
        emb_list = [cell_emb_3, cell_emb_2, cell_emb_1]

        for i, emb in enumerate(emb_list):
            if i == exec_layer:
                break
            denom = mapping[i]
            temp_num_points = torch.div(num_points + denom - 1, denom, rounding_mode='trunc')
            # temp_num_points = torch.div(num_points, denom, rounding_mode='trunc')
            out, target_emb, attn_target, var_covar, all_masks, attn_context = self.run_layer(
                emb, t, temp_num_points, adj, poi_seq, motion, i, prev_attn_c, prev_attn_t)
            """
            prepare upsampling and update previous attention weights.
            """
            if i + 1 < exec_layer:
                next_denom = mapping[i + 1]
                next_len = torch.div(num_points + next_denom - 1, next_denom, rounding_mode='trunc').max().item()
                # next_len = torch.div(num_points, next_denom, rounding_mode='trunc').max().item()
                prev_attn_c = self.upsampler(attn_context, target_len=next_len)
                prev_attn_t = self.upsampler(attn_target, target_len=next_len)
            else:
                prev_attn_c = None
                prev_attn_t = None
            # calculate loss.
            loss = self.loss_fn(out, target_emb, var_covar)
            losses.append(loss)

        loss = 0.8 * losses[0] + 0.15 * losses[1] + 0.05 * losses[2]

        return loss, losses

    def interp_all(self, cell_emb, t, num_points, poi_seq, adj, motion, exec_layer):
        # exec_layer = 1
        if Config.humob:
            B, L = cell_emb.shape
            d = Config.seq_embedding_dim
        else:
            B, L, d = cell_emb.shape

        mapping = {0: 4, 1: 2, 2: 1}
        # we only abstract the seq emb at first stage.
        cell_emb_1, cell_emb_2, cell_emb_3 = self.unet(cell_emb)
        # print(cell_emb_3.shape)
        prev_attn_c = None
        emb_all = []
        attn_all = []
        padding_all = []
        emb_list = [cell_emb_3, cell_emb_2, cell_emb_1]
        for i, emb in enumerate(emb_list):
            if i == exec_layer:
                break
            denom = mapping[i]
            temp_num_points = torch.div(num_points + denom - 1, denom, rounding_mode='trunc')
            # print(num_points)
            max_traj_len = temp_num_points.max().item()  # in essense -- trajs1_len[0]
            # print(cell_emb.shape)
            src_padding_mask = torch.arange(max_traj_len, device=cell_emb.device)[None, :] >= temp_num_points[:, None]
            # print(src_padding_mask.shape)
            if i == 0:
                emb, attn_context = self.context_encoder_1(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })
            elif i == 1:
                emb, attn_context = self.context_encoder_2(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })
            else:
                emb, attn_context = self.context_encoder_3(**{
                    'cell_emb': emb,
                    't': t,
                    'src_padding_mask': src_padding_mask,
                    'adj': adj,
                    'indicator': i,
                    'poi_seq': poi_seq,
                    'motion': motion,
                    'prev_attn': prev_attn_c
                })

                pe = self.pe_3.repeat(B, 1, 1)
                pe = pe[:, :L, :]
                output = emb + pe
                pe_z = self.pe_3.repeat(B, 1, 1)
                pe_z = pe_z[:, :L, :]
                z = self.latent_var_3.repeat(B, L, 1)
                z = z + pe_z
                out = torch.cat([output, z], dim=1)
                out = self.predictor_3(out)
                out = self.norm_3(out)
                out = out[:, L:, :]
                out = self.proj_out_3(out)

            emb_all.append(emb)

            attn_all.append(attn_context)
            padding_all.append(src_padding_mask)

            if i + 1 < exec_layer:
                next_len = emb_list[i + 1].size(1)
                prev_attn_c = self.upsampler(attn_context, target_len=next_len)
            else:
                prev_attn_c = None


        return emb_all, padding_all, out, attn_all

    def load_checkpoint(self):

        checkpoint_file = '../HT-JEPA/checkpoints/porto_20200_new_porto_ujepa_1e-4_cell-res11_seed2000_gamma0.5_step5_M4_bs64_optim_conv_nores_maxpool_norm_256_256_8155_loss_cosine.pt'
        
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])

        return self
