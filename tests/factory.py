# jepa/factory.py
from config import Config
from jepa.base import JEPA
from jepa.hmtraj_components import (
    JEPA_VIC_REG,
    AttentionUpsampler,
    JEPA_HMTraj_Mask,
    PredictionHead,
    TransformerStack,
    UNet1DEncoder,
    UNetHierarchy,
)
from model.base_model import Model, init_weights


def build_default_jepa():
    dims = [Config.l1_dim, Config.l2_dim, Config.seq_embedding_dim]  # coarse -> fine

    def stack():
        return TransformerStack(
            [
                Model(
                    Config,
                    input_dim=d,
                    hidden_dim=Config.hidden_dim,
                    num_layers=Config.attn_layer,
                    nheads=Config.attn_head,
                    maxlen=Config.max_traj_len + 1,
                )
                for d in dims
            ],
            upsampler=AttentionUpsampler(Config.attn_head, normalize=False),
        )

    context, target = stack(), stack()
    heads = [
        PredictionHead(dim=d, max_len=Config.max_traj_len, depth=1, heads=8)
        for d in dims
    ]

    for part in (context, target, *heads):  # same init scope as before
        for m in part.modules():
            init_weights(m)

    return JEPA(
        hierarchy=UNetHierarchy(
            UNet1DEncoder(Config.seq_embedding_dim, Config.seq_embedding_dim)
        ),
        context_encoder=context,
        target_encoder=target,
        heads=heads,
        mask=JEPA_HMTraj_Mask(
            sample_number=Config.sample_number,
            mask_ratio=Config.mask_ratio,
            succ_prob=Config.succ_prob,
            context_lb=Config.context_lb,
        ),
        loss=JEPA_VIC_REG(),
    )
