# tests/test_jepa_equivalence.py
import gc
import re

import pytest
import torch

from config import Config

# =============================================================================
# GLOBAL CONFIG OVERRIDE (VRAM Management)
# We must patch the Config dimensions BEFORE importing the models.
# The original architecture defaults to large embedding dimensions (256).
# By shrinking them globally here, we force the PyTorch networks to initialize
# with a much smaller memory footprint, allowing the test to fit on an 8GB GPU.
# =============================================================================
Config.seq_embedding_dim = 64
Config.cell_embedding_dim = 64
Config.l1_dim = 32
Config.l2_dim = 32
Config.hidden_dim = 64
Config.attn_head = 4  # Must cleanly divide hidden_dim

from jepa.hmtraj_components import UNetHierarchy
from model.jepa import JEPA_base as Original  # untouched copy of the original
from tests.factory import build_default_jepa as Refactored

# =============================================================================
# TESTING CONSTANTS & TOLERANCES
# PyTorch GPU operations (like reductions in attention/loss) are non-deterministic
# at the microscopic floating-point level due to atomic adds in CUDA.
# Therefore, we define different tolerance thresholds for different checks.
# =============================================================================
M = 2  # Number of masked target segments to generate (kept small to save memory)
EXACT = dict(
    rtol=0, atol=0
)  # Used for direct tensor outputs before reduction operations
LOSS = dict(rtol=1e-6, atol=1e-7)  # Slight tolerance for scalar loss accumulations
GRAD = dict(
    rtol=1e-5, atol=1e-8
)  # Gradients compound floating-point drift, needing looser tolerance

# Test parameters: (Random Seed, [Num points in Traj 1, Num points in Traj 2])
CASES = [(seed, n) for seed in (0, 1, 2) for n in ([16, 12], [14, 16])]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Force PyTorch to use deterministic CUDA algorithms where possible
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


# =============================================================================
# WEIGHT SYNCHRONIZATION
# =============================================================================
def rename_key(k):
    """
    Translate Original state_dict keys (per-level attributes, 1-based)
    into Refactored keys (ModuleList indices, 0-based).
    """
    # FPN
    k = re.sub(r"^unet\.", "hierarchy.fpn.", k)

    # CHANGE 1 -- Encoders: the three per-level encoders now live inside ONE stack per role.
    #   context_encoder_1.* -> context_encoder.models.0.*
    #   target_encoder_1.*  -> target_encoder.models.0.*
    k = re.sub(
        r"^context_encoder_(\d+)\.",
        lambda m: f"context_encoder.models.{int(m.group(1)) - 1}.",
        k,
    )
    k = re.sub(
        r"^target_encoder_(\d+)\.",
        lambda m: f"target_encoder.models.{int(m.group(1)) - 1}.",
        k,
    )

    # Loss expanders: expander_1.* -> loss_module.expanders.0.*
    k = re.sub(
        r"^expander_(\d+)\.",
        lambda m: f"loss_module.expanders.{int(m.group(1)) - 1}.",
        k,
    )

    # Predictor head submodules
    k = re.sub(
        r"^predictor_(\d+)\.",
        lambda m: f"prediction_heads.{int(m.group(1)) - 1}.decoder.",
        k,
    )
    k = re.sub(
        r"^norm_(\d+)\.",
        lambda m: f"prediction_heads.{int(m.group(1)) - 1}.norm.",
        k,
    )
    k = re.sub(
        r"^proj_out_(\d+)\.",
        lambda m: f"prediction_heads.{int(m.group(1)) - 1}.proj_out.",
        k,
    )

    # Bare parameters: pe_1 -> prediction_heads.0.pe, latent_var_1 -> prediction_heads.0.latent_var
    k = re.sub(
        r"^pe_(\d+)$",
        lambda m: f"prediction_heads.{int(m.group(1)) - 1}.pe",
        k,
    )
    k = re.sub(
        r"^latent_var_(\d+)$",
        lambda m: f"prediction_heads.{int(m.group(1)) - 1}.latent_var",
        k,
    )

    return k


def sync(orig, ref):
    """
    Copies the exact initialized weights from the Original model to the Refactored model.
    This is the mathematical foundation of the equivalence test: both models MUST start
    from the exact same state to prove the forward pass is structurally identical.
    """
    sd = {rename_key(k): v.clone() for k, v in orig.state_dict().items()}
    assert len(sd) == len(orig.state_dict()), (
        "Two original keys mapped to the same name!"
    )
    # strict=True ensures that every single parameter in the refactored model was accounted for
    ref.load_state_dict(sd, strict=True)


# =============================================================================
# DATA & EXECUTION HARNESS
# =============================================================================
def make_batch(seed, n_pts, L=16):
    """
    Generates a deterministic dummy batch of trajectories.
    L (Sequence Length) is kept extremely short (16 instead of 200+) to drastically
    reduce the O(N^2) memory footprint of the Transformer attention maps during testing.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d = Config.seq_embedding_dim
    return (
        torch.randn(len(n_pts), L, d, generator=g, device=device),  # cell_emb
        torch.randn(len(n_pts), L, d, generator=g, device=device),  # temporal_emb
        torch.tensor(n_pts, dtype=torch.long, device=device).clamp(max=L),  # num_points
        None,  # adj
        None,  # poi_seq
        None,  # motion
        3,  # exec_layer (run all 3 levels of the hierarchy)
    )


def call(model, name, cell_emb, t, num_points, adj, poi_seq, motion, exec_layer):
    """
    CHANGE 2 -- Translator. The two models no longer have the same call signature:
      Original:   model(cell_emb, t, num_points, adj, poi_seq, motion, exec_layer)
      Refactored: model(cell_emb, num_points, side, exec_layer)   # side = dict of extras
    Tests give this function the same data and it calls each model the way it expects.
    `name` is "forward", "interpret" or "interp_all".
    """
    fn = getattr(model, name)
    if isinstance(model, Original):
        if name == "forward":
            return fn(cell_emb, t, num_points, adj, poi_seq, motion, exec_layer)
        return fn(
            cell_emb=cell_emb,
            t=t,
            num_points=num_points,
            poi_seq=poi_seq,
            adj=adj,
            motion=motion,
            exec_layer=exec_layer,
        )
    side = {"t": t, "adj": adj, "poi_seq": poi_seq, "motion": motion}
    return fn(cell_emb, num_points, side, exec_layer)


def disable_dropout(model):
    """
    CHANGE 3 -- Dropout draws random numbers. The old model drew them interleaved
    (mask level 0, dropout level 0, mask level 1, ...); the new one draws every level's
    masks first and runs the encoders afterwards. Same randomness, different ORDER, so with
    dropout on the two models would see different dropout masks even though the math is the
    same. Zeroing dropout in both keeps the comparison fair; train mode is still exercised
    (gradients, train-mode behaviour).
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0


def normalize(out):
    """
    Adapter function. The Original model's run_layer returned a raw Tuple, while the
    Refactored model returns a clean Dictionary. This maps both to a standard dict format.
    """
    if isinstance(out, dict):
        return dict(
            pred=out["pred"],
            target=out["target"],
            mask=out["mask"],
            attn_context=out["aux_context"],
            attn_target=out["aux_target"],
        )
    pred, target, attn_t, _var_covar, mask, attn_c = out
    return dict(
        pred=pred, target=target, mask=mask, attn_context=attn_c, attn_target=attn_t
    )


def collect(model, batch, seed, train, grads=False):
    """
    The core execution engine for the test. It runs the forward pass (and optionally
    the backward pass) and intercepts all intermediate tensors for inspection.
    """
    cell_emb, t, num_points, adj, poi_seq, motion, exec_layer = batch

    model.train(train)
    if train:
        disable_dropout(model)  # see CHANGE 3
    model.zero_grad(set_to_none=True)

    levels = []

    def record(out):
        levels.append(
            {
                n: v.detach().clone() if torch.is_tensor(v) else v
                for n, v in normalize(out).items()
            }
        )

    # CHANGE 4 -- Where to spy. We want each level's pred / target / mask / attention maps.
    #   Original:   `run_layer` returns them (as a tuple)  -> spy on run_layer.
    #   Refactored: there is no run_layer any more, but every level's tensors are handed to
    #               `loss_module.level_loss(out, i)` as a dict           -> spy on that.
    # `normalize` turns both shapes into the same dict.
    if isinstance(model, Original):
        real = model.run_layer

        def spy(*a, **kw):
            out = real(*a, **kw)
            record(out)
            return out

        model.run_layer = spy

        def restore():
            del model.run_layer
    else:
        real = model.loss_module.level_loss

        def spy(out, i):
            record(out)
            return real(out, i)

        model.loss_module.level_loss = spy

        def restore():
            del model.loss_module.level_loss

    try:
        # CRITICAL: seed immediately before the forward pass so both models draw
        # the same random masks.
        torch.manual_seed(seed)
        loss, losses = call(
            model,
            "forward",
            cell_emb.clone(),
            t.clone(),
            num_points,
            adj,
            poi_seq,
            motion,
            exec_layer,
        )
    finally:
        restore()

    res = {
        "levels": levels,
        "loss": loss.detach().clone(),
        "losses": [l.detach().clone() for l in losses],
    }

    # If testing gradients, run backpropagation and save the gradient of every parameter
    if grads:
        loss.backward()
        res["grads"] = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
    return res


# =============================================================================
# PYTEST FIXTURES (Setup / Teardown)
# =============================================================================
@pytest.fixture
def get_synced_models():
    """
    A Factory Fixture that yields one model at a time.
    Because holding two copies of a deep neural network + computational graphs in VRAM
    causes Out-Of-Memory (OOM) errors on 8GB GPUs, we must build and run them sequentially.
    """

    def _build(model_class):
        # 1. Always create the Original model first to establish the baseline weights
        torch.manual_seed(0)
        orig = Original(M=M).to(device)

        if model_class == Original:
            return orig

        # 2. If the test asked for the Refactored model, build it and sync the weights
        torch.manual_seed(1)
        ref = Refactored(M=M).to(device)
        sync(orig, ref)

        # 3. Destroy the Original model to free up 50% of the VRAM before returning
        del orig
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ref

    return _build


@pytest.fixture(autouse=True)
def clear_vram():
    """
    Runs automatically before and after EVERY single test parameterization.
    Prevents PyTorch's memory allocator from hoarding cached VRAM across tests,
    which would otherwise cause a slow, creeping OOM crash.
    """
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# THE ACTUAL TESTS
# =============================================================================
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("seed,n_pts", CASES)
def test_forward_equivalence(get_synced_models, seed, n_pts, train):
    """
    Proves that given the same weights and same input, both architectures calculate
    the exact same intermediate representations, attention maps, and final loss.
    """
    batch = make_batch(seed, n_pts)

    # Run original, collect tensors, clear VRAM
    orig = get_synced_models(Original)
    a = collect(orig, batch, seed, train)
    del orig

    # Run refactored, collect tensors, clear VRAM
    ref = get_synced_models(Refactored)
    b = collect(ref, batch, seed, train)
    del ref

    # Assert intermediate level outputs (embeddings, masks, attention maps) are identical
    assert len(a["levels"]) == len(b["levels"])
    for i, (la, lb) in enumerate(zip(a["levels"], b["levels"])):
        for name in la:
            torch.testing.assert_close(
                la[name], lb[name], **EXACT, msg=f"level {i}, {name}"
            )

    # Assert individual level losses and total aggregated loss match
    for i, (x, y) in enumerate(zip(a["losses"], b["losses"])):
        torch.testing.assert_close(x, y, **LOSS, msg=f"level loss {i}")
    torch.testing.assert_close(a["loss"], b["loss"], **LOSS, msg="total loss")


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("seed,n_pts", CASES)
def test_gradient_equivalence(get_synced_models, seed, n_pts, train):
    """
    Proves that PyTorch's Autograd engine traverses the refactored computational graph
    in the exact same way as the original, applying identical weight updates.
    """
    batch = make_batch(seed, n_pts)

    orig = get_synced_models(Original)
    ga = collect(orig, batch, seed, train, grads=True)["grads"]
    del orig

    ref = get_synced_models(Refactored)
    gb = collect(ref, batch, seed, train, grads=True)["grads"]
    del ref

    # Map the original gradient dictionary keys to the refactored namespace
    ga = {rename_key(n): g for n, g in ga.items()}
    assert set(ga) == set(gb), (
        f"Parameters receiving gradients differ: {set(ga) ^ set(gb)}"
    )

    # Assert every single weight gradient matches
    for n in ga:
        torch.testing.assert_close(ga[n], gb[n], **GRAD, msg=f"grad of {n}")


def test_inference_paths(get_synced_models):
    """
    Tests the specialized `interp_all` and `interpret` methods used strictly during
    evaluation/inference (e.g., in `train_trajsimi.py`), bypassing the predictive loss steps.
    """
    cell_emb, t, num_points, adj, poi, motion, ex = make_batch(seed=0, n_pts=[16, 12])
    args = (t, num_points, adj, poi, motion, ex)

    orig = get_synced_models(Original)
    orig.eval()
    with torch.no_grad():
        ia = call(orig, "interp_all", cell_emb.clone(), *args)
        interpret_orig = call(orig, "interpret", cell_emb.clone(), *args)
    del orig

    ref = get_synced_models(Refactored)
    ref.eval()
    with torch.no_grad():
        ib = call(ref, "interp_all", cell_emb.clone(), *args)
        interpret_ref = call(ref, "interpret", cell_emb.clone(), *args)
    del ref

    # Assert outputs of interp_all (which returns a complex tuple of lists)
    for x, y in zip(ia[0] + ia[1] + [ia[2]] + ia[3], ib[0] + ib[1] + [ib[2]] + ib[3]):
        torch.testing.assert_close(x, y, **EXACT)

    # Assert primary interpret embedding output
    torch.testing.assert_close(interpret_orig, interpret_ref, **EXACT)


class _FakeFPN(torch.nn.Module):  # finest-first, ceil pooling
    def forward(self, x):
        B, L, d = x.shape
        return tuple(torch.zeros(B, -(-L // s), d) for s in (4, 2, 1))


@pytest.mark.parametrize("L", [15, 16, 17, 30, 63])
def test_hierarchy_padding_matches_ceil_div(L):
    n = torch.arange(1, L + 1)  # one item per valid length
    levels = UNetHierarchy(_FakeFPN())(torch.zeros(L, L, 4), n)
    for lvl, den in zip(levels, (4, 2, 1)):
        want = torch.div(n + den - 1, den, rounding_mode="trunc")
        assert torch.equal(
            lvl.padding_mask, torch.arange(lvl.x.size(1))[None, :] >= want[:, None]
        )


def test_harness_can_fail(get_synced_models):
    """
    Sanity Check (The Negative Test):
    If a testing harness always reports 'success', you cannot trust it. We intentionally
    sabotage the Refactored model's weights to ensure `torch.testing.assert_close`
    will actually fail and catch architectural divergence.
    """
    batch = make_batch(0, [16, 12])

    orig = get_synced_models(Original)
    a = collect(orig, batch, 0, False)
    del orig
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ref = get_synced_models(Refactored)
    with torch.no_grad():
        # Sabotage: Add a massive error value (1.0) to the final output projection layer.
        # This guarantees the predicted embeddings will be radically wrong, forcing the
        # Smooth L1 reconstruction loss to spike compared to the Original model.
        ref.prediction_heads[0].proj_out.weight.add_(1.0)

    b = collect(ref, batch, 0, False)
    del ref

    # Assert that the losses are explicitly NOT equal
    assert not torch.equal(a["loss"], b["loss"]), (
        "The harness failed to catch a massive model divergence!"
    )
