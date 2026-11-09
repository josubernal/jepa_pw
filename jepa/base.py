from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from jepa.components import (
    JEPA_Encoder,
    JEPA_Head,
    JEPA_Hierarchy,
    JEPA_Loss,
    JEPA_Mask,
    Level,
)


class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) orchestrator.

    This module coordinates the hierarchical data flow. It remains agnostic to
    how the levels relate (handled by encoders), what the side inputs are,
    or the specific data corruption strategy (handled by the mask).
    """

    def __init__(
        self,
        hierarchy: JEPA_Hierarchy,
        context_encoder: JEPA_Encoder,
        target_encoder: JEPA_Encoder,
        heads: Iterable[JEPA_Head],
        mask: JEPA_Mask,
        loss: JEPA_Loss,
    ) -> None:
        """
        Initializes the JEPA orchestrator and validates architectural alignment.
        """
        super().__init__()

        prediction_heads_list: list[JEPA_Head] = list(heads)
        context_dimensions: list[int] = list(context_encoder.out_dims)

        # Step 1: Validate the presence of at least one hierarchical level
        if not context_dimensions:
            raise ValueError("The architecture must contain at least one level.")

        # Step 2: Ensure dimensionality and layer counts match across context, target, and heads
        target_dimensions: list[int] = list(target_encoder.out_dims)
        if context_dimensions != target_dimensions or len(context_dimensions) != len(
            prediction_heads_list
        ):
            raise ValueError(
                f"Architecture mismatch: context dims {context_dimensions}, target dims "
                f"{target_dimensions}, {len(prediction_heads_list)} heads must agree."
            )

        # Step 3: Assign initialized modules to the orchestrator
        self.hierarchy: JEPA_Hierarchy = hierarchy
        self.context_encoder: JEPA_Encoder = context_encoder
        self.target_encoder: JEPA_Encoder = target_encoder
        self.prediction_heads: nn.ModuleList = nn.ModuleList(prediction_heads_list)
        self.mask: JEPA_Mask = mask
        self.loss_module: JEPA_Loss = loss

        self.num_levels: int = len(context_dimensions)

        # Step 4: Register the dimensions with the loss module for proper metric aggregation
        self.loss_module.build(context_dimensions)

    def train(self, mode: bool = True) -> "JEPA":
        """
        Sets the module to training mode.
        Overrides standard behavior to ensure the target encoder remains in evaluation
        mode, as it only generates ground truth objectives and never tracks gradients.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _levels(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        side_information: dict[str, torch.Tensor | None],
        max_execution_layer: int,
    ) -> list[Level]:
        """
        Generates the hierarchical sequence representations from raw input.
        """
        # Step 1: Extract hierarchical levels up to the maximum requested layer
        generated_levels: list[Level] = self.hierarchy(
            input_embeddings, sequence_lengths
        )
        truncated_levels: list[Level] = generated_levels[
            : min(max_execution_layer, self.num_levels)
        ]

        # Step 2: Inject auxiliary side information into each level's dataclass
        return [replace(level, side=side_information) for level in truncated_levels]

    def forward(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        side_information: dict[str, torch.Tensor | None],
        max_execution_layer: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Executes the main training loop across the required hierarchy.
        """
        hierarchical_levels: list[Level] = self._levels(
            input_embeddings, sequence_lengths, side_information, max_execution_layer
        )

        # 1. Target Pipeline: Process full, unmasked views over all levels.
        # Gradients are disabled because the target encoder only provides ground-truth objectives.
        with torch.no_grad():
            target_views: list[Level] = [
                self.mask.target_view(level) for level in hierarchical_levels
            ]
            target_outputs: Any = self.target_encoder(target_views)

        # 2. Corrupt each level: Generate masking specifications and sample disjoint contexts.
        context_views: list[Level] = []
        masked_target_embeddings: list[torch.Tensor] = []
        mask_specifications: list[Any] = []

        for level_index, level in enumerate(hierarchical_levels):
            with torch.no_grad():
                # Extract the specific embeddings to serve as targets and their locations
                target_emb, mask_spec = self.mask.generate_target(
                    target_outputs.embs[level_index], level.padding_mask, level_index
                )

            # Sample the context that the model is allowed to observe
            context_view: Level = self.mask.generate_context(
                level, mask_spec, level_index
            )

            context_views.append(context_view)
            masked_target_embeddings.append(target_emb)
            mask_specifications.append(mask_spec)

        # 3. Context Pipeline: Process the corrupted views in a single pass.
        # Gradients flow normally here to learn the representations.
        context_outputs: Any = self.context_encoder(context_views)

        # 4. Predictor and Loss Pipeline: Score each level independently.
        level_losses: list[torch.Tensor] = []
        for level_index in range(len(hierarchical_levels)):
            # Predict the missing target embeddings from the encoded context
            level_predictions: torch.Tensor = self.prediction_heads[level_index](
                context_outputs.embs[level_index], mask_specifications[level_index]
            )

            # Assemble the routing dictionary required by the loss module
            prediction_dictionary: dict[str, Any] = {
                "pred": level_predictions,
                "target": masked_target_embeddings[level_index],
                "context_emb": context_outputs.embs[level_index],
                "target_full": target_outputs.embs[level_index],
                "mask": mask_specifications[level_index],
                "aux_context": context_outputs.extras[level_index],
                "aux_target": target_outputs.extras[level_index],
            }

            # Calculate and store the loss for this hierarchical level
            current_level_loss: torch.Tensor = self.loss_module.level_loss(
                prediction_dictionary, level_index
            )
            level_losses.append(current_level_loss)

        # Aggregate the hierarchical losses into a single scalar for backpropagation
        combined_loss: torch.Tensor = self.loss_module.combine(level_losses)
        return combined_loss, level_losses

    def _encode(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        side_information: dict[str, torch.Tensor | None],
        max_execution_layer: int,
    ) -> tuple[list[Level], Any]:
        """
        Internal helper method used during inference to process the sequence entirely
        through the context encoder, bypassing target masking and prediction heads.
        """
        hierarchical_levels: list[Level] = self._levels(
            input_embeddings, sequence_lengths, side_information, max_execution_layer
        )
        encoded_outputs: Any = self.context_encoder(hierarchical_levels)
        return hierarchical_levels, encoded_outputs

    def interpret(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        side_information: dict[str, torch.Tensor | None],
        max_execution_layer: int,
    ) -> torch.Tensor:
        """
        Returns the context embedding from the finest (deepest) executed level
        for use in downstream evaluation tasks.
        """
        _, encoded_outputs = self._encode(
            input_embeddings, sequence_lengths, side_information, max_execution_layer
        )
        return encoded_outputs.embs[-1]

    def interp_all(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        side_information: dict[str, torch.Tensor | None],
        max_execution_layer: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None, list[Any]]:
        """
        Returns a complete inspection of the network's state during inference, yielding
        all intermediate embeddings, padding masks, and the final reconstruction.
        """
        hierarchical_levels, encoded_outputs = self._encode(
            input_embeddings, sequence_lengths, side_information, max_execution_layer
        )

        finest_level_prediction: torch.Tensor | None = None

        # If the context execution reached the deepest layer, reconstruct the full sequence
        if len(encoded_outputs.embs) == self.num_levels:
            finest_level_prediction = self.prediction_heads[-1].predict_all(
                encoded_outputs.embs[-1]
            )

        padding_masks: list[torch.Tensor] = [
            level.padding_mask for level in hierarchical_levels
        ]

        return (
            encoded_outputs.embs,
            padding_masks,
            finest_level_prediction,
            encoded_outputs.extras,
        )
