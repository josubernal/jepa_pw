### Code for manuscript: HiT-JEPA: A Hierarchical Self-supervised Trajectory Embedding Framework for Similarity Computation.

Trajsimi is their evaluation on trajectory similarity

# model/
## `baseline_jepa.py`

This file serves as the main execution script, containing the training loop, evaluation logic, and data preparation for the JEPA model.

* **`_collate_hex` and `_collate_test`:** Data collator functions used by the PyTorch `DataLoader` to batch raw trajectories. They handle sequence padding, extract pre-trained spatial embeddings, and format timestamps, adjacency matrices, and Point of Interest (POI) data.


* **`TrajJEPATrainer` (Class):** The primary controller for the training and testing phases.


* `__init__`: Initializes the dataset, dataloader, the `JEPA_base` model, and checkpoint configurations.


* `train`: Executes the main training loop. It feeds batches into the model, computes the losses for the three hierarchical levels, clips gradients, updates the optimizer, and applies the Exponential Moving Average (EMA) to update the target encoders. It also logs performance and plots loss curves.


* `eval` and `test`: Evaluates the trained model on trajectory similarity tasks by testing against varying database sizes, downsampling rates, and trajectory distortion rates.


* `test_merc_seq_to_embs`: A helper function during evaluation that converts batches of raw test trajectories into compressed embedding representations using the trained model.


* `save_checkpoint` / `load_checkpoint`: Manages saving and loading the model's weights to disk.





## `base_model.py`

This file defines the core Transformer-based encoder architectures and the custom embedding modules that process the trajectory sequences.

* **`_no_grad_trunc_normal_` / `trunc_normal_` / `init_weights`:** Utility functions that initialize the neural network weights using a truncated normal distribution.


* **`TransformerEncoderLayer` (Class):** A custom transformer layer utilizing multi-head self-attention and feedforward networks. It supports a customized `blend_gamma` parameter designed to fuse attention weights from previous hierarchical levels.


* **`TransformerEncoder` (Class):** Stacks multiple `TransformerEncoderLayer` modules together and manages passing the detached attention weights sequentially through the layers.


* **`Model` (Class):** Acts as the main wrapper for the Context and Target encoders. It integrates raw grid cell embeddings with learnable spatial, temporal, and user embeddings before routing them through the `TransformerEncoder`.


* **`LearnablePositionalEncoding` / `PositionalEncoding` (Classes):** Modules that inject sequential ordering information into the embeddings, using either learnable parameters or fixed sine/cosine formulas.



## `jepa.py`

This file contains the overarching Joint-Embedding Predictive Architecture (JEPA), orchestrating the multi-level hierarchy, the masking strategies, and the loss calculations.

* **`JEPA_base` (Class):** The master model uniting all sub-components.


* `__init__`: Initializes the 1D UNet, the attention upsampler, three distinct pairs of Context and Target encoders (using the `Model` class), the predictor decoders, and the variance/covariance parameters.


* `mask_tensor`: An optimized masking function that applies either random or successive (block) masking to the sequence embeddings based on a specified mask ratio.


* `target_seg`: Generates the unmasked target representations by passing the full sequence through the target encoders, and defines the masked segments.


* `context_seg`: Generates the masked context representations by dropping sequence elements before they are fed into the context encoders.


* `variance_loss` and `covariance_loss`: Computes VICReg-style regularization penalties to prevent representation collapse in the latent space.


* `loss_fn`: Computes the final objective by combining a Smooth L1 loss (between the predicted and actual target embeddings) with the variance and covariance penalties.


* `interpret` / `interp_all`: Helper functions used during inference to extract specific trajectory embeddings from the hierarchy without executing the predictive loss steps.


* `run_layer`: Orchestrates a single layer of the JEPA process. It extracts targets, applies masks to create context, runs the predictor, and calculates the loss components for that specific hierarchical level.


* `forward`: The main forward pass of the network. It pushes the sequence through the UNet to get three abstraction levels, iteratively calls `run_layer` for each level, upsamples attention maps to pass to the next level, and aggregates the multi-level losses.





## `node2vec_.py`

This file is a standalone utility for pre-training structural graph embeddings.

* **`train_node2vec`:** Sets up and trains a PyTorch Geometric `Node2Vec` model on a provided `edge_index` graph (representing spatial grid cell adjacencies). It learns and saves foundational spatial embeddings (`Config.dataset_embs_file`) which the JEPA model later relies on.



## `unet.py`

This file defines the convolutional networks responsible for constructing the three scales of abstraction.

* **`UNet1DEncoder` (Class):** A 1D Convolutional Neural Network with max-pooling that compresses the input trajectory sequence.


* `forward`: Processes the input through convolutions and pooling to return three distinct outputs, representing the point level, sub-trajectory level, and global abstraction level.




* **`AttentionUpsampler` (Class):** Uses bilinear interpolation to scale up the resolution of attention maps generated at the coarse hierarchical levels, allowing them to be passed as `prev_attn` guidance to the finer, longer sequence levels.