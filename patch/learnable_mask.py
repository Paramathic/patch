from types import MethodType
import torch
import torch.nn as nn


def _init_normal_(t: torch.Tensor, std: float = 0.014):
    """
    Initialize a tensor with a normal distribution around a mean with a standard deviation

    Args:
        t (torch.Tensor): The tensor to initialize
        std (float): The standard deviation of the normal distribution

    Returns:
        torch.Tensor: The initialized tensor
    """
    return torch.nn.init.normal_(t, mean=0.0, std=std)


def _register_mask_choices(model: nn.Module, dtype):
    """
    Register 6 2:4 patterns once if not present

    Args:
        model (nn.Module): The model to register the mask choices to
        dtype: The data type for the mask choices tensor

    Returns:
        torch.Tensor: The registered mask choices tensor
    """

    if getattr(model, "mask_choices", None) is not None:
        return model.mask_choices
    mc = torch.zeros((6, 4), device=model.device, dtype=dtype)
    mc[0] += torch.tensor([1, 1, 0, 0], device=model.device, dtype=dtype)
    mc[1] += torch.tensor([1, 0, 1, 0], device=model.device, dtype=dtype)
    mc[2] += torch.tensor([1, 0, 0, 1], device=model.device, dtype=dtype)
    mc[3] += torch.tensor([0, 1, 1, 0], device=model.device, dtype=dtype)
    mc[4] += torch.tensor([0, 1, 0, 1], device=model.device, dtype=dtype)
    mc[5] += torch.tensor([0, 0, 1, 1], device=model.device, dtype=dtype)
    model.register_buffer("mask_choices", mc)
    return mc


def _is_linear(layer: nn.Module) -> bool:
    """
    Checks if layer is linear

    Args:
        layer (nn.Module): The layer to check

    Returns:
        bool: True if layer is linear, False otherwise
    """
    return isinstance(layer, nn.Linear)


def _skip_embed_or_lm_head(layer: nn.Linear, tokenizer) -> bool:
    """
    Checks if layer is the lm head

    Args:
        layer (nn.Linear): The layer to check
        tokenizer: The tokenizer to check against

    Returns:
        bool: True if layer is the lm head, False otherwise
    """
    if tokenizer is None:
        return False
    return tokenizer.vocab_size in layer.weight.shape


def _compute_tile_grid(weight: torch.Tensor, row_tile: int, col_tile: int):
    """
    Computes number of tiles in rows and columns given tile shape

    Args:
        weight (torch.Tensor): The weight tensor to compute the tile grid for
        row_tile (int): The number of rows in a tile
        col_tile (int): The number of columns in a tile

    Returns:
        (int, int): The number of row tiles and column tiles
    """

    out_features, in_features = weight.shape
    assert (
        out_features % row_tile == 0 and in_features % col_tile == 0
    ), f"Weight {tuple(weight.shape)} not divisible by tile {row_tile}x{col_tile}"
    return out_features // row_tile, in_features // col_tile


def _apply_tile_prior_(
    layer: nn.Linear, compressed_layer, target_density: float, strength: float
):
    """
    Uses compressed_layer.distribution_scores to bias tile logits toward desired density

    Args:
        layer (nn.Linear): The layer to apply the tile prior to
        compressed_layer: The compressed layer to use for the prior
        target_density (float): The target density for the tile prior
        strength (float): The strength of the tile prior

    Returns:
        None
    """
    if strength <= 0:
        return
    tile_scores = compressed_layer.distribution_scores
    flat = tile_scores.flatten()
    N_total = flat.numel()

    N_dense = round(2 * N_total * (target_density - 0.5))
    topk_idx = torch.topk(flat, k=N_dense, largest=True).indices

    prior = torch.full_like(flat, -1.0)
    prior[topk_idx] = 1.0
    prior = prior.view_as(tile_scores)

    std = torch.std(layer.tile_mask).detach().cpu()
    layer.tile_mask.data += (prior * std * strength).to(layer.tile_mask.device)


def _init_2_4_mask_from_prior(layer, compressed_layer, mask_choices, strength, dtype):
    """
    Transfer learning to 2:4 mask from one shot pruned compressed layers

    Args:
        layer (nn.Linear): The layer to initialize the 2:4 mask for
        compressed_layer: The compressed layer to use for the prior
        mask_choices (torch.Tensor): The mask choices tensor
        strength (float): The strength of the prior
        dtype: The data type for the mask tensor

    Returns:
        None
    """
    with torch.no_grad():
        compressed_mask = (
            compressed_layer.weight.view(-1, 1, 4).ne(0).to(mask_choices.device)
        )
        compressed_mask = compressed_mask.to(dtype)
        priors = (mask_choices * compressed_mask).sum(dim=2)
        layer.mask_2_4.data = (priors - 1) * torch.std(layer.mask_2_4) * strength


def _bind_forward(layer: nn.Linear, masked_forward):
    """
    Binds a forward function to a layer

    Args:
        layer (nn.Linear): The layer to bind the forward function to
        masked_forward: The forward function to bind

    Returns:
        None
    """
    layer.forward = MethodType(masked_forward, layer)


def masked_forward(self, input):
    """
    Forward pass with weights masked before the linear operation

    Args:
        self: The layer to perform the forward pass on
        input: The input tensor

    Returns:
        output: The output tensor
    """
    output = torch.nn.functional.linear(input, self.weight * self.last_mask, self.bias)
    return output


def add_mask_parameters(
    model,
    compressed_model,
    tokenizer,
    mode,
    mask_tile_size,
    dtype,
    prior_strength_2_4,
    prior_strength_tile,
    target_density,
    init_std_2_4: float = 0.014,
    init_std_tile: float = 0.014,
):
    """
    Unifies param wiring for:
      - 2:4-only         (mode="mask_llm")
      - tile-only + fixed 2:4 from compressed (mode="tile")
      - joint tile + 2:4 (mode="joint")

    Creates:
      - layer.mask         : [num_groups, 6] logits for 2:4 (mask_llm/joint) or/and tile logits (tile-only/joint)
      - layer.mask_2_4     : fixed boolean mask from compressed (tile mode)
      - layer.tile_mask    : [num_row_tiles, num_col_tiles] logits (tile/joint)
      - model.mask_choices : [6,4] legal 2:4 patterns (mask_llm/joint)

    Then binds the new forward pass

    Args:
        model: The model to add the mask parameters to
        compressed_model: The compressed model to use for the prior
        tokenizer: The tokenizer to check for lm head skipping
        mode: The mode to use ("mask_llm", "tile", "joint")
        mask_tile_size: The tile size to use for the tile mask
        dtype: The data type for the mask parameters
        prior_strength_2_4: The strength of the 2:4 prior
        prior_strength_tile: The strength of the tile prior
        target_density: The target density for the tile prior
        init_std_2_4: The standard deviation for initializing the 2:4 mask
        init_std_tile: The standard deviation for initializing the tile mask

    Returns:
        None
    """
    assert mode in {"mask_llm", "tile", "joint"}

    with torch.no_grad():
        for model_layer, compressed_layer in zip(
            model.model.modules(), compressed_model.model.modules()
        ):
            if not _is_linear(model_layer):
                continue
            if _skip_embed_or_lm_head(model_layer, tokenizer):
                continue

            W = model_layer.weight
            device = W.device

            if mode == "mask_llm" or mode == "joint":
                # ----- 2:4 logits  -----
                mask_choices = _register_mask_choices(model, dtype)
                num_groups = W.numel() // 4
                logits_2_4 = torch.empty(num_groups, 6, device=device, dtype=dtype)
                _init_normal_(logits_2_4, std=init_std_2_4)
                model_layer.mask_2_4 = torch.nn.Parameter(logits_2_4)
                if prior_strength_2_4 > 0:
                    _init_2_4_mask_from_prior(
                        model_layer,
                        compressed_layer,
                        mask_choices,
                        prior_strength_2_4,
                        dtype,
                    )
                _bind_forward(model_layer, masked_forward)

            if mode == "tile" or mode == "joint":
                # ----- Tile logits  -----
                row_t, col_t = mask_tile_size
                assert (row_t, col_t) != (
                    1,
                    1,
                ), "For tile mode, provide mask_tile_size != (1,1)."
                nrt, nct = _compute_tile_grid(W, row_t, col_t)

                model_layer.tile_row_size = row_t
                model_layer.tile_col_size = col_t

                tile_logits = torch.empty(nrt, nct, device=device, dtype=dtype)
                _init_normal_(tile_logits, std=init_std_tile)
                model_layer.tile_mask = torch.nn.Parameter(tile_logits)  # learnable

                if mode == "tile":
                    fixed_2_4 = compressed_layer.weight.data != 0
                    model_layer.fixed_mask_2_4 = fixed_2_4.to(
                        device=device, dtype=dtype
                    )

                _apply_tile_prior_(
                    model_layer, compressed_layer, target_density, prior_strength_tile
                )

            _bind_forward(model_layer, masked_forward)


def create_2_4_mask(
    mask_2_4_param: torch.Tensor,
    weight_shape: tuple,
    mask_choices: torch.Tensor,
    hard: bool,
    tau: float,
    scaler: float,
) -> torch.Tensor:
    """
    Create a 2:4 sparsity mask from parameters using Gumbel-Softmax.

    Args:
        mask_2_4_param (torch.Tensor): Input tensor of 2:4 mask logits.
        weight_shape (tuple): Shape of the weight tensor to reshape the mask to.
        mask_choices (torch.Tensor): Candidate 2:4 patterns
        hard (bool): If True, return a discrete mask. If False, return soft probabilities.
        tau (float): Temperature for the Gumbel-Softmax.
        scaler (float): Scaling factor applied to logits before sampling.

    Returns:
        (torch.Tensor): Binary (or soft) 2:4 mask tensor with shape `weight_shape`.
    """
    logit = mask_2_4_param * scaler
    gumbel_output = torch.nn.functional.gumbel_softmax(
        logit, tau=tau, hard=hard, dim=-1
    )
    mask = (gumbel_output @ mask_choices).view(weight_shape)

    return mask


def create_tile_mask(
    mask_tile_param: torch.Tensor,
    hard: bool,
    tau: float,
    scaler: float,
    tile_row_size: int,
    tile_col_size: int,
) -> torch.Tensor:
    """
    Create a binary (0/1) mask from a parameter tensor using Gumbel-Softmax,
    expanded to a specified tile size.

    Args:
        mask_tile_param (torch.Tensor): Input tensor of tile mask logits.
        hard (bool): If True, return a discrete mask (0s and 1s). If False, return soft probabilities.
        tau (float): Temperature for the Gumbel-Softmax.
        scaler (float): Scaling factor applied to logits before sampling.
        tile_row_size (int): Number of times to repeat each row.
        tile_col_size (int): Number of times to repeat each column.
            Number of times to repeat each column.

    Returns:
        (torch.Tensor): Binary (or soft) mask tensor.
    """
    if tau <= 0:
        raise ValueError("Temperature `tau` must be positive.")

    zeros = torch.zeros_like(mask_tile_param).unsqueeze(-1)
    params_logit = mask_tile_param.unsqueeze(-1)

    logits = torch.cat([zeros, params_logit], dim=-1) * scaler

    gumbel_output = torch.nn.functional.gumbel_softmax(
        logits, tau=tau, hard=hard, dim=-1
    )

    tile_mask = gumbel_output[..., 1]
    tile_mask = tile_mask.repeat_interleave(tile_row_size, dim=0).repeat_interleave(
        tile_col_size, dim=1
    )

    return tile_mask


def patch_joint(
    mask_2_4_param: torch.Tensor,
    weight_shape: tuple,
    mask_choices: torch.Tensor,
    hard_2_4: bool,
    tau_2_4: float,
    scaler_2_4: float,
    mask_tile_param: torch.Tensor,
    hard_tile: bool,
    tau_tile: float,
    scaler_tile: float,
    tile_row_size: int,
    tile_col_size: int,
) -> torch.Tensor:
    """
    Combine 2:4 sparsity and tile masking into a joint mask.

    Args:
        mask_2_4_param (torch.Tensor): Input tensor of 2:4 mask logits.
        weight_shape (tuple): Shape of the weight tensor to reshape the 2:4 mask to.
        mask_choices (torch.Tensor): Candidate 2:4 patterns.
        hard_2_4 (bool): If True, return a discrete 2:4 mask. If False, return soft probabilities.
        tau_2_4 (float): Temperature for the Gumbel-Softmax for 2:4 mask.
        scaler_2_4 (float): Scaling factor applied to 2:4 logits before sampling.
        mask_tile_param (torch.Tensor): Input tensor of tile mask logits.
        hard_tile (bool): If True, return a discrete tile mask. If False, return soft probabilities.
        tau_tile (float): Temperature for the Gumbel-Softmax for tile mask.
        scaler_tile (float): Scaling factor applied to tile logits before sampling.
        tile_row_size (int): Number of times to repeat each row in the tile mask.
        tile_col_size (int): Number of times to repeat each column in the tile mask.

    Returns:
        (torch.Tensor): Combined mask tensor.
    """
    mask_2_4 = create_2_4_mask(
        mask_2_4_param, weight_shape, mask_choices, hard_2_4, tau_2_4, scaler_2_4
    )

    mask_tile = create_tile_mask(
        mask_tile_param, hard_tile, tau_tile, scaler_tile, tile_row_size, tile_col_size
    )

    return mask_tile + (1 - mask_tile) * mask_2_4


def patch_tile_only(
    fixed_2_4_mask: torch.Tensor,
    mask_tile_param: torch.Tensor,
    hard_tile: bool,
    tau_tile: float,
    scaler_tile: float,
    tile_row_size: int,
    tile_col_size: int,
) -> torch.Tensor:
    """
    Apply tile masking on top of a fixed 2:4 sparsity mask

    Args:
        fixed_2_4_mask (torch.Tensor): Fixed binary 2:4 mask tensor.
        mask_tile_param (torch.Tensor): Input tensor of tile mask logits.
        hard_tile (bool): If True, return a discrete tile mask. If False, return soft probabilities.
        tau_tile (float): Temperature for the Gumbel-Softmax for tile mask.
        scaler_tile (float): Scaling factor applied to tile logits before sampling.
        tile_row_size (int): Number of times to repeat each row in the tile mask.
        tile_col_size (int): Number of times to repeat each column in the tile mask.

    Returns:
        (torch.Tensor): Combined mask tensor.
    """
    mask_tile = create_tile_mask(
        mask_tile_param, hard_tile, tau_tile, scaler_tile, tile_row_size, tile_col_size
    )

    return mask_tile + (1 - mask_tile) * fixed_2_4_mask
