import torch
import wandb
from slim_local.slim.utils import find_layers
from slim_local.slim.utils import get_layers_list


def compute_model_density_ratio(model, device="cuda:0"):
    """
    Compute the overall mask density (nonzero fraction) across the whole model

    Args:
        model (torch.nn.Module): Model containing linear layers with `last_mask` attributes.
        device (str, optional): Device to perform computation on.

    Returns:
        (torch.Tensor): Scalar tensor representing overall density of the model masks.
    """
    unwrapped_model = model.module if hasattr(model, "module") else model
    nnz = torch.zeros(1, device=device).float()
    numel = 0.0
    for module in unwrapped_model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "last_mask"):
            mask = module.last_mask
            nnz += torch.sum(mask).float().to(device)
            numel += mask.numel()

    return nnz / numel


def log_density_per_transformer_layer(model):
    """
    Log per-transformer-layer mask densities Wandb

    Args:
        model (torch.nn.Module): Model containing layers with `last_mask` attributes.

    Returns:
        None
    """
    with torch.no_grad():
        layers = get_layers_list(model)

        for layer_idx, layer in enumerate(layers):
            nnz = 0
            numel = 0
            subset = find_layers(layer)
            for name in subset:
                mask = subset[name].last_mask
                nnz += torch.sum(mask).float()
                numel += mask.numel()

            density = round((nnz / numel).item(), 2)

            wandb.log({f"Transformer Layer Density/layer{layer_idx}": density})


def compute_layer_density_ratio(model, device="cuda:0"):
    """
    Compute the density ratio (nonzero fraction) of masks in each layer.

    Args:
        model (torch.nn.Module): Model containing layers with `last_mask` attributes.
        device (str, optional): Device to perform computation on

    Returns:
        (torch.Tensor): 1D tensor of layer-wise mask densities, with length equal to the number of layers.
    """
    unwrapped_model = model.module if hasattr(model, "module") else model
    layers = get_layers_list(unwrapped_model)
    layer_densities = torch.zeros(len(layers), device=model.device)

    for layer_idx, layer in enumerate(layers):
        nnz = torch.zeros(1, device=device)
        numel = 0.0
        subset = find_layers(layer)
        for name in subset:
            mask = subset[name].last_mask
            nnz += torch.sum(mask).float().to(device)
            numel += mask.numel()

        layer_densities[layer_idx] = nnz / numel

    return layer_densities


def compute_weight_sum(model, device="cuda:0", mask_llm=False):
    """
    Compute the ratio of masked weight norm to total weight norm.

    Args:
        model (torch.nn.Module): Model containing linear layers with `last_mask` attributes.
        device (str, optional): Device to perform computation on
        mask_llm (bool, optional):
            If True, only compute the masked weight norm
            If False, normalize by the full weight norm.

    Returns:
        (torch.Tensor): Scalar tensor representing the ratio of masked-to-total norm,
            or masked norm alone if `mask_llm=True`.
    """
    unwrapped_model = model.module if hasattr(model, "module") else model
    weight_sum = torch.zeros(1, device=device).float()
    total = torch.zeros(1, device=device).float()

    for module in unwrapped_model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "last_mask"):
            mask = module.last_mask

            weight = module.weight.detach()
            weight_sum += torch.norm(mask * weight).to(device)
            if not mask_llm:
                total += torch.norm(weight).to(device)
    if not mask_llm:
        return weight_sum / total
    return weight_sum
