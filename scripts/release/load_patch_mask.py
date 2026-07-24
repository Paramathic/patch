"""
Apply a released PATCH / MaskLLM sparsity mask to a base model.

The released repositories contain ONLY the binary keep/prune mask
(bit-packed in `mask.npz`) - no weight values. To materialize the sparse
model you download the original base model and zero out the pruned weights:

    python load_patch_mask.py \
        --base_model meta-llama/Llama-2-7b-hf \
        --mask_repo mohammad-mozaffari/llama_2_7b-PATCH-45Sparse

or use `apply_patch_mask(...)` from your own code.
"""

import argparse
import json

import numpy as np


def load_mask_npz(npz_path):
    """
    Load `mask.npz` and return {weight_name: bool_ndarray} plus metadata.

    The archive stores, per masked weight:
      * "<name>"          : bit-packed uint8 array (np.packbits of the flat mask)
      * "<name>.__shape__": original 2-D shape
    and a single JSON blob under "__meta__".
    """
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(data["__meta__"])) if "__meta__" in data else {}
    masks = {}
    for key in data.files:
        if key == "__meta__" or key.endswith(".__shape__"):
            continue
        shape = tuple(int(x) for x in data[f"{key}.__shape__"])
        n = int(np.prod(shape))
        flat = np.unpackbits(data[key], count=n).astype(bool)
        masks[key] = flat.reshape(shape)
    return masks, meta


def apply_patch_mask(model, npz_path):
    """Zero out pruned weights of `model` in place using the mask archive."""
    import torch

    masks, meta = load_mask_npz(npz_path)
    sd = dict(model.named_parameters())
    applied = 0
    for name, mask in masks.items():
        if name not in sd:
            raise KeyError(
                f"Mask key '{name}' not found in model parameters. "
                f"Is '{meta.get('base_model', '?')}' the correct base model?"
            )
        w = sd[name]
        m = torch.from_numpy(mask).to(device=w.device)
        if tuple(m.shape) != tuple(w.shape):
            raise ValueError(f"Shape mismatch for {name}: mask {tuple(m.shape)} vs weight {tuple(w.shape)}")
        with torch.no_grad():
            w.mul_(m.to(w.dtype))
        applied += 1
    print(f"Applied mask to {applied} weight tensors "
          f"(target sparsity {meta.get('sparsity', '?')}, pattern {meta.get('pattern', '?')}).")
    return model


def _main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="HF id of the original dense base model")
    p.add_argument("--mask_repo", required=True, help="HF id of the released mask repo")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    import torch

    npz_path = hf_hub_download(repo_id=args.mask_repo, filename="mask.npz")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype)
    )
    apply_patch_mask(model, npz_path)

    total = sum(p.numel() for p in model.parameters())
    nz = sum(int((p != 0).sum()) for p in model.parameters())
    print(f"Overall model density after masking: {nz / total:.4f}")
    return model


if __name__ == "__main__":
    _main()
