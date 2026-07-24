# Uploading the checkpoints to HuggingFace

Everything is prepared. Uploading needs **your** HF credentials, so you run the
final commands. The scripts never see or store your token.

## 0. One-time setup (on an internet-connected login node)

```bash
pip install -U "huggingface_hub[cli]"   # this env is missing `filelock`
huggingface-cli login                   # paste a WRITE token from https://huggingface.co/settings/tokens
```

## 1. Small models — already built, just push

The 12 small-model repos are already extracted under
`/scratch/mozaffar/patch_release_build` (Qwen-2.5 0.5B, Gemma-3 1B,
LLaMA-3.2 1B PATCH x3 each, plus the 3 MaskLLM baselines). Push them straight
from the login node (reuses the built `mask.npz`, no heavy read):

```bash
cd /scratch/mozaffar/patch
python scripts/release/build_release.py \
    --push --reuse-existing --out /scratch/mozaffar/patch_release_build \
    --only qwen2.5_0.5b-PATCH-45Sparse        # repeat per repo, or drop --only to do all
```

Drop `--only` to push every repo whose `mask.npz` is already built. Add
`--private` if you want to stage them privately first.

## 2. Large models — extract on a compute node, then push

The login node's CPU limit kills the 7B/8B reads, and compute nodes have no
internet. So split it:

```bash
# (a) On a COMPUTE node (sbatch / salloc) — extraction only, no upload:
python scripts/release/build_release.py \
    --out /scratch/mozaffar/patch_release_build \
    --only llama_2_7b-PATCH-45Sparse      # and the other llama_2_7b / llama_3.1_8b repos

# (b) Back on the LOGIN node — push the already-built masks:
python scripts/release/build_release.py \
    --push --reuse-existing --out /scratch/mozaffar/patch_release_build
```

`--reuse-existing` makes step (b) skip the heavy read and upload the `mask.npz`
that step (a) produced.

## Repos that get created (namespace: `mohammad-mozaffari`)

PATCH (25/35/45% each): `qwen2.5_0.5b`, `llama_3.2_1b`, `gemma_3_1b`,
`llama_2_7b`, `llama_2_13b`, `llama_3.1_8b`.
MaskLLM (50%): `gemma_3_1b`, `llama_3.2_1b`, `llama_3.2_3b`.

All 21 repositories were published on 2026-07-24. Note: the earlier
`Llama-2-13b-hf_wanda_2:4_0.5_*.pt` files were one-shot Wanda 2:4 *priors* (50%
dense everywhere) and were NOT released; the released `llama_2_13b-PATCH-*` masks
come from the genuine PATCH-Tile checkpoints
(`Llama-2-13b-hf_LR0.001_REG2.0_..._Sparsity0.5-*.pt`). See `LICENSING.md`.
