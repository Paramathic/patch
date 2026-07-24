# Licensing review — releasing PATCH & MaskLLM masks

This note records the licensing check done before publishing the mask-only
checkpoints (University of Toronto academic release). **We distribute only the
binary keep/prune mask — no base-model weight values.**

## Summary

Releasing the masks is permissible. Each mask is treated as a *derivative* of
its base model and published **under that base model's license**, with the
required attribution. There is direct precedent: NVIDIA's MaskLLM project
released LLaMA-2 / LLaMA-3 masks on the HuggingFace Hub the same way.

| Base model | License | Redistribute a derivative? | What we must do |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B | Apache-2.0 | Yes, permissively | Keep the license + NOTICE; state changes. |
| meta-llama/Llama-2-7b-hf | Llama 2 Community License | Yes | Include license copy, "Llama" in the repo name (✓), attribution, respect the >700M-MAU commercial clause + AUP. |
| meta-llama/Llama-3.1-8B | Llama 3.1 Community License | Yes | "Built with Llama" notice, "Llama" in name (✓), include license, AUP. |
| meta-llama/Llama-3.2-1B / 3B | Llama 3.2 Community License | Yes | Same as 3.1. |
| google/gemma-3-1b-pt | Gemma Terms of Use | Yes | Propagate the Gemma Terms + use restrictions, provide notice. |

The mask-generation code (this repo) is **MIT** (see `../../LICENSE`), so we are
free to license our own artifacts; the base-model terms flow through because the
mask is derived from the base model.

## MaskLLM baselines

Our MaskLLM checkpoints (`gemma_3_1b`, `llama_3.2_1b`, `llama_3.2_3b`) were
trained with **our own PyTorch re-implementation** of the MaskLLM method (the
`mask_llm=True` path in this MIT-licensed repo), **not** NVIDIA's code and **not**
NVIDIA's released masks. NVIDIA's "NVIDIA Source Code License for MaskLLM" is
**non-commercial / research-only**, but it binds *their* code and *their*
released mask artifacts — it does **not** attach to masks we trained ourselves.
We therefore release these under the corresponding base-model license and simply
**cite** the MaskLLM paper (Fang et al., NeurIPS 2024). The model cards include
that citation.

## Provenance check (resolved)

✅ **LLaMA-2 7B PATCH 2:4 prior provenance — confirmed correct by the authors
(2026-07-24).** Although `scripts/submit_jobs.sh` references
`MASKLLM_CHECKPOINT="Vinnnf/LLaMA-2-7B-MaskLLM-C4"` (NVIDIA's non-commercial
mask), the released `llama_2_7b-PATCH-*` checkpoints do **not** depend on it —
they use the SparseGPT-derived 2:4 prior, matching the paper. No NVIDIA
non-commercial restriction attaches. The LLaMA-2 13B PATCH checkpoints are
likewise genuine PATCH-Tile masks (hybrid dense/2:4, verified >50% density),
not the earlier Wanda 2:4 priors.

## What ships in each repo

- `mask.npz` — bit-packed binary masks only (no weights).
- `README.md` — model card with `license:` = base-model license + attribution.
- `NOTICE` — states the mask is a derivative under the base license.
- `load_patch_mask.py` — loader that applies the mask to the base model.
- `PATCH-Logo.png`, `PATCH-Pipeline.svg`.
