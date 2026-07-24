#!/usr/bin/env python3
"""
Build and (optionally) upload the mask-only PATCH / MaskLLM checkpoints to the
HuggingFace Hub.

For every checkpoint in ``patch_release_data.CHECKPOINTS`` the script:

  1. Loads the source ``.pt`` state-dict (masked weights, i.e. pruned entries
     are exactly 0), and extracts the binary keep/prune mask
     ``(weight != 0)`` for every decoder Linear weight. **No weight values are
     ever read out or written** - only the boolean mask.
  2. Bit-packs the masks and saves them to ``mask.npz`` (numpy
     ``savez_compressed``), MaskLLM-style.
  3. Renders a model card (``README.md``) with the PATCH logo + pipeline
     figure, the paper's accuracy / PPL results, the training hyper-parameters,
     usage instructions, licensing and citation.
  4. Copies the loader helper, the images and a NOTICE file into the repo
     folder.
  5. With ``--push`` (and a HF token) creates the repo and uploads everything.

By default the script runs in **dry-run** mode: it writes a full local preview
of every repo under ``--out`` and does NOT touch the Hub.

Examples
--------
    # Preview one repo locally (fast, smallest checkpoint):
    python build_release.py --only qwen2.5_0.5b-PATCH-45Sparse

    # Preview all cards WITHOUT extracting the (large) masks:
    python build_release.py --cards-only

    # Actually publish everything:
    HF_TOKEN=hf_xxx python build_release.py --push
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_release_data as D  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
DEFAULT_TILED_DIR = "/scratch/mozaffar/tiled_models"


# --------------------------------------------------------------------------- #
# Mask extraction
# --------------------------------------------------------------------------- #
def _is_maskable(name, tensor):
    """A decoder Linear weight that PATCH/MaskLLM prunes."""
    return (
        name.endswith(".weight")
        and tensor.dim() == 2
        and ".layers." in name
    )


def extract_masks(pt_path):
    """
    Return ({name: bit-packed uint8 ndarray}, {name: shape}, stats).

    Only the sign of "is this entry zero" is ever read - weight magnitudes
    are discarded, so the artifact leaks no weights.
    """
    sd = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    packed, shapes = {}, {}
    total = kept = 0
    for name, t in sd.items():
        if not _is_maskable(name, t):
            continue
        mask = (t != 0).contiguous().view(-1).numpy()
        packed[name] = np.packbits(mask)
        shapes[name] = tuple(int(x) for x in t.shape)
        total += mask.size
        kept += int(mask.sum())
    del sd
    stats = {
        "num_layers": len(packed),
        "params": total,
        "kept": kept,
        "measured_sparsity": round(1.0 - kept / total, 4) if total else None,
    }
    return packed, shapes, stats


def save_mask_npz(out_path, packed, shapes, meta):
    arrays = {"__meta__": np.array(json.dumps(meta))}
    for name, arr in packed.items():
        arrays[name] = arr
        arrays[f"{name}.__shape__"] = np.array(shapes[name], dtype=np.int64)
    np.savez_compressed(out_path, **arrays)


# --------------------------------------------------------------------------- #
# Model-card rendering
# --------------------------------------------------------------------------- #
def _md_results_table(model_key, highlight):
    """Average accuracy + PPL comparison table for a model, highlighting a row."""
    rows = D.RESULTS.get(model_key, [])
    if not rows:
        return "_This model is not evaluated in the paper; no benchmark numbers are available._"
    lines = [
        "| Sparsity | Method | Pattern | Avg Acc (% ↑) | WikiText2 PPL (↓) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        is_hl = (r["sparsity"] == highlight["sparsity"]
                 and ((highlight["method"] == "PATCH" and r["method"].startswith("PATCH"))
                      or r["method"] == highlight["method"]))
        method = f"**{r['method']}**" if is_hl else r["method"]
        acc = f"**{r['avg']:.2f}**" if is_hl else f"{r['avg']:.2f}"
        ppl = f"**{r['ppl']:.2f}**" if is_hl else f"{r['ppl']:.2f}"
        star = " ⭐" if is_hl else ""
        lines.append(f"| {r['sparsity']} | {method}{star} | {r['pattern']} | {acc} | {ppl} |")
    return "\n".join(lines)


def _md_pertask_row(model_key, highlight):
    r = D.find_result_row(model_key, highlight["method"], highlight["sparsity"])
    if r is None:
        return ""
    header = "| " + " | ".join(D.TASK_LABELS[t] for t in D.TASKS) + " | **Average** |"
    sep = "|" + "---|" * (len(D.TASKS) + 1)
    vals = "| " + " | ".join(f"{r[t]:.2f}" for t in D.TASKS) + f" | **{r['avg']:.2f}** |"
    return f"Per-task zero-shot accuracy (%) for this checkpoint:\n\n{header}\n{sep}\n{vals}"


def _md_hyperparams(variant, sparsity, override=None):
    hp = override if override else D.HYPERPARAMS.get((variant, sparsity), {})
    lines = ["| Hyper-parameter | Value |", "|---|---|"]
    for k, v in D.COMMON_TRAINING.items():
        lines.append(f"| {k} | {v} |")
    for k, v in hp.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def render_card(entry, stats=None):
    m = D.MODELS[entry["model"]]
    variant = "MaskLLM" if entry["method"] == "MaskLLM" else m["variant"]
    row = D.find_result_row(entry["model"], entry["method"], entry["sparsity"])
    name = D.repo_name(entry)

    # YAML frontmatter
    fm = [
        "---",
        f"base_model:\n- {m['base_model']}",
        "library_name: transformers",
        f"license: {m['license']}",
        "tags:\n- pruning\n- sparsity\n- 2:4-sparsity\n- patch\n- maskllm\n- mask",
        "pipeline_tag: text-generation",
        "---",
    ]

    headline = ""
    if row is not None:
        headline = (
            f"> **This checkpoint** ({m['display']}, {variant}, "
            f"{entry['sparsity']} sparsity): "
            f"**{row['avg']:.2f}%** average zero-shot accuracy, "
            f"**{row['ppl']:.2f}** WikiText2 perplexity.\n"
        )
    else:
        headline = (
            f"> **This checkpoint**: {m['display']}, {variant}, {entry['sparsity']} "
            f"(2:4) sparsity. This model is not evaluated in the PATCH paper, so no "
            f"benchmark numbers are reported here.\n"
        )

    method_blurb = (
        "PATCH (Pruning with a Learnable Tile-level Configuration for Hybrid "
        "Sparsity) learns a structured mask on **frozen** pretrained weights, "
        "assigning each tile as dense (0% sparsity) or 2:4 sparse (50% sparsity) "
        "to hit a flexible global sparsity target while staying hardware-friendly."
        if entry["method"] == "PATCH" else
        "This is a **MaskLLM** 2:4 baseline, trained with our re-implementation of "
        "MaskLLM (learnable semi-structured 2:4 sparsity via Gumbel-Softmax over "
        "frozen weights). It is released as a baseline for the PATCH paper."
    )

    meas = ""
    if stats and stats.get("measured_sparsity") is not None:
        meas = f"\n- Measured mask sparsity: **{stats['measured_sparsity'] * 100:.2f}%**"
        if stats.get("num_layers") and stats.get("params"):
            meas += f" over {stats['num_layers']} Linear layers ({stats['params']:,} weights)"
        meas += "."

    pertask = _md_pertask_row(entry["model"], entry) if row is not None else ""
    pertask_block = f"\n\n{pertask}\n" if pertask else "\n"

    attribution = f"\n\n> {m['attribution']}" if m["attribution"] else ""

    card = f"""{os.linesep.join(fm)}

<div align="center">
<img src="./PATCH-Logo.png" alt="PATCH" width="360">
</div>

# {name}

{headline}
[![Paper](https://img.shields.io/badge/arXiv-2509.23410-b31b1b.svg)]({D.PAPER_URL}) [![GitHub](https://img.shields.io/badge/GitHub-Paramathic%2Fpatch-black.svg?logo=github)]({D.GITHUB_URL})

This repository hosts a **mask only** release for the paper
**[PATCH: Learnable Tile-level Hybrid Sparsity for LLMs]({D.PAPER_URL})**.
{method_blurb}

Because PATCH/MaskLLM keep the base weights **frozen**, we distribute *only the
binary keep/prune mask* (bit-packed in `mask.npz`) - **no weight values**. You
recover the sparse model by downloading the original base model and applying the
mask.

- Base model: [`{m['base_model']}`](https://huggingface.co/{m['base_model']})
- Method: **{variant}** &nbsp;|&nbsp; Target sparsity: **{entry['sparsity']}** &nbsp;|&nbsp; Pattern: **{'2:4' if entry['method'] == 'MaskLLM' else 'Dense / 2:4 tiles'}**{meas}

<div align="center">
<img src="./PATCH-Pipeline.svg" alt="PATCH pipeline" width="760">
</div>

## Results ({m['display']})

{_md_results_table(entry['model'], entry)}
{pertask_block}
All numbers are from the PATCH paper ([arXiv:2509.23410]({D.PAPER_URL})); accuracy
is the average over MMLU, PIQA, ARC-Easy, ARC-Challenge, Winogrande, OpenBookQA,
RACE and HellaSwag, evaluated with the LM-Evaluation-Harness. PPL is WikiText2.

## Training hyper-parameters

{_md_hyperparams(variant, entry['sparsity'], entry.get('hyperparams'))}

## How to use

```python
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import torch
from load_patch_mask import apply_patch_mask  # shipped in this repo

npz = hf_hub_download(repo_id="{D.repo_id(entry)}", filename="mask.npz")
model = AutoModelForCausalLM.from_pretrained("{m['base_model']}", torch_dtype=torch.bfloat16)
apply_patch_mask(model, npz)   # zeroes the pruned weights in place
```

Or from the command line:

```bash
python load_patch_mask.py --base_model {m['base_model']} --mask_repo {D.repo_id(entry)}
```

Speedup on real hardware requires a 2:4-aware / hybrid sparse kernel; see the
[GitHub repository]({D.GITHUB_URL}) and [STOICC](https://github.com/Paramathic/stoicc).

## License

The released mask is a derivative of the base model and is distributed under the
base model's license (**`{m['license']}`**). You must comply with that license and
obtain access to the base model separately.{attribution}

The mask-generation code is released under the MIT license (see the
[PATCH repository]({D.GITHUB_URL})).

## Citation

```bibtex
@article{{hourri2025patch,
    title  = {{PATCH: Learnable Tile-level Hybrid Sparsity for LLMs}},
    author = {{Hourri, Younes and Mozaffari, Mohammad and Mehri Dehnavi, Maryam}},
    year   = 2025,
    journal = {{arXiv preprint arXiv:2509.23410}}
}}
```
{"" if entry['method'] != 'MaskLLM' else '''
This checkpoint is a MaskLLM baseline; please also cite MaskLLM:

```bibtex
@inproceedings{fang2024maskllm,
    title     = {MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models},
    author    = {Fang, Gongfan and Yin, Hongxu and Muralidharan, Saurav and Heinrich, Greg and Pool, Jeff and Kautz, Jan and Molchanov, Pavlo and Wang, Xinchao},
    booktitle = {NeurIPS},
    year      = {2024}
}
```
'''}"""
    return card


NOTICE_TEXT = """This repository distributes ONLY a binary pruning mask (which weight
positions are kept vs. zeroed). It contains no weights from the base model.

The mask is a derivative of the base model listed in the model card and is
provided under that base model's license. You must independently obtain the base
model and comply with its license and acceptable-use policy.

Mask-generation code: MIT (https://github.com/Paramathic/patch).
MaskLLM baselines are our own re-implementation; the MaskLLM method is due to
Fang et al. (NVIDIA), NeurIPS 2024.
"""


# --------------------------------------------------------------------------- #
# Per-repo build + push
# --------------------------------------------------------------------------- #
def build_repo(entry, out_dir, tiled_dir, cards_only, reuse_existing=False):
    name = D.repo_name(entry)
    repo_dir = os.path.join(out_dir, name)
    os.makedirs(repo_dir, exist_ok=True)

    stats = None
    npz_path = os.path.join(repo_dir, "mask.npz")
    if not cards_only and reuse_existing and os.path.exists(npz_path):
        # Two-phase flow: mask.npz already extracted (e.g. on a compute node);
        # just (re)render the card + shipped files and skip the heavy read.
        try:
            meta = json.loads(str(np.load(npz_path, allow_pickle=False)["__meta__"]))
            stats = {"measured_sparsity": meta.get("measured_sparsity"),
                     "num_layers": None, "params": None, "kept": None}
        except Exception:
            stats = None
        print("  reusing existing mask.npz (skipping extraction)")
    elif not cards_only:
        src = os.path.join(tiled_dir, entry["source"])
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source checkpoint not found: {src}")
        packed, shapes, stats = extract_masks(src)
        meta = {
            "repo": name,
            "base_model": D.MODELS[entry["model"]]["base_model"],
            "method": entry["method"],
            "variant": "MaskLLM" if entry["method"] == "MaskLLM" else D.MODELS[entry["model"]]["variant"],
            "sparsity": entry["sparsity"],
            "pattern": "2:4" if entry["method"] == "MaskLLM" else "Dense/2:4 tiles",
            "format": "bit-packed uint8 (np.packbits); unpack with np.unpackbits",
            "measured_sparsity": stats["measured_sparsity"],
        }
        save_mask_npz(os.path.join(repo_dir, "mask.npz"), packed, shapes, meta)

    # README + shipped files
    with open(os.path.join(repo_dir, "README.md"), "w") as f:
        f.write(render_card(entry, stats))
    with open(os.path.join(repo_dir, "NOTICE"), "w") as f:
        f.write(NOTICE_TEXT)
    shutil.copy(os.path.join(os.path.dirname(__file__), "load_patch_mask.py"),
                os.path.join(repo_dir, "load_patch_mask.py"))
    for img in ("PATCH-Logo.png", "PATCH-Pipeline.svg"):
        srcimg = os.path.join(ASSETS_DIR, img)
        if os.path.exists(srcimg):
            shutil.copy(srcimg, os.path.join(repo_dir, img))
    return repo_dir, stats


def push_repo(entry, repo_dir, token, private):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            f"Pushing needs a working `huggingface_hub` install ({e}). "
            "Install it with:  pip install -U 'huggingface_hub[cli]'"
        )

    api = HfApi(token=token)
    rid = D.repo_id(entry)
    api.create_repo(rid, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=repo_dir, repo_id=rid, repo_type="model",
                      commit_message=f"Release {D.repo_name(entry)} (mask only)")
    print(f"  pushed -> {D.repo_url(entry)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiled_dir", default=DEFAULT_TILED_DIR,
                    help="Directory holding the source .pt checkpoints")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "release_preview"),
                    help="Where to write the local per-repo preview")
    ap.add_argument("--only", default=None,
                    help="Only build this repo short-name (e.g. gemma_3_1b-PATCH-45Sparse)")
    ap.add_argument("--cards-only", action="store_true",
                    help="Only render README/assets; skip (slow) mask extraction")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="Reuse an already-extracted mask.npz in --out and skip the "
                         "heavy read. Enables a two-phase flow: extract masks on a "
                         "compute node, then push from an internet-connected node.")
    ap.add_argument("--push", action="store_true",
                    help="Create + upload the repos to the Hub (needs a token)")
    ap.add_argument("--private", action="store_true",
                    help="Create the HF repos as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (defaults to $HF_TOKEN)")
    args = ap.parse_args()

    if args.push and not args.token:
        ap.error("--push requires a token via --token or $HF_TOKEN")

    entries = D.CHECKPOINTS
    if args.only:
        entries = [e for e in entries if D.repo_name(e) == args.only]
        if not entries:
            ap.error(f"No checkpoint named {args.only!r}. "
                     f"Options:\n  " + "\n  ".join(D.repo_name(e) for e in D.CHECKPOINTS))

    print(f"{'PUSH' if args.push else 'DRY-RUN'}: {len(entries)} checkpoint(s) -> "
          f"{'the Hub' if args.push else args.out}\n")

    for e in entries:
        print(f"[{D.repo_name(e)}]")
        repo_dir, stats = build_repo(e, args.out, args.tiled_dir, args.cards_only,
                                     reuse_existing=args.reuse_existing)
        if stats:
            print(f"  mask.npz: {stats['num_layers']} layers, "
                  f"measured sparsity {stats['measured_sparsity']*100:.2f}%")
        print(f"  built -> {repo_dir}")
        if args.push:
            push_repo(e, repo_dir, args.token, args.private)
    print("\nDone.")


if __name__ == "__main__":
    main()
