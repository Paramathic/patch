"""
Static release metadata for the PATCH / MaskLLM HuggingFace mask checkpoints.

Everything a model card needs lives here:
  * MODELS      : per-model-family identity (base model, license, PATCH variant)
  * RESULTS     : paper numbers (per-task accuracy, average, WikiText2 PPL)
  * HYPERPARAMS : training hyper-parameters per (variant, sparsity)
  * CHECKPOINTS : the release registry mapping each source .pt -> HF repo

All numbers are transcribed from the PATCH paper (arXiv:2509.23410).
We release *only* the binary keep/prune mask, never the weight values.
"""

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #

HF_NAMESPACE = "mohammad-mozaffari"
GITHUB_URL = "https://github.com/Paramathic/patch"
PAPER_URL = "https://arxiv.org/abs/2509.23410"
PAPER_PDF = "https://arxiv.org/pdf/2509.23410"

# Zero-shot task columns, in the order the paper reports them.
TASKS = ["mmlu", "piqa", "arc_e", "arc_c", "winog", "obqa", "race", "hellas"]
TASK_LABELS = {
    "mmlu": "MMLU",
    "piqa": "PIQA",
    "arc_e": "ARC-E",
    "arc_c": "ARC-C",
    "winog": "WinoG.",
    "obqa": "OBQA",
    "race": "RACE",
    "hellas": "HellaS.",
}

# Training setup common to every released checkpoint (paper Section 5 / App. D).
COMMON_TRAINING = {
    "Fine-tuning dataset": "SlimPajama (2B tokens)",
    "Training steps": "2000",
    "Global batch size": "256",
    "Sequence length": "4096",
    "Mask tile size": "128 x 128 (hardware tiles: 128x128 / 128x64 / 64x128 / 64x64)",
    "Logits init.": "N(0, 0.014)",
    "Tile-logit prior": "SparseGPT (strength 3)",
    "Regularization scope": "Global (single target density)",
    "Evaluation": "LM-Eval-Harness (8 zero-shot tasks) + WikiText2 PPL @ seqlen 4096",
    "Hardware": "1 node x 4 GPUs, data parallel (HuggingFace Trainer)",
}

# Per-(variant, sparsity) hyper-parameters (paper Table 16).
# variant in {"PATCH-Joint", "PATCH-Tile", "MaskLLM"}.
HYPERPARAMS = {
    ("PATCH-Joint", "25%"): {"Optimizer": "Adam", "Learning rate": "1e-3",
                             "Gumbel scaling (kappa)": "25 -> 350", "Gumbel temp (tau)": "2 -> 0.05",
                             "Sparsity reg. (lambda1)": "7", "Weight reg. (lambda2)": "10"},
    ("PATCH-Joint", "35%"): {"Optimizer": "Adam", "Learning rate": "1e-3",
                             "Gumbel scaling (kappa)": "25 -> 350", "Gumbel temp (tau)": "2 -> 0.05",
                             "Sparsity reg. (lambda1)": "7", "Weight reg. (lambda2)": "10"},
    ("PATCH-Joint", "45%"): {"Optimizer": "Adam", "Learning rate": "1e-3",
                             "Gumbel scaling (kappa)": "25 -> 350", "Gumbel temp (tau)": "4 -> 0.05",
                             "Sparsity reg. (lambda1)": "7", "Weight reg. (lambda2)": "10"},
    ("PATCH-Tile", "25%"): {"Optimizer": "Adam", "Learning rate": "1e-4",
                            "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                            "Sparsity reg. (lambda1)": "3", "Weight reg. (lambda2)": "0.1"},
    ("PATCH-Tile", "35%"): {"Optimizer": "Adam", "Learning rate": "1e-4",
                            "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                            "Sparsity reg. (lambda1)": "3", "Weight reg. (lambda2)": "0.1"},
    ("PATCH-Tile", "45%"): {"Optimizer": "Adam", "Learning rate": "1e-4",
                            "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                            "Sparsity reg. (lambda1)": "3", "Weight reg. (lambda2)": "0.1"},
    # MaskLLM baselines follow the MaskLLM 2:4 configuration (2:4 logits only).
    ("MaskLLM", "50%"): {"Optimizer": "Adam", "Learning rate": "1e-4 (2:4 logits, MaskLLM config)",
                         "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "4 -> 0.05",
                         "Sparsity pattern": "2:4 (fixed 50%)"},
}

# --------------------------------------------------------------------------- #
# Model-family identity
# --------------------------------------------------------------------------- #
# license: the HuggingFace license identifier declared in the base model repo.
# attribution: extra notice required by the base license (or None).
MODELS = {
    "qwen2.5_0.5b": {
        "display": "Qwen-2.5 0.5B", "base_model": "Qwen/Qwen2.5-0.5B",
        "license": "apache-2.0", "variant": "PATCH-Joint", "attribution": None,
    },
    "llama_3.2_1b": {
        "display": "LLaMA-3.2 1B", "base_model": "meta-llama/Llama-3.2-1B",
        "license": "llama3.2", "variant": "PATCH-Joint",
        "attribution": "Built with Llama. Use governed by the Llama 3.2 Community License.",
    },
    "llama_3.2_3b": {
        "display": "LLaMA-3.2 3B", "base_model": "meta-llama/Llama-3.2-3B",
        "license": "llama3.2", "variant": "MaskLLM",
        "attribution": "Built with Llama. Use governed by the Llama 3.2 Community License.",
    },
    "gemma_3_1b": {
        "display": "Gemma-3 1B", "base_model": "google/gemma-3-1b-pt",
        "license": "gemma", "variant": "PATCH-Joint",
        "attribution": "Gemma is provided under and subject to the Gemma Terms of Use.",
    },
    "llama_2_7b": {
        "display": "LLaMA-2 7B", "base_model": "meta-llama/Llama-2-7b-hf",
        "license": "llama2", "variant": "PATCH-Tile",
        "attribution": "Use governed by the Llama 2 Community License.",
    },
    "llama_3.1_8b": {
        "display": "LLaMA-3.1 8B", "base_model": "meta-llama/Llama-3.1-8B",
        "license": "llama3.1", "variant": "PATCH-Tile",
        "attribution": "Built with Llama. Use governed by the Llama 3.1 Community License.",
    },
    "llama_2_13b": {
        "display": "LLaMA-2 13B", "base_model": "meta-llama/Llama-2-13b-hf",
        "license": "llama2", "variant": "PATCH-Tile",
        "attribution": "Use governed by the Llama 2 Community License.",
    },
}


# --------------------------------------------------------------------------- #
# Paper results (per-task accuracy %, average %, WikiText2 PPL)
# --------------------------------------------------------------------------- #
def _row(sparsity, method, pattern, vals, avg, ppl):
    d = {"sparsity": sparsity, "method": method, "pattern": pattern, "avg": avg, "ppl": ppl}
    d.update(dict(zip(TASKS, vals)))
    return d


# Each entry: list of rows for that model (paper Appendix B per-task tables).
RESULTS = {
    "qwen2.5_0.5b": [  # Table 9 (PATCH-Joint)
        _row("0%",  "Dense",       "-",          [47.71, 70.24, 64.48, 29.52, 56.20, 24.20, 35.02, 40.63], 46.00, 12.08),
        _row("50%", "Magnitude",   "2:4",        [23.00, 54.24, 31.23, 19.20, 49.96, 13.60, 23.44, 26.59], 30.16, 6734.97),
        _row("50%", "Wanda",       "2:4",        [24.43, 58.71, 43.18, 17.75, 51.62, 12.20, 26.32, 29.58], 32.97, 72.48),
        _row("50%", "SparseGPT",   "2:4",        [22.93, 60.77, 46.60, 20.82, 52.88, 14.00, 29.57, 30.93], 34.81, 36.59),
        _row("50%", "Thanos",      "2:4",        [22.97, 60.17, 45.37, 19.20, 53.59, 15.20, 31.00, 31.31], 34.85, 37.32),
        _row("50%", "ProxSparse",  "2:4",        [23.00, 57.34, 40.53, 18.26, 48.62, 14.00, 25.65, 29.02], 32.05, 111.05),
        _row("50%", "MaskLLM",     "2:4",        [25.11, 67.03, 56.57, 23.98, 52.57, 20.20, 33.30, 35.90], 39.33, 15.22),
        _row("45%", "PATCH-Joint", "Dense/2:4",  [27.39, 68.44, 59.13, 25.77, 53.67, 19.80, 32.15, 35.99], 40.29, 14.57),
        _row("35%", "PATCH-Joint", "Dense/2:4",  [29.04, 68.88, 60.40, 26.37, 55.09, 20.40, 32.44, 36.58], 41.15, 13.84),
        _row("25%", "PATCH-Joint", "Dense/2:4",  [30.89, 69.15, 62.79, 29.10, 55.33, 20.00, 34.16, 37.71], 42.39, 13.47),
    ],
    "llama_3.2_1b": [  # Table 13 (PATCH-Joint)
        _row("0%",  "Dense",       "-",          [37.57, 74.54, 65.53, 31.32, 60.62, 26.40, 37.89, 47.76], 47.70, 9.06),
        _row("50%", "Magnitude",   "2:4",        [23.31, 53.81, 27.74, 18.94, 51.38, 11.80, 24.02, 26.26], 29.66, 563.44),
        _row("50%", "Wanda",       "2:4",        [22.90, 58.11, 37.08, 19.20, 49.09, 13.20, 25.17, 28.11], 31.61, 78.18),
        _row("50%", "SparseGPT",   "2:4",        [22.93, 61.43, 45.03, 22.35, 54.93, 15.80, 29.86, 32.08], 35.55, 32.73),
        _row("50%", "Thanos",      "2:4",        [23.12, 62.40, 44.91, 21.76, 54.30, 16.00, 31.10, 32.09], 35.71, 33.03),
        _row("50%", "ProxSparse",  "2:4",        [22.96, 60.83, 39.44, 20.31, 51.54, 16.80, 25.17, 31.37], 33.55, 49.33),
        _row("50%", "MaskLLM",     "2:4",        [26.28, 69.10, 57.41, 25.85, 55.48, 21.40, 32.82, 39.94], 41.04, 12.93),
        _row("45%", "PATCH-Joint", "Dense/2:4",  [23.81, 70.89, 60.77, 27.22, 56.27, 22.80, 34.07, 40.78], 42.08, 12.23),
        _row("35%", "PATCH-Joint", "Dense/2:4",  [25.13, 71.32, 60.27, 29.18, 57.06, 22.00, 34.64, 42.17], 42.72, 11.67),
        _row("25%", "PATCH-Joint", "Dense/2:4",  [28.59, 71.44, 61.57, 28.67, 58.25, 23.20, 35.22, 43.52], 43.81, 11.00),
    ],
    "gemma_3_1b": [  # Table 14 (PATCH-Joint)
        _row("0%",  "Dense",       "-",          [24.95, 75.03, 71.84, 34.90, 58.64, 28.60, 34.83, 47.26], 47.01, 11.67),
        _row("50%", "Magnitude",   "2:4",        [23.08, 59.79, 37.29, 17.66, 50.59, 14.00, 22.87, 27.97], 31.66, 5005.56),
        _row("50%", "Wanda",       "2:4",        [23.96, 59.52, 48.02, 18.34, 51.22, 14.20, 27.85, 30.18], 34.16, 69.41),
        _row("50%", "SparseGPT",   "2:4",        [23.62, 62.79, 49.83, 19.03, 51.54, 15.20, 30.62, 31.99], 35.58, 44.59),
        _row("50%", "Thanos",      "2:4",        [23.44, 62.24, 48.86, 18.34, 50.12, 15.60, 30.81, 31.28], 35.09, 62.63),
        _row("50%", "ProxSparse",  "2:4",        [23.10, 64.25, 50.72, 21.59, 53.43, 18.00, 29.09, 32.86], 36.63, 90.50),
        _row("50%", "MaskLLM",     "2:4",        [25.03, 69.91, 60.27, 27.65, 56.27, 21.20, 34.55, 39.84], 41.84, 12.82),
        _row("45%", "PATCH-Joint", "Dense/2:4",  [23.54, 71.65, 63.97, 27.47, 57.30, 23.60, 33.49, 41.39], 42.80, 11.96),
        _row("35%", "PATCH-Joint", "Dense/2:4",  [25.38, 72.31, 63.80, 27.39, 56.67, 24.00, 34.74, 42.07], 43.30, 11.48),
        _row("25%", "PATCH-Joint", "Dense/2:4",  [25.45, 71.87, 66.16, 30.55, 57.85, 22.80, 34.55, 43.33], 44.07, 11.17),
    ],
    "llama_2_7b": [  # Table 10 (PATCH-Tile)
        _row("0%",  "Dense",      "-",          [41.82, 78.07, 76.35, 43.52, 69.06, 31.40, 39.52, 57.13], 54.61, 5.12),
        _row("50%", "Magnitude",  "2:4",        [25.82, 70.02, 61.78, 30.12, 61.01, 21.80, 31.48, 45.45], 43.44, 54.39),
        _row("50%", "Wanda",      "2:4",        [25.80, 71.00, 63.80, 30.29, 61.09, 25.20, 35.50, 41.75], 44.30, 11.15),
        _row("50%", "SparseGPT",  "2:4",        [26.17, 70.73, 63.80, 30.63, 65.04, 24.00, 37.13, 43.18], 45.09, 10.12),
        _row("50%", "Thanos",     "2:4",        [25.27, 70.78, 63.43, 30.97, 64.56, 23.80, 36.46, 43.11], 44.80, 11.19),
        _row("50%", "ProxSparse", "2:4",        [26.77, 71.60, 65.70, 33.02, 62.90, 24.20, 35.31, 47.84], 45.92, 9.18),
        _row("50%", "MaskLLM",    "2:4",        [27.65, 74.76, 69.44, 35.58, 65.04, 26.80, 38.56, 51.15], 48.62, 6.78),
        _row("45%", "PATCH-Tile", "Dense/2:4",  [27.28, 75.41, 70.16, 35.84, 65.27, 27.60, 38.76, 51.61], 48.99, 6.55),
        _row("35%", "PATCH-Tile", "Dense/2:4",  [29.93, 76.71, 70.88, 36.95, 65.67, 28.20, 39.33, 52.96], 50.08, 6.18),
        _row("25%", "PATCH-Tile", "Dense/2:4",  [32.33, 76.99, 72.81, 38.57, 68.27, 29.80, 39.52, 54.34], 51.58, 5.86),
    ],
    "llama_3.1_8b": [  # Table 12 (PATCH-Tile)
        _row("0%",  "Dense",      "-",          [63.57, 80.09, 81.44, 51.37, 73.48, 33.40, 39.14, 60.02], 60.31, 5.84),
        _row("50%", "Magnitude",  "2:4",        [23.06, 63.82, 45.33, 25.94, 53.91, 15.20, 26.70, 33.49], 35.93, 765.92),
        _row("50%", "Wanda",      "2:4",        [27.85, 68.88, 58.33, 26.71, 60.93, 19.00, 33.78, 38.70], 41.77, 21.29),
        _row("50%", "SparseGPT",  "2:4",        [31.82, 70.46, 63.85, 31.74, 64.56, 21.60, 37.22, 42.99], 45.53, 15.11),
        _row("50%", "Thanos",     "2:4",        [34.23, 70.40, 63.13, 31.40, 63.61, 23.20, 37.03, 42.75], 45.72, 16.09),
        _row("50%", "ProxSparse", "2:4",        [29.89, 71.71, 62.63, 33.28, 58.56, 23.80, 35.22, 46.03], 45.14, 15.17),
        _row("50%", "MaskLLM",    "2:4",        [42.47, 77.04, 73.15, 40.19, 68.43, 28.80, 38.28, 54.04], 52.80, 8.58),
        _row("45%", "PATCH-Tile", "Dense/2:4",  [47.32, 77.96, 73.61, 41.89, 68.03, 29.00, 36.56, 54.44], 53.60, 8.20),
        _row("35%", "PATCH-Tile", "Dense/2:4",  [51.15, 77.97, 76.14, 42.41, 69.46, 31.40, 38.18, 55.54], 55.28, 7.89),
        _row("25%", "PATCH-Tile", "Dense/2:4",  [52.95, 77.75, 77.57, 44.62, 70.56, 31.80, 39.90, 56.69], 56.48, 7.34),
    ],
    "llama_2_13b": [  # Table 11 (PATCH-Tile). MaskLLM row is N/A (no public ckpt).
        _row("0%",  "Dense",      "-",          [52.07, 79.16, 79.42, 48.46, 71.98, 35.40, 40.48, 60.08], 58.38, 4.89),
        _row("50%", "Magnitude",  "2:4",        [27.53, 72.03, 62.46, 32.17, 62.35, 24.20, 36.65, 50.10], 45.94, 8.89),
        _row("50%", "Wanda",      "2:4",        [29.51, 73.01, 68.90, 35.07, 66.93, 24.80, 38.76, 46.61], 47.95, 8.91),
        _row("50%", "SparseGPT",  "2:4",        [33.42, 73.56, 68.60, 36.86, 69.61, 28.00, 39.52, 47.78], 49.67, 8.86),
        _row("50%", "Thanos",     "2:4",        [33.51, 73.50, 68.90, 36.92, 66.90, 28.00, 39.03, 47.86], 49.33, 8.80),
        _row("50%", "ProxSparse", "2:4",        [34.86, 75.68, 71.46, 38.31, 66.85, 28.60, 37.51, 53.09], 50.80, 7.11),
        _row("45%", "PATCH-Tile", "Dense/2:4",  [41.67, 76.88, 73.02, 40.44, 70.24, 30.40, 38.09, 55.16], 53.24, 5.85),
        _row("35%", "PATCH-Tile", "Dense/2:4",  [41.07, 77.75, 75.55, 44.03, 70.80, 31.20, 39.43, 56.95], 54.60, 5.44),
        _row("25%", "PATCH-Tile", "Dense/2:4",  [47.24, 78.13, 76.81, 45.73, 71.19, 34.20, 38.37, 58.76], 56.31, 5.00),
    ],
    # LLaMA-3.2 3B is not evaluated in the paper -> no results table.
    "llama_3.2_3b": [],
}


# --------------------------------------------------------------------------- #
# Release registry: one entry per checkpoint to upload.
# --------------------------------------------------------------------------- #
# source: filename inside the tiled-models directory.
# model : key into MODELS / RESULTS.
# method: "PATCH" or "MaskLLM" (goes into the repo name).
# sparsity: "25%" / "35%" / "45%" / "50%".
CHECKPOINTS = [
    # ---- PATCH: small models (PATCH-Joint) ----
    {"model": "qwen2.5_0.5b", "method": "PATCH", "sparsity": "45%",
     "source": "Qwen2.5-0.5B_LR0.001_REG7.0_OPTadamw_torch_Sparsity0.5-0.45_T4.0-0.05_S100.0-500.0_STR3.0_TT4.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},
    {"model": "qwen2.5_0.5b", "method": "PATCH", "sparsity": "35%",
     "source": "Qwen2.5-0.5B_LR0.001_REG7.0_OPTadamw_torch_Sparsity0.5-0.35_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},
    {"model": "qwen2.5_0.5b", "method": "PATCH", "sparsity": "25%",
     "source": "Qwen2.5-0.5B_LR0.001_REG7.0_OPTadamw_torch_Sparsity0.5-0.25_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},

    {"model": "llama_3.2_1b", "method": "PATCH", "sparsity": "45%",
     "source": "llama-3.2-1B-tiled-0.45.pt"},
    {"model": "llama_3.2_1b", "method": "PATCH", "sparsity": "35%",
     "source": "Llama-3.2-1B_LR0.001_REG7.0_OPTadamw_torch_Sparsity0.5-0.35_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},
    {"model": "llama_3.2_1b", "method": "PATCH", "sparsity": "25%",
     "source": "Llama-3.2-1B_LR0.001_REG7.0_OPTadamw_torch_Sparsity0.5-0.25_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},

    {"model": "gemma_3_1b", "method": "PATCH", "sparsity": "45%",
     "source": "gemma-3-1b-pt_LR0.001_REG7.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.45_T4.0-0.05_S100.0-500.0_STR3.0_TT4.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},
    {"model": "gemma_3_1b", "method": "PATCH", "sparsity": "35%",
     "source": "gemma-3-1b-pt_LR0.001_REG7.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.35_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},
    {"model": "gemma_3_1b", "method": "PATCH", "sparsity": "25%",
     "source": "gemma-3-1b-pt_LR0.001_REG7.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.25_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST25.0-350.0_TSTR3.0_WREG10.0.pt"},

    # ---- PATCH: large models (PATCH-Tile) ----
    {"model": "llama_2_7b", "method": "PATCH", "sparsity": "45%",
     "source": "Llama-2-7b-hf_LR0.0001_REG3.0_OPTadamw_torch_Sparsity0.5-0.45_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},
    {"model": "llama_2_7b", "method": "PATCH", "sparsity": "35%",
     "source": "Llama-2-7b-hf_LR0.0001_REG3.0_OPTadamw_torch_Sparsity0.5-0.35_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},
    {"model": "llama_2_7b", "method": "PATCH", "sparsity": "25%",
     "source": "Llama-2-7b-hf_LR0.0001_REG3.0_OPTadamw_torch_Sparsity0.5-0.25_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},

    {"model": "llama_3.1_8b", "method": "PATCH", "sparsity": "45%",
     "source": "Llama-3.1-8B_LR0.0001_REG3.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.45_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},
    {"model": "llama_3.1_8b", "method": "PATCH", "sparsity": "35%",
     "source": "Llama-3.1-8B_LR0.0001_REG3.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.35_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},
    {"model": "llama_3.1_8b", "method": "PATCH", "sparsity": "25%",
     "source": "Llama-3.1-8B_LR0.0001_REG3.0_OPTadamw_torch_Prune-sparsegpt_Sparsity0.5-0.25_T2.0-0.05_S100.0-500.0_STR3.0_WREG0.1.pt"},

    # ---- PATCH: LLaMA-2 13B (PATCH-Tile). These checkpoints were trained with
    #      different hyper-parameters than the 7B/8B (LR 1e-3, sparse_reg 2,
    #      weight_reg 0.05), so each carries an explicit `hyperparams` override. ----
    {"model": "llama_2_13b", "method": "PATCH", "sparsity": "45%",
     "hyperparams": {"Optimizer": "Adam", "Learning rate": "1e-3",
                     "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                     "Sparsity reg. (lambda1)": "2", "Weight reg. (lambda2)": "0.05"},
     "source": "Llama-2-13b-hf_LR0.001_REG2.0_OPTadamw_torch_Sparsity0.5-0.45_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST100.0-500.0_TSTR3.0_WREG0.05.pt"},
    {"model": "llama_2_13b", "method": "PATCH", "sparsity": "35%",
     "hyperparams": {"Optimizer": "Adam", "Learning rate": "1e-3",
                     "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                     "Sparsity reg. (lambda1)": "2", "Weight reg. (lambda2)": "0.05"},
     "source": "Llama-2-13b-hf_LR0.001_REG2.0_OPTadamw_torch_Sparsity0.5-0.35_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST100.0-500.0_TSTR3.0_WREG0.05.pt"},
    {"model": "llama_2_13b", "method": "PATCH", "sparsity": "25%",
     "hyperparams": {"Optimizer": "Adam", "Learning rate": "1e-3",
                     "Gumbel scaling (kappa)": "100 -> 500", "Gumbel temp (tau)": "2 -> 0.05",
                     "Sparsity reg. (lambda1)": "2", "Weight reg. (lambda2)": "0.05"},
     "source": "Llama-2-13b-hf_LR0.001_REG2.0_OPTadamw_torch_Sparsity0.5-0.25_T4.0-0.05_S100.0-500.0_STR3.0_TT2.0-0.05_ST100.0-500.0_TSTR3.0_WREG0.05.pt"},

    # ---- MaskLLM baselines (2:4, 50%) ----
    {"model": "gemma_3_1b",   "method": "MaskLLM", "sparsity": "50%", "source": "gemma_3_1b_maskllm.pt"},
    {"model": "llama_3.2_1b", "method": "MaskLLM", "sparsity": "50%", "source": "llama_3.2_1b_maskllm.pt"},
    {"model": "llama_3.2_3b", "method": "MaskLLM", "sparsity": "50%", "source": "llama_3.2_3b_maskllm.pt"},
]


def repo_name(entry):
    """Build the HF repo short-name, e.g. gemma_3_1b-PATCH-45Sparse."""
    pct = entry["sparsity"].replace("%", "")
    return f"{entry['model']}-{entry['method']}-{pct}Sparse"


def repo_id(entry):
    return f"{HF_NAMESPACE}/{repo_name(entry)}"


def repo_url(entry):
    return f"https://huggingface.co/{repo_id(entry)}"


def find_result_row(model, method, sparsity):
    """Return the paper result row for (model, method, sparsity), or None."""
    want = "PATCH" if method == "PATCH" else method
    for r in RESULTS.get(model, []):
        rm = r["method"]
        is_patch = rm.startswith("PATCH")
        if r["sparsity"] == sparsity and (
            (want == "PATCH" and is_patch) or (rm == want)
        ):
            return r
    return None
