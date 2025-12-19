<div align="center">
<img src="./assets/LEAP-Logo.png" alt="LEAP" width="400">  
</div>

# LEAP: Learnable End-to-End Unstructured Sparsity for LLMs


LEAP optimizes large language models (LLMs) by learning a fully unstructured, per-weight sparsity mask end-to-end on frozen weights, enabling high-accuracy compression without layer-wise pruning heuristics.


**LEAP: Learnable End-to-End Unstructured Sparsity for LLM**

*Mohammad Mozaffari¹ and Younes Hourri¹*

- *¹Equal contribution*

[Blog Post](https://www.cs.toronto.edu/~mmozaffari/compression-trinity/leap/index.html)



## Setup

To clone the repository, run the following command:

```
git clone --branch leap --recurse-submodules https://github.com/Paramathic/patch.git
```

The `--recurse-submodules` flag is used to clone the [SLiM repository](https://github.com/Paramathic/slim/tree/main) as a submodule. The SLiM repository is located in the `slim_local` directory.

The list of requirements can be found in the `requirements.txt` file. To install the requirements, run the following command:

```bash 
pip install -r requirements.txt
```

## Quick Start


**Adding `slim_local` to Python Path: Before running the code, `slim_local` should be added to the python path. This can be done by running the following command inside the python script:

``` python
import os
import sys

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'slim' directory
slim_path = os.path.join(script_dir, "slim_local")

# Add the 'slim' directory to the Python path
if slim_path not in sys.path:
    sys.path.insert(0, slim_path)
```

**Model and Tokenizer Instantiation:** Our code base supports models from HuggingFace's transformers library. In this example, we use the OPT-125M model from [facebook/opt-125m](https://huggingface.co/facebook/opt-125m).

```python
from transformers import AutoTokenizer
from slim_local.utils.model import get_llm

model_name = "facebook/opt-125m" 

model, lm_eval_model = get_llm(
    model_name=model_name,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
)
```

The `lm_eval_model` is a wrapper around the model that provides a simple interface for evaluating the model on language modeling tasks. It is used in the evaluation scripts.

**Sparse Mask Generation**: We use the `prepare_pruned_model` function to initialize the sparse mask for LEAP using a one-shot unstructured pruning method as a starting point. This function takes the model, the desired sparsity ratio, and the tile size as input and returns the pruned model. 

If `checkpoint_name` exists, it loads the mask from the checkpoint. Otherwise, it generates a new mask and saves it to the checkpoint.

`one_shot_args` is a dictionary that contains the arguments for the one-shot pruning method. In this example, we use the Wanda method with unstructured sparsity pattern and 60% sparsity ratio. More details about the arguments can be found in the *Function Documentation* section.


```python
from patch.utils import prepare_pruned_model

one_shot_args = {
    "prune_method": "wanda",
    "sparsity_type": "unstructured",
    "sparsity_ratio": 0.6,
    "nsamples": 128,
    "maskllm_checkpoint": None,
    "optimizer_FT_pruning": "adamw_torch",
    "calibration_dataset": "c4",
    "eval_dataset": "wikitext2",
    "shift_zero_metrics": False,
    "fine_tune": False,
}

target_sparsity_ratio = 0.6
mask_tile_size = (1, 1) 


compressed_model = prepare_pruned_model(
    model,
    tokenizer,
    checkpoint_name,
    prune_args=one_shot_args,
    mask_tile_size=mask_tile_size,
    target_sparsity_ratio=target_sparsity_ratio,
)
```

**LEAP Training:** After generating the sparse mask, the model is ready for training with LEAP. `mask_args` is a dictionary that contains the arguments for the LEAP training. We use a tile size of (1, 1) and a target sparsity ratio of 60%. More details about the arguments can be found in the *Function Documentation* section.


```python
from patch.fine_tune_mask import learn_mask

learnable_args = {
    "learnable_mask": True,
    "mask_tile_size": mask_tile_size,
    "grad_checkpoint": True,
    "local_bs": 1,
    "optimizer": "adamw_torch",
    "fine_tuning_sequence_length": 4096,
    "target_sparsity_ratio": target_sparsity_ratio,
    "lr": 1e-3,
    "sparse_reg": 7,
    "weight_reg": 10.0,
    "joint_optim": False,
    "temp_schedule_tile": [4.0, 0.05],
    "scaler_schedule_tile": [25.0, 350.0],
    "hard_tile": False,
    "prior_strength_tile": 3.0,
    "mask_llm": False,
    "layer_target": False,
    "unstructured": True # enables LEAP: per-weight unstructured mask learning
}

model, lm_eval_model = learn_mask(
    model_name=model_name,
    compressed_model=compressed_model,
    tokenizer=tokenizer,
    mask_args=learnable_args,
)
```

**Evaluation:** After training, the model can be evaluated using the `evaluate` function. This function takes the model, tokenizer, and evaluation arguments as input and returns the evaluation results.

```python
from patch.utils import evaluate

ppl_test, lmharness_results = evaluate(
    model,
    lm_eval_model,
    tokenizer,
    evaluate_perplexity=True,
    eval_dataset="wikitext2",
    eval_batch_size=1,
    test_lmharness=True,
)
```

For a more automated script to run PATCH on SLURM clusters, please refer to the `scripts/submit_jobs.sh` script.

## Experimental Results

We evaluate LEAP on a range of transformer models from 0.5B to 3B parameters, including Qwen-2.5, LLaMA-3, and Gemma-3 families. Models are trained on the SlimPajama dataset for 2000 steps with batch size 128 and sequence length 4096. Evaluation includes average accuracy across eight zero-shot tasks (PIQA, ARC-Easy, ARC-Challenge, Winogrande, OpenBookQA, , MMLU) and perplexity (PPL) on WikiText2.

### Comparative Results at 50% Sparsity Ratio

| Model | Method | Sparsity Ratio | PPL | MMLU | PIQA | ARC-E | ARC-C | Wino. | OBQA | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-3 1B** | Dense | 0% | 14.17 | 24.95 | 74.81 | 71.93 | 35.41 | 58.72 | 28.80 | 49.10 |
| | Best* | 50% | 26.63 | 24.05 | 70.08 | 63.80 | 27.73 | 56.51 | 25.20 | 44.56 |
| | LEAP | 50% | 11.89 | 23.80 | 71.27 | 63.13 | 27.90 | 60.03 | 23.20 | 44.93 |
| **LLaMA-3.2 1B** | Dense | 0% | 9.75 | 36.92 | 74.27 | 65.53 | 31.31 | 60.30 | 26.20 | 49.09 |
| | Best* | 50% | 17.35 | 27.92 | 69.75 | 56.06 | 26.37 | 56.04 | 22.20 | 43.05 |
| | LEAP | 50% | 11.29 | 27.39 | 71.98 | 60.90 | 28.24 | 57.53 | 22.80 | 44.81 |
| **LLaMA-3.2 3B** | Dense | 0% | 7.81 | 54.13 | 76.55 | 74.28 | 42.75 | 69.38 | 30.60 | 57.95 |
| | Best* | 50% | 11.61 | 42.90 | 74.48 | 66.62 | 34.56 | 66.93 | 28.20 | 52.28 |
| | LEAP | 50% | 8.67 | 44.55 | 75.19 | 71.09 | 39.33 | 67.48 | 29.00 | 54.44 |
| **Qwen-2.5 0.5B** | Dense | 0% | 13.08 | 47.36 | 69.97 | 64.18 | 29.18 | 55.80 | 24.40 | 48.48 |
| | Best* | 50% | 19.70 | 29.38 | 65.67 | 55.89 | 25.26 | 56.27 | 21.20 | 42.28 |
| | LEAP | 50% | 14.09 | 34.37 | 68.44 | 61.99 | 26.11 | 55.25 | 21.00 | 44.53 |

---
**\*Note on Best Method:** The "Best*" rows report the results for **ADMM**, which was the highest-performing baseline among Wanda, SparseGPT, Thanos, and ADMM across all evaluated models.

### Comparative Results at 60% Sparsity Ratio

| Model | Method | Sparsity Ratio | PPL | MMLU | PIQA | ARC-E | ARC-C | Wino. | OBQA | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-3 1B** | Dense | 0% | 14.17 | 24.95 | 74.81 | 71.93 | 35.41 | 58.72 | 28.80 | 49.10 |
| | Best* | 60% | 50.55 | 25.16 | 65.29 | 55.89 | 22.69 | 53.82 | 19.60 | 40.41 |
| | LEAP | 60% | 13.16 | 24.44 | 68.66 | 60.61 | 25.09 | 58.64 | 23.00 | 43.41 |
| **LLaMA-3.2 1B** | Dense | 0% | 9.75 | 36.92 | 74.27 | 65.53 | 31.31 | 60.30 | 26.20 | 49.09 |
| | Best* | 60% | 33.87 | 25.77 | 64.15 | 47.22 | 22.44 | 54.62 | 18.20 | 38.73 |
| | LEAP | 60% | 13.06 | 24.90 | 69.70 | 57.53 | 25.60 | 55.80 | 21.00 | 42.42 |
| **LLaMA-3.2 3B** | Dense | 0% | 7.81 | 54.13 | 76.55 | 74.28 | 42.75 | 69.38 | 30.60 | 57.95 |
| | Best* | 60% | 19.14 | 33.46 | 69.15 | 57.70 | 27.39 | 59.82 | 22.40 | 44.99 |
| | LEAP | 60% | 9.77 | 37.55 | 74.32 | 66.50 | 34.81 | 63.14 | 26.00 | 50.39 |
| **Qwen-2.5 0.5B** | Dense | 0% | 13.08 | 47.36 | 69.97 | 64.18 | 29.18 | 55.80 | 24.40 | 48.48 |
| | Best* | 60% | 33.41 | 24.22 | 62.40 | 50.13 | 22.44 | 52.96 | 17.80 | 38.33 |
| | LEAP | 60% | 15.66 | 24.66 | 67.90 | 56.69 | 25.09 | 55.49 | 19.40 | 41.53 |

---
**\*Note on Best Method:** The "Best*" rows report the results for **ADMM**, which was the highest-performing baseline among Wanda, SparseGPT, Thanos, and ADMM across all evaluated models.


## Function Documentation

### patch.utils.prepare_pruned_model
- `model`: The model to be pruned/compressed.
- `tokenizer`: The tokenizer associated with the model.
- `checkpoint_name`: Path to save/load the pruned model checkpoint.
- `mask_tile_size`: Tile size for mask parameters as (row_tile_size, col_tile_size).
- `target_sparsity_ratio`: Target sparsity ratio for the unstructured reference model.
- `seed`: Random seed for reproducibility.
- `prune_args`: Arguments for pruning and quantization in dictionary or arguments format. The dictionary should contain the following keys:
  - `prune_method`: Pruning method to use. Options: `magnitude`, `wanda`, `sparsegpt`, `thanos`, `prox_sparse`.
  - `sparsity_type`: Sparsity pattern to use. Options: `unstructured`, `2:4`.
  - `sparsity_ratio`: Sparsity ratio to achieve (0 < ratio < 1).
  - `nsamples`: Number of samples for data-dependent methods (e.g., Wanda, SparseGPT).
  - `maskllm_checkpoint`: Path to MaskLLM checkpoint if using MaskLLM.
  - `optimizer_FT_pruning`: Optimizer for fine-tuning during pruning. Options: `adamw_torch`, `adamw_apex`.
  - `calibration_dataset`: Dataset for calibration. Options: `c4`, `wikitext2`.
  - `eval_dataset`: Dataset for evaluation. Options: `wikitext2`, `ptb`.
  - `shift_zero_metrics`: Whether to shift zero metrics.
  - `fine_tune`: Whether to fine-tune the model after pruning.

### patch.fine_tune_mask.learn_mask
- `model_name`: Name of the model to be fine-tuned.
- `local_files_only`: Whether to load the model from local files only.
- `compressed_model`: The pruned/compressed model to be fine-tuned.
- `tokenizer`: The tokenizer associated with the model.
- `local_files_only`: Whether to load the model from local files only.
- `hf_token`: HuggingFace token for private models (default: None).
- `wandb`: Whether to log training with Weights & Biases.
- `mask_args`: Arguments for mask learning in dictionary or arguments format. The dictionary should contain the following keys:
  - `learnable_mask`: Whether to learn the mask.
  - `mask_tile_size`: Tile size for mask parameters as (row_tile_size, col_tile_size).
  - `grad_checkpoint`: Whether to use gradient checkpointing.
  - `local_bs`: Local batch size for training.
  - `optimizer`: Optimizer for training. Options: `adamw_torch`, `adamw_apex`.
  - `fine_tuning_sequence_length`: Sequence length for fine-tuning  steps.
  - `target_sparsity_ratio`: Target sparsity ratio for the learned mask.
  - `lr`: Learning rate for training.
  - `sparse_reg`: Regularization strength for sparsity.
  - `weight_reg`: Regularization strength for weight decay.
  - `temp_schedule_2_4`: Temperature schedule for 2:4 sparsity pattern as [start_temp, end_temp].
  - `scaler_schedule_2_4`: Scaling schedule for 2:4 sparsity pattern as [start_step, end_step].
  - `hard_2_4`: Whether to use hard 2:4 sparsity during training.
  - `prior_strength_2_4`: Prior strength for 2:4 tile logits.
  - `temp_schedule_tile`: Temperature schedule for tile selection as [start_temp, end_temp].
  - `scaler_schedule_tile`: Scaling schedule for tile selection as [start_step, end_step].
  - `hard_tile`: Whether to use hard tile selection during training.
  - `prior_strength_tile`: Prior strength for tile logits.
  - `mask_llm`: Whether to train with MaskLLM (2:4 mask only).
  - **`unstructured`: Whether to train with LEAP (Unstructured Mask).**
  - `layer_target`: Whether to apply target sparsity per layer.


## Acknowledgement
This repository is build upon the [SLiM](https://github.com/Paramathic/slim) repository.

## Citation
If you use LEAP in your research, please cite:
```angular2html
@misc{mozaffari2025leap,
  author = {Mozaffari, Mohammad and Hourri, Younes},
  title = {LEAP: Learnable End-to-End Adaptive Pruning of LLMs},
  year = {2025},
  month = {December},
  day = {17},
  howpublished = {\url{https://www.cs.toronto.edu/~mmozaffari/compression-trinity/leap/index.html}},
  note = {Blog post}
}
```
