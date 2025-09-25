import os
import numpy as np
import torch
import lm_eval

import os
import sys

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'slim' directory
slim_path = os.path.join(script_dir, "../slim_local")

# Add the 'slim' directory to the Python path
if slim_path not in sys.path:
    sys.path.insert(0, slim_path)

from slim_local.slim.prune import prune_and_quantize
from slim_local.slim.eval import eval_ppl
from slim_local.slim.utils import report_gpu_memory, check_sparsity
from slim_local.slim.lora import quantize_lora
from slim_local.slim.fine_tune import fine_tune
from copy import deepcopy


def save_tile_scores(
    model,
    unstructured_model,
    tile_size,
):
    """
    Compute and attach per-tile sparsity distribution scores for Linear layers

    Args:
        model: Model whose layers will receive the distribution scores as buffers
        unstructured_model: Reference model with unstructured sparsity
                            used to compute per-tile nonzero counts
        tile_size (tuple[int, int]): Tile granularity as (row_tile_size, col_tile_size)

    Returns:
        None (the model layers are modified in place)
    """
    for layer, layer_unstructued in zip(
        model.model.modules(), unstructured_model.model.modules()
    ):
        if isinstance(layer, torch.nn.Linear):
            row_tile_size, col_tile_size = tile_size

            # Tiled params
            out_features, in_features = layer.weight.shape
            num_row_tiles = out_features // row_tile_size
            num_col_tiles = in_features // col_tile_size

            tiled = layer_unstructued.weight.view(
                num_row_tiles, tile_size[0], num_col_tiles, tile_size[1]
            )
            distribution_scores = tiled.ne(0).sum(dim=(1, 3)) / (
                tile_size[0] * tile_size[1]
            )

            layer.register_buffer("distribution_scores", distribution_scores)


def prepare_pruned_model(
    model,
    tokenizer,
    checkpoint_name,
    prune_args,
    mask_tile_size,
    target_sparsity_ratio,
    seed=0,
):
    """
    Prepare a compressed model via one-shot pruning (e.g., Wanda, SparseGPT)
    or by loading a MaskLLM checkpoint.

    - Saves tile distribution scores from an unstructured reference model.
    - Optionally applies weight/LORA quantization or fine-tuning.
    - Saves the final compressed checkpoint.

    Args:
        model: The model to be pruned/compressed.
        tokenizer: The tokenizer associated with the model.
        checkpoint_name: Path to save/load the pruned model checkpoint.
        prune_args: Arguments for pruning and quantization.
        mask_tile_size: Tile size for mask parameters as (row_tile_size, col_tile_size
        target_sparsity_ratio: Target sparsity ratio for the unstructured reference model.
        seed: Random seed for reproducibility.

    Returns:
        The pruned/compressed model on CPU.
    """
    if type(prune_args) is dict:
        prune_args = Args(**prune_args)
    if not os.path.exists(checkpoint_name):
        unstructured_model = deepcopy(model)

        prune_and_quantize(
            model,
            tokenizer,
            prune_method=prune_args.prune_method,
            sparsity_ratio=prune_args.sparsity_ratio,
            sparsity_type=prune_args.sparsity_type,
            nsamples=prune_args.nsamples,
            shift_zero_metrics=prune_args.shift_zero_metrics,
            seed=seed,
            calibration_dataset=prune_args.calibration_dataset,
            mask_checkpoint=prune_args.maskllm_checkpoint,
        )

        prune_and_quantize(
            unstructured_model,
            tokenizer,
            prune_method=(
                prune_args.prune_method
                if prune_args.prune_method != "maskllm"
                else "wanda"
            ),
            sparsity_ratio=target_sparsity_ratio,
            sparsity_type="unstructured",
            nsamples=prune_args.nsamples,
            shift_zero_metrics=prune_args.shift_zero_metrics,
            seed=seed,
            calibration_dataset=prune_args.calibration_dataset,
        )

        save_tile_scores(model, unstructured_model, mask_tile_size)

        ################################################################
        if prune_args.fine_tune:
            report_gpu_memory("Before Fine-tuning")
            fine_tune(model, tokenizer, optimizer=prune_args.optimizer_FT_pruning)
            report_gpu_memory("After Fine-tuning")
            print("*" * 30)

        if os.path.dirname(checkpoint_name) != "":
            os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_name)

        report_gpu_memory("After pruning")
        ################################################################
        print("*" * 30)
        sparsity_ratio = check_sparsity(model)
        print(f"Model Sparsity Ratio: {sparsity_ratio:.2f}")
        print("*" * 30)
        ################################################################

    return model.cpu()


def load_compressed_model(model, mask_tile_size, checkpoint_name):
    """
    Load a compressed model checkpoint into a base architecture.

    - If not using MaskLLM, initializes per-tile `distribution_scores` buffers
      based on the specified mask tile size.

    Args:
        model (torch.nn.Module): Base model architecture to load weights into.
        mask_tile_size (tuple(int, int)): (row_tile_size, col_tile_size) used for creating tile score buffers.
        checkpoint_name (str): Path to the compressed checkpoint file to load.
        mask_llm (bool): If True, skip buffer initialization

    Returns:
        model (torch.nn.Module): Model with loaded weights and initialized buffers.
    """
    model = model.cpu()
    for name, module in model.named_modules():
        if "layers" in name and isinstance(module, torch.nn.Linear):
            row_tile_size, col_tile_size = mask_tile_size

            out_features, in_features = module.weight.shape
            num_row_tiles = out_features // row_tile_size
            num_col_tiles = in_features // col_tile_size

            empty_tensor = torch.empty(num_row_tiles, num_col_tiles)
            module.register_buffer("distribution_scores", empty_tensor)

    model.load_state_dict(torch.load(checkpoint_name), strict=False)
    return model


def evaluate(
    model,
    lm_eval_model,
    tokenizer,
    evaluate_perplexity=True,
    eval_dataset="wikitext2",
    eval_batch_size=1,
    test_lmharness=True,
):
    """
    Evaluates perplexity and accuracy over different tasks

    Args:
        model (torch.nn.Module): The model to evaluate
        lm_eval_model (lm_eval.models.base.LM): The wrapped model for LM Evaluation Harness
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
        evaluate_perplexity (bool): If True, compute perplexity on `eval_dataset`.
        eval_dataset (str): Dataset name or path for perplexity evaluation.
        eval_batch_size (int): Batch size for evaluation.
        test_lmharness (bool): If True, run LM Harness benchmark tasks.

    Returns:
        ppl_test (float): Computed perplexity (if `evaluate_perplexity` is True, else 0.0)
        lmharness_results (dict): Dictionary of LM Harness results (if `test_lmharness` is True, else empty dict)
    """
    model = model.cuda()
    seqlen = 4096
    model.config.max_position_embeddings = seqlen
    model.seqlen = seqlen
    ################################################################
    ppl_test = 0.0
    if evaluate_perplexity:
        ppl_test = eval_ppl(
            model,
            tokenizer,
            eval_dataset,
            eval_batch_size,
        )
        print(f"Perplexity: {ppl_test:.2f}")
        print("*" * 30)
    ################################################################

    lmharness_results = {}
    if test_lmharness:
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=[
                "mmlu",
                "piqa",
                "arc_easy",
                "arc_challenge",
                "winogrande",
                "openbookqa",
            ],
            verbosity="ERROR",
        )
        lmharness_results["mmlu"] = results["results"]["mmlu"]["acc,none"]
        lmharness_results["piqa"] = results["results"]["piqa"]["acc,none"]
        lmharness_results["arc_easy"] = results["results"]["arc_easy"]["acc,none"]
        lmharness_results["arc_challenge"] = results["results"]["arc_challenge"][
            "acc,none"
        ]
        lmharness_results["winogrande"] = results["results"]["winogrande"]["acc,none"]
        lmharness_results["openbookqa"] = results["results"]["openbookqa"]["acc,none"]
        average = []
        for task in lmharness_results:
            average.append(lmharness_results[task])
        average = np.mean(average)
        lmharness_results["average"] = average
        print("LM Harness Results: ", lmharness_results)

    return ppl_test, lmharness_results


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)
