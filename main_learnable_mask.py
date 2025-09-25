import os
import sys

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'slim' directory
slim_path = os.path.join(script_dir, "slim_local")

# Add the 'slim' directory to the Python path
if slim_path not in sys.path:
    sys.path.insert(0, slim_path)

import numpy as np
import torch
import wandb

from transformers import AutoTokenizer
from slim_local.slim.utils import report_gpu_memory, check_sparsity
from slim_local.slim.quantization.quantization import attach_input_quantization_hooks
from slim_local.utils.model import get_llm
import torch.distributed as dist
from arguments import parse_args
from patch.logging import *
from patch.utils import prepare_pruned_model, load_compressed_model, evaluate
from patch.fine_tune_mask import learn_mask


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if is_distributed:
        import datetime

        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

    # Saved one-shot pruned model checkpoint name
    checkpoint_name = (
        f"saved_models/{args.model.split('/')[-1]}_"
        f"{args.one_shot.prune_method}_{args.one_shot.sparsity_type}_"
        f"{args.one_shot.sparsity_ratio}_{args.learnable.target_sparsity_ratio}.pt"
    )

    model_name = args.model.split("/")[-1]
    if rank == 0:
        print(f"Loading model {model_name}")

    model, lm_eval_model = get_llm(
        model_name=args.model,
        local_files_only=args.local_files_only,
        hf_token=args.hf_token,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        token=args.hf_token,
    )

    if rank == 0:
        if args.wandb:
            run = init_wandb(args)

        report_gpu_memory("Before Pruning")
        model = prepare_pruned_model(
            model,
            tokenizer,
            checkpoint_name,
            prune_args=args.one_shot,
            mask_tile_size=args.learnable.mask_tile_size,
            target_sparsity_ratio=args.learnable.target_sparsity_ratio,
            seed=args.seed,
        )

    if is_distributed:
        dist.barrier()

    if args.learnable.learnable_mask:
        compressed = load_compressed_model(
            model,
            args.learnable.mask_tile_size,
            checkpoint_name,
        )
        model, lm_eval_model = learn_mask(
            model_name=args.model,
            local_files_only=args.local_files_only,
            hf_token=args.hf_token,
            compressed_model=compressed,
            tokenizer=tokenizer,
            mask_args=args.learnable,
            wandb=args.wandb,
        )

        if args.save_model_path:
            torch.save(model.state_dict(), args.save_model_path)

    if rank == 0:
        print("*" * 30)
        sparsity_ratio = check_sparsity(model)
        print(f"Model Sparsity Ratio: {sparsity_ratio:.2f}")
        print("*" * 30)

        if args.wandb:
            wandb.log({"Model Sparsity Ratio": sparsity_ratio})

        ################################################################
        if args.quantize_input:
            print("Enabling input quantization:")
            attach_input_quantization_hooks(
                model,
                args.input_bitwidth,
                args.input_group_size,
            )
        ################################################################

        ppl_test, lmharness_results = evaluate(
            model,
            lm_eval_model,
            tokenizer,
            args.evaluate_perplexity,
            args.eval_dataset,
            args.eval_batch_size,
            args.test_lmharness,
        )

        if args.output_csv_path:
            add_result_to_csv(args, ppl_test, lmharness_results)

        if args.wandb:
            log_results_to_wandb(ppl_test, lmharness_results, sparsity_ratio)
            run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
