import argparse

# ---- helpers ---------------------------------------------------------------


def _ns_from(args, keys):
    """Build a Namespace with only the provided keys from args."""
    return argparse.Namespace(**{k: getattr(args, k) for k in keys if hasattr(args, k)})


def one_shot_pruning_args(parser):
    """
    Arguments specific to one-shot pruning
    """

    grp = parser.add_argument_group("One Shot Pruning")
    dests = []

    def add(*a, **kw):
        action = grp.add_argument(*a, **kw)
        dests.append(action.dest)
        return action

    add(
        "--prune_method",
        type=str,
        choices=["magnitude", "wanda", "sparsegpt", "maskllm", "joint_pq"],
    )
    add("--sparsity_type", type=str, choices=["unstructured", "2:4"])
    add("--nsamples", type=int, default=128, help="Number of calibration samples.")
    add(
        "--sparsity_ratio",
        type=float,
        default=0.0,
        help="Initial sparsity ratio (From One-shot Pruning)",
    )
    add(
        "--optimizer_FT_pruning",
        type=str,
        default="adamw_torch",
        help="Optimizer for training",
    )

    add("--calibration_dataset", type=str, default="c4", choices=["c4", "slimpajama"])
    add(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "openwebtext", "slimpajama"],
    )
    add("--shift_zero_metrics", action="store_true")
    add("--fine_tune", action="store_true", help="Fine-tune after pruning")
    add(
        "--maskllm_checkpoint",
        type=str,
        default=None,
        help="Load pretrained MaskLLM model",
    )

    return dests


def learnable_mask_args(parser):
    """
    Arguments specific to learnable mask
    """
    grp = parser.add_argument_group("Learnable mask")
    dests = []

    def add(*a, **kw):
        action = grp.add_argument(*a, **kw)
        dests.append(action.dest)
        return action

    add("--learnable_mask", action="store_true", help="Enables training with PATCH")
    add(
        "--mask_tile_size",
        type=str,
        default="128,128",
        help="Tile size for mask parameters as 'row,col'",
    )
    add("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing")
    add("--local_bs", type=int, default=1, help="Local micro-batch size")
    add("--optimizer", type=str, default="adamw_torch", help="Optimizer for training")
    add(
        "--fine_tuning_sequence_length",
        type=int,
        default=4096,
        help="Sequence length used during training",
    )
    add(
        "--target_sparsity_ratio", type=float, default=0.0, help="Target sparsity ratio"
    )

    add("--lr", type=float, default=1e-3, help="Learning rate for mask training")
    add("--sparse_reg", type=float, default=7, help="Sparsity regularization penalty")
    add("--weight_reg", type=float, default=10.0, help="Weight regularizing penalty")
    add(
        "--temp_schedule_2_4",
        type=float,
        nargs=2,
        default=[4.0, 0.05],
        help="2:4 mask logit temperature schedule: start end",
    )
    add(
        "--scaler_schedule_2_4",
        type=float,
        nargs=2,
        default=[100.0, 500.0],
        help="2:4 mask logit scaling schedule: start end",
    )
    add("--hard_2_4", action="store_true", help="Use hard sampling for 2:4 gumbel")
    add(
        "--prior_strength_2_4",
        type=float,
        default=3.0,
        help="Prior strength for mask logits",
    )

    add("--joint_optim", action="store_true", help="Combine tile-level and 2:4 masking")
    add(
        "--temp_schedule_tile",
        type=float,
        nargs=2,
        default=[4.0, 0.05],
        help="Tile mask logit temperature schedule: start end",
    )
    add(
        "--scaler_schedule_tile",
        type=float,
        nargs=2,
        default=[25.0, 350.0],
        help="Tile mask logit scaling schedule: start end",
    )
    add("--hard_tile", action="store_true", help="Use hard sampling for tile gumbel")
    add(
        "--prior_strength_tile",
        type=float,
        default=3.0,
        help="Prior strength for tile logits",
    )

    add("--mask_llm", action="store_true", help="Train with MaskLLM (2:4 mask only)")

    add("--layer_target", action="store_true", help="Apply target sparsity per layer")

    return dests


def general_args(parser):
    grp = parser.add_argument_group("General")
    dests = []

    def add(*a, **kw):
        action = grp.add_argument(*a, **kw)
        dests.append(action.dest)
        return action

    add("--wandb", action="store_true", help="Enable Weight and Biases")
    add("--model", type=str, help="HF model name or path")
    add("--seed", type=int, default=0, help="Random seed")
    add("--hf_token", type=str, default="", help="Hugging Face token")
    add("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    add(
        "--output_csv_path",
        type=str,
        default=None,
        help="Path to accumulate results as CSV",
    )
    add("--test_lmharness", action="store_true", help="Run LM Harness evaluation")
    add(
        "--evaluate_perplexity",
        action="store_true",
        help="Evaluate perplexity on dataset",
    )
    add(
        "--local_files_only",
        action="store_true",
        help="Force HF to use local files only",
    )
    add(
        "--save_model_path",
        type=str,
        default="",
        help="Save model at the path after training.",
    )

    return dests


def parse_args():
    parser = argparse.ArgumentParser()

    one_shot_keys = one_shot_pruning_args(parser)
    learnable_keys = learnable_mask_args(parser)
    general_keys = general_args(parser)

    args = parser.parse_args()
    args.mask_tile_size = [int(x) for x in args.mask_tile_size.split(",")]

    args.one_shot = _ns_from(args, one_shot_keys)
    args.learnable = _ns_from(args, learnable_keys)
    args.general = _ns_from(args, general_keys)

    if getattr(args.learnable, "mask_llm", False) and getattr(
        args.learnable, "joint_optim", False
    ):
        parser.error(
            "Please enable only one of the --mask_llm and --joint-optim options."
        )

    return args
