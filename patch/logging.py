import os
import pandas as pd
import wandb


CSV_COLUMNS = [
    "model",
    "prune_method",
    "sparsity_ratio",
    "sparsity_type",
    "lora_rank",
    "slim_lora",
    "shift_zero_metrics",
    "prune_lora",
    "quantize_lora",
    "lora_tile_size",
    "eval_dataset",
    "quantize_weight",
    "bitwidth",
    "tiled_weight_quantization",
    "weight_tile_size",
    "quantize_input",
    "input_bitwidth",
    "input_group_size",
    "fine_tune",
    "optimizer",
    "slim_quant",
    "perplexity",
    "mmlu",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
    "average",
]


def add_result_to_csv(args, ppl, lmharness_results):
    """
    Adds or updates the results of the current experiment to a CSV file.
    If a row with the same configuration already exists, it updates the perplexity
    and LM Harness results. Otherwise, it appends a new row.

    Args:
        args: The argument namespace object containing experiment configuration.
        ppl: The calculated perplexity score (float).
        lmharness_results: A dictionary where keys are LM Harness task names
                           (str) and values are the corresponding scores (float).

    Returns:
        None
    """
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    directory = os.path.dirname(args.output_csv_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(args.output_csv_path):
        df = pd.read_csv(args.output_csv_path)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    num_tasks = 8

    # Check if the row combination exists and update perplexity
    new_row_data = {
        column: getattr(args, column) for column in CSV_COLUMNS[:-num_tasks]
    }
    row_exists = df.index[
        (df[CSV_COLUMNS[:-num_tasks]] == pd.Series(new_row_data)).all(axis=1)
    ]

    # Now we don't mind adding perplexity
    new_row_data["perplexity"] = ppl
    for task in lmharness_results:
        new_row_data[task] = lmharness_results[task]

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=CSV_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, "perplexity"] = new_row_data["perplexity"]
        for task in lmharness_results:
            df.at[index_to_update, task] = lmharness_results[task]

    # Save to CSV
    df.to_csv(args.output_csv_path, index=False)


def log_results_to_wandb(ppl, lmharness_results, sparsity_ratio):
    """
    Logs perplexity and LM Harness results to the current Weights & Biases run.

    Assumes wandb.init() has already been called and the relevant configuration
    (from args) has been passed to it.

    Args:
        args: The argument namespace object containing experiment configuration.
              Although not directly used for logging metrics here (config is
              usually set during wandb.init), it's kept for function signature
              consistency with the original CSV function, and might be useful
              if you decide to log specific args as metrics later.
        ppl: The calculated perplexity score (float).
        lmharness_results: A dictionary where keys are LM Harness task names
                           (str) and values are the corresponding scores (float).
    """
    if wandb.run is None:
        print("Warning: wandb.init() has not been called. Skipping W&B logging.")
        # Or raise an error:
        # raise RuntimeError("wandb.init() must be called before logging results.")
        return

    # Prepare the dictionary of metrics to log
    metrics_to_log = {}

    # Add perplexity
    metrics_to_log["perplexity"] = ppl

    # Add final sparsity ratio
    metrics_to_log["final_sparsity_ratio"] = sparsity_ratio

    # Add LM Harness results
    # It's often good practice to potentially prefix task names to avoid
    # collisions with other metrics, e.g., 'lmharness/task_name'
    # For simplicity here, we'll use the raw task names as keys.
    metrics_to_log.update(lmharness_results)
    # If you prefer prefixing:
    # for task, score in lmharness_results.items():
    #     metrics_to_log[f"lmharness/{task}"] = score

    # Log the metrics to the current W&B run
    wandb.log(metrics_to_log)

    print(f"Results logged to W&B run: {wandb.run.name} (ID: {wandb.run.id})")


def init_wandb(args):
    """
    Initialize a Weights & Biases (W&B) run for logging.
    """
    try:
        model_name = args.model.split("/")[-1]
        name_parts = [
            model_name,
            f"LR{args.lr}",
            f"REG{args.reg_factor}",
            f"OPT{args.optimizer}",
            f"Prune-{args.prune_method}",
            f"Sparsity{args.sparsity_ratio}-{args.target_sparsity_ratio}",
            f"T{args.temp_schedule[0]}-{args.temp_schedule[1]}",
            f"S{args.scaler_schedule[0]}-{args.scaler_schedule[1]}",
            f"STR{args.prior_strength}",
        ]
        if args.joint_optim:
            name_parts += [
                f"TT{args.tile_temp_schedule[0]}-{args.tile_temp_schedule[1]}",
                f"ST{args.tile_scaler_schedule[0]}-{args.tile_scaler_schedule[1]}",
                f"TSTR{args.prior_strength_tile}",
            ]

        if args.weight_reg > 0:

            name_parts += [
                f"WREG{args.weight_reg}",
            ]
        run_name = "_".join(name_parts)
        project = "PATCH" if not args.mask_llm else "MaskLLM"
        run = wandb.init(
            project=project,
            config=args,
            name=run_name,
            tags=[args.model.split("/")[-1], str(args.target_sparsity_ratio)],
            reinit=True,
        )
        print(f"W&B run initialized: {run.name} (ID: {run.id})")
        return run
    except Exception as e:
        print(f"Could not initialize W&B: {e}. Skipping W&B logging.")
        args.wandb = False
