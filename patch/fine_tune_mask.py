import os
import torch
import transformers
from transformers import (
    TrainingArguments,
    default_data_collator,
)

from types import MethodType
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from slim_local.slim.utils import report_gpu_memory, get_layers_list
from .data import get_data, compute_metrics, preprocess_logits_for_metrics
from .learnable_mask import (
    add_mask_parameters,
    patch_tile_only,
    patch_joint,
    create_2_4_mask,
)
from .mask_trainer import *
import os
from slim_local.utils.model import get_llm, distribute_model
from .utils import Args


def apply_checkpointing_to_layers_methodtype(layers):
    """
    Overwrites the forward method of each transformer layer to use gradient checkpointing.
    Handles both positional and keyword arguments correctly for compatibility with OPT models.

    Args:
        layers (list): List of transformer blocks (nn.Module instances).

    Returns:
        list: List of modified layers with checkpointed forward methods.
    """

    def create_checkpointed_forward(layer):
        def checkpointed_forward(self, *args, **kwargs):
            # Wrap the forward pass in a function that handles kwargs
            def forward_fn(*pos_args):
                return self._original_forward(*pos_args, **kwargs)

            return checkpoint(forward_fn, *args, use_reentrant=False)

        return checkpointed_forward

    for layer in layers:
        # Store the original forward method
        layer._original_forward = layer.forward
        # Overwrite the forward method with checkpointed version
        layer.forward = MethodType(create_checkpointed_forward(layer), layer)

    return layers


def dense_linear_forward(module, input):
    """
    Standard forward pass for a dense
    linear layer without any masking.
    """
    return torch.nn.functional.linear(input, module.weight, module.bias)


def fine_tune_mask(
    model,
    compressed_model,
    tokenizer,
    # Data / preprocessing
    streaming=False,
    preprocessing_num_workers=os.cpu_count(),
    overwrite_cache=False,
    sequence_length=4096,
    max_train_samples=512000,
    max_eval_samples=128,
    # Training setup
    epochs=1,
    global_batch_size=256,
    local_batch_size=4,
    gradient_checkpointing=False,
    # Optimization
    optimizer="adamw_torch",
    lr=2.5e-2,
    # Regularization / sparsity
    weight_reg=0,
    sparse_reg=1e-10,
    target_density=0.5,
    hard_2_4=False,
    hard_tile=False,
    prior_strength_2_4=3,
    prior_strength_tile=0,
    layer_target=False,
    temp_range_2_4=[1, 1],
    scaler_range_2_4=[1, 1],
    temp_range_tile=[1, 1],
    scaler_range_tile=[1, 1],
    # Masking
    mask_tile_size=[128, 128],
    mask_llm=True,
    joint_training=False,
    # Logging
    log_wandb=False,
):
    ...

    """
    Fine-tunes the mask parameters of a given model using a specified dataset and training configuration.

    Args:
        model (torch.nn.Module): The model to fine-tune
        compressed_model (torch.nn.Module): One-shot/MaskLLM compressed version of the model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process input data

        # Data / preprocessing
        streaming (bool): Enable dataset streaming mode 
        preprocessing_num_workers (int): Number of worker processes for preprocessing
        overwrite_cache (bool): Overwrite cached training and evaluation sets
        sequence_length (int): Input sequence length after tokenization
        max_train_samples (int): If set, truncate training dataset to this number of samples
        max_eval_samples (int): If set, truncate evaluation dataset to this number of samples

        # Training setup
        epochs (int): Number of training epochs
        global_batch_size (int): Global batch size across devices
        local_batch_size (int): Per-device batch size
        gradient_checkpointing (bool): Enable gradient checkpointing 

        # Optimization
        optimizer (str): Optimizer to use 
        lr (float): Learning rate

        # Regularization / sparsity
        weight_reg (float): Coefficient for weight regularization
        sparse_reg (float): Coefficient for sparsity regularization
        target_density (float): Global target density of the  model
        hard_2_4 (bool): Whether to apply hard gumbel on the 2:4 sampling
        hard_tile (bool): Whether to apply hard gumbel on the tile sampling
        prior_strength_2_4 (float): Prior strength for 2:4 logits
        prior_strength_tile (float): Prior strength for tile logits
        layer_target (bool): Whether to apply layer-specific density targetting
        temp_range_2_4 (list[float]): Start and end temperature for 2:4 gumbel logits
        scaler_range_2_4 (list[float]): Start and end scaling factor for 2:4 gumbel logits
        temp_range_tile (list[float]): Start and end temperature for tile gumbel logits
        scaler_range_tile (list[float]): Start and end scaling factor for tile gumbel logits

        # Masking
        mask_tile_size (list[int]): Tile size for masking
        mask_llm (bool): Whether we're training 2:4 logits only (maskllm)
        joint_training (bool): Whether to jointly optimize tile and 2:4 structure
        #If neither mask_llm or joint_training is true, then only the tile structure will be trained

        # Logging
        log_wandb (bool): Whether to log training metrics to Weights & Biases

    Returns:
        None
    """
    dtype = (
        torch.bfloat16
        if transformers.utils.import_utils.is_torch_bf16_gpu_available()
        else torch.float32
    )
    model = model.to(dtype)

    grad_acc = (
        global_batch_size
        // local_batch_size
        // (dist.get_world_size() if int(os.environ.get("WORLD_SIZE", 1)) > 1 else 1)
    )

    training_args = TrainingArguments(
        output_dir="data/mask_training_output",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=local_batch_size,
        per_device_eval_batch_size=local_batch_size,
        num_train_epochs=epochs,
        logging_dir="logs",
        logging_steps=1,
        eval_steps=100,
        save_safetensors=False,
        save_steps=500,
        save_total_limit=1,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is not torch.bfloat16,
        group_by_length=False,
        gradient_accumulation_steps=grad_acc,
        warmup_steps=5,
        optim=optimizer,
        save_strategy="steps",
        gradient_checkpointing=False,
        learning_rate=lr,
        report_to="wandb" if log_wandb else "none",
        auto_find_batch_size=False,
        lr_scheduler_type="cosine",
        weight_decay=0 if not mask_llm else 0.1,
        adam_beta2=0.95,
    )

    mask_cfg = MaskConfig(
        mask_llm=mask_llm,
        layer_target=layer_target,
        target_density=target_density,
        sparse_reg=sparse_reg,
        weight_reg=weight_reg,
        log_wandb=log_wandb,
        grad_acc=grad_acc,
    )

    # ------------------------------------------- Create Data -------------------------------------------
    train_dataset, eval_dataset = get_data(
        "DKYoon/SlimPajama-6B",
        training_args,
        tokenizer,
        streaming,
        preprocessing_num_workers,
        overwrite_cache,
        sequence_length,
        max_train_samples,
        max_eval_samples,
    )

    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    train_samples = min(max_train_samples, len(train_dataset))
    training_steps = (train_samples // (global_batch_size)) * epochs

    # ----------------------------- Create Parameters / Replace Forward -----------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    if mask_llm:
        mode = "mask_llm"

        def make_generate_masks_pre_hook(temp_scaler_controller_2_4):
            def generate_masks_pre_hook(self, input):
                tau, scaler = temp_scaler_controller_2_4.get()
                for layer in self.modules():
                    if isinstance(layer, torch.nn.Linear) and hasattr(
                        layer, "mask_2_4"
                    ):
                        new_mask = create_2_4_mask(
                            layer.mask_2_4,
                            layer.weight.shape,
                            self.mask_choices,
                            hard_2_4,
                            tau,
                            scaler,
                        )
                        layer.last_mask = new_mask

            return generate_masks_pre_hook

    elif joint_training:
        mode = "joint"

        def make_generate_masks_pre_hook(
            temp_scaler_controller_2_4, temp_scaler_controller_tile
        ):
            def generate_masks_pre_hook(self, input):
                tau_2_4, scaler_2_4 = temp_scaler_controller_2_4.get()
                tau_tile, scaler_tile = temp_scaler_controller_tile.get()
                for layer in self.modules():
                    if isinstance(layer, torch.nn.Linear) and hasattr(
                        layer, "mask_2_4"
                    ):
                        new_mask = patch_joint(
                            layer.mask_2_4,
                            layer.weight.shape,
                            self.mask_choices,
                            hard_2_4,
                            tau_2_4,
                            scaler_2_4,
                            layer.tile_mask,
                            hard_tile,
                            tau_tile,
                            scaler_tile,
                            layer.tile_row_size,
                            layer.tile_col_size,
                        )
                        layer.last_mask = new_mask

            return generate_masks_pre_hook

    else:
        mode = "tile"

        def make_generate_masks_pre_hook(temp_scaler_controller_tile):
            def generate_masks_pre_hook(self, input):
                tau, scaler = temp_scaler_controller_tile.get()
                for layer in self.modules():
                    if isinstance(layer, torch.nn.Linear) and hasattr(
                        layer, "fixed_mask_2_4"
                    ):
                        new_mask = patch_tile_only(
                            layer.fixed_mask_2_4,
                            layer.tile_mask,
                            hard_2_4,
                            tau,
                            scaler,
                            layer.tile_row_size,
                            layer.tile_col_size,
                        )
                        layer.last_mask = new_mask

            return generate_masks_pre_hook

    add_mask_parameters(
        model,
        compressed_model,
        tokenizer,
        mode,
        mask_tile_size,
        dtype,
        prior_strength_2_4,
        prior_strength_tile,
        target_density,
    )

    if gradient_checkpointing:
        layers = get_layers_list(model)
        apply_checkpointing_to_layers_methodtype(layers)

    report_gpu_memory("After adding masks")
    compressed_model.cpu()
    torch.cuda.empty_cache()
    report_gpu_memory("After emptying cache")

    # ------------------------------------------- Train -------------------------------------------
    model.config.use_cache = False
    if training_args.do_train:
        model.train()

        controllers = {}
        if mode in ("mask_llm", "joint"):
            ctrl_2_4 = TemperatureScalerController(
                start_temp=temp_range_2_4[0],
                end_temp=temp_range_2_4[1],
                start_scaler=scaler_range_2_4[0],
                end_scaler=scaler_range_2_4[1],
                total_steps=training_steps,
            )
            controllers["temp_scaler_controller_2_4"] = ctrl_2_4

        if mode in ("tile", "joint"):
            ctrl_tile = TemperatureScalerController(
                start_temp=temp_range_tile[0],
                end_temp=temp_range_tile[1],
                start_scaler=scaler_range_tile[0],
                end_scaler=scaler_range_tile[1],
                total_steps=training_steps,
            )
            controllers["temp_scaler_controller_tile"] = ctrl_tile

        hook = model.register_forward_pre_hook(
            make_generate_masks_pre_hook(**controllers)
        )

        trainer = MaskTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=(
                preprocess_logits_for_metrics if training_args.do_eval else None
            ),
            callbacks=[
                MaskLogger(mask_cfg),
                TemperatureScalerSchedulerCallback(**controllers, log_wandb=log_wandb),
            ],
            mask_cfg=mask_cfg,
        )

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = train_samples
        trainer.log_metrics("train", metrics)

    if "hook" in locals() and hook is not None:
        hook.remove()
        print("Removed model forward pre-hook.")

    # ------------------------------------------- Get Hard Masks -------------------------------------------

    if trainer.is_world_process_zero():
        unwrapped_model = model.module if hasattr(model, "module") else model
        for layer in unwrapped_model.modules():
            if isinstance(layer, torch.nn.Linear) and hasattr(layer, "last_mask"):
                if hasattr(layer, "mask_2_4"):
                    mask_2_4 = (
                        model.mask_choices[torch.argmax(layer.mask_2_4, dim=-1)]
                        .view(layer.weight.shape[0], layer.weight.shape[1])
                        .bool()
                    )

                    if mask_llm:
                        mask = mask_2_4

                    del layer.mask_2_4

                elif hasattr(layer, "fixed_mask_2_4"):
                    mask_2_4 = layer.fixed_mask_2_4
                    del layer.fixed_mask_2_4

                if hasattr(layer, "tile_mask"):
                    tile_mask = (layer.tile_mask.data > 0).to(torch.bfloat16)
                    tile_mask = tile_mask.repeat_interleave(
                        layer.tile_row_size, dim=0
                    ).repeat_interleave(layer.tile_col_size, dim=1)
                    mask = (tile_mask + (1 - tile_mask) * mask_2_4).bool()

                    del layer.tile_mask
                    del layer.tile_row_size
                    del layer.tile_col_size

                layer.weight.data[~mask] = 0.0
                layer.forward = MethodType(dense_linear_forward, layer)

        if hasattr(model, "mask_choices"):
            del model.mask_choices

    hook.remove()
    return model


def learn_mask(
    model_name,
    compressed_model,
    tokenizer,
    mask_args,
    local_files_only=False,
    hf_token=None,
    wandb=False,
):
    """
    Learns pruning mask on top of frozen weights

    Args:
        model_name (str): HuggingFace model name or path
        compressed_model (torch.nn.Module): One-shot/MaskLLM compressed version of the model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process input data
        mask_args (argparse.Namespace): Arguments for mask learning
        local_files_only (bool): Whether to use local files only when loading the model
        hf_token (str or None): HuggingFace token for private models
        wandb (bool): Whether to log training metrics to Weights & Biases

    Returns:
        model (torch.nn.Module): The fine-tuned model
        lm_eval_model (transformers.PreTrainedModel): The model for LM Harness evaluation
    """
    if type(mask_args) is dict:
        mask_args = Args(**mask_args)
    model, lm_eval_model = get_llm(
        model_name=model_name,
        local_files_only=local_files_only,
        hf_token=hf_token,
        seqlen=mask_args.fine_tuning_sequence_length,
    )

    model = model.to(torch.bfloat16)
    model = distribute_model(model)

    fine_tune_mask(
        model,
        compressed_model,
        tokenizer,
        sequence_length=mask_args.fine_tuning_sequence_length,
        local_batch_size=mask_args.local_bs,
        gradient_checkpointing=mask_args.grad_checkpoint,
        optimizer=mask_args.optimizer,
        lr=mask_args.lr,
        weight_reg=mask_args.weight_reg,
        sparse_reg=mask_args.sparse_reg,
        target_density=1.0 - mask_args.target_sparsity_ratio,
        hard_2_4=mask_args.hard_2_4,
        hard_tile=mask_args.hard_tile,
        prior_strength_2_4=mask_args.prior_strength_2_4,
        prior_strength_tile=mask_args.prior_strength_tile,
        layer_target=mask_args.layer_target,
        temp_range_2_4=mask_args.temp_schedule_2_4,
        scaler_range_2_4=mask_args.scaler_schedule_2_4,
        temp_range_tile=mask_args.temp_schedule_tile,
        scaler_range_tile=mask_args.scaler_schedule_tile,
        mask_tile_size=mask_args.mask_tile_size,
        mask_llm=mask_args.mask_llm,
        joint_training=mask_args.joint_optim,
        log_wandb=wandb,
    )

    return model, lm_eval_model
