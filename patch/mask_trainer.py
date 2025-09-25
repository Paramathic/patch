import torch
import wandb

from transformers import (
    Trainer,
    TrainerCallback,
)
from .regularizers import *
from dataclasses import dataclass


@dataclass
class MaskConfig:
    """
    Masking and sparsity settings for training.
    """

    mask_llm: bool
    layer_target: bool
    target_density: float
    sparse_reg: float
    weight_reg: float
    log_wandb: bool
    grad_acc: int


class MaskTrainer(Trainer):
    """
    Trainer class with custom loss computation for mask training.
    """

    def __init__(self, *args, mask_cfg: MaskConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_cfg = mask_cfg

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute the loss with additional regularization for mask training.

        Args:
            model: The model being trained.
            inputs: The input data for the model.
            return_outputs: Whether to return the model outputs along with the loss.
            num_items_in_batch: Optional number of items in the batch for loss computation.

        Returns:
            The computed loss, and optionally the model outputs.
        """
        cfg = self.mask_cfg
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Model Loss
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        final_loss = loss

        # Target Density loss
        if not cfg.mask_llm:
            if cfg.layer_target:
                density_ratio = compute_layer_density_ratio(model, loss.device)
            else:
                density_ratio = compute_model_density_ratio(model, loss.device)

            regularization = (
                torch.abs(density_ratio - cfg.target_density)
                * cfg.sparse_reg
                / cfg.grad_acc
            )

            final_loss += regularization.sum()

        # Weight Regularization loss
        if cfg.weight_reg > 0:
            final_loss -= (
                compute_weight_sum(model, final_loss.device, cfg.mask_llm)[0]
                * (cfg.weight_reg * (1 - self.state.global_step / self.state.max_steps))
                / cfg.grad_acc
            )

        return (final_loss, outputs) if return_outputs else final_loss


class TemperatureScalerController:
    """
    Control annealing of Gumbel temperature and scaling factor during training.
    """

    def __init__(
        self,
        start_temp: float,
        end_temp: float,
        start_scaler: float,
        end_scaler: float,
        total_steps: int,
    ):
        self.start_t = start_temp
        self.end_t = end_temp
        self.start_s = start_scaler
        self.end_s = end_scaler
        self.total = total_steps
        self.current_step = 0

    def get(self):
        ratio = min(self.current_step / self.total, 1.0)

        temp = self.start_t + (self.end_t - self.start_t) * ratio
        scaler = self.start_s + (self.end_s - self.start_s) * ratio

        return temp, scaler

    def step(self):
        self.current_step += 1


class MaskLogger(TrainerCallback):
    def __init__(self, mask_cfg: MaskConfig = None):
        super().__init__()
        self.mask_cfg = mask_cfg

    @torch.no_grad()
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Log mask density and regularization values to console and wandb.
        """
        # Ensure this runs only on the main process to avoid redundant logging
        if not state.is_world_process_zero:
            return

        model = kwargs["model"]
        cfg = self.mask_cfg

        density_ratio = compute_model_density_ratio(model)

        if cfg.layer_target:
            density_ratio_reg = compute_layer_density_ratio(model)
        else:
            density_ratio_reg = density_ratio
        mask_reg_value = (
            torch.abs(density_ratio_reg - cfg.target_density) * cfg.sparse_reg
        )

        mask_reg_value = mask_reg_value.sum()
        print(
            f"Density ratio: {density_ratio.item() * 100:.2f}%, regularization component: {mask_reg_value.item():.4f}"
        )

        if logs is not None:
            logs["mask_density_percent"] = (density_ratio * 100).item()
            logs["mask_regularization_value"] = mask_reg_value.item()

            logs["weight_regularization_value"] = (
                (
                    compute_weight_sum(model, mask_llm=cfg.mask_llm)
                    * (cfg.weight_reg * (1 - state.global_step / state.max_steps))
                ).item()
                if cfg.weight_reg > 0
                else 0
            )

        log_model_loss = (
            logs.get("loss", None) is not None
            and logs.get("mask_regularization_value", None) is not None
        )
        if cfg.log_wandb and logs is not None:
            to_substract = logs.get("mask_regularization_value", 0) - logs.get(
                "weight_regularization_value", 0
            )
            wandb_logs = {
                "train/Total Loss": logs.get(
                    "loss", None
                ),  # Already averaged over devices & grad accum steps
                "train/Model loss": (
                    logs.get("loss", None) - to_substract if log_model_loss else None
                ),
                "train/Learning Rate": logs.get("learning_rate", None),
                "train/Mask Density (%)": logs.get("mask_density_percent", None),
                "train/Weight Regularization Value": logs.get(
                    "weight_regularization_value", None
                ),
                "train/Mask Regularization Value": logs.get(
                    "mask_regularization_value", None
                ),
                "train/Mask Similarity Avg": logs.get("mask_similarity_avg", None),
                "train/Epoch": logs.get("epoch", None),
                "train/Step": state.global_step,
            }
            # Filter out None values before logging
            wandb.log({k: v for k, v in wandb_logs.items() if v is not None})
            log_density_per_transformer_layer(model)


class TemperatureScalerSchedulerCallback(TrainerCallback):
    """
    Callback to update Gumbel temperature and scaling factor at each training step.
    """

    def __init__(
        self,
        temp_scaler_controller_2_4=None,
        temp_scaler_controller_tile=None,
        log_wandb=True,
    ):
        self.temp_scaler_controller_weight = temp_scaler_controller_2_4
        self.temp_scaler_controller_tile = temp_scaler_controller_tile
        self.log_wandb = log_wandb

    def on_step_end(self, args, state, control, **kwargs):
        if self.temp_scaler_controller_weight is not None:
            self.temp_scaler_controller_weight.step()
        if self.temp_scaler_controller_tile is not None:
            self.temp_scaler_controller_tile.step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.log_wandb and logs is not None:
            if self.temp_scaler_controller_weight is not None:
                t, s = self.temp_scaler_controller_weight.get()
                wandb.log({"train/Gumbel Temperature Weight": t})
                wandb.log({"train/Gumbel Scaler Weight": s})

            if self.temp_scaler_controller_tile is not None:
                tt, st = self.temp_scaler_controller_tile.get()
                wandb.log({"train/Gumbel Temperature Tile": tt})
                wandb.log({"train/Gumbel Scaler Tile": st})
