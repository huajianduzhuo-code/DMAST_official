from collections import defaultdict
from typing import Optional, Any, Union

import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator

from transformers.trainer import Trainer, _is_peft_model
from trl.trainer.utils import disable_dropout_in_model, prepare_deepspeed, first_true_indices, selective_log_softmax

from utils.utils import IGNORE_INDEX, to_numpy
from utils.two_hot import symlog, symexp, two_hot, compute_softmax_over_buckets

EPS = 1e-4


def extract_model_inputs(inputs, keys=None):
    """
    Extract model inputs from the batch, supporting both text-only and multimodal inputs.
    
    Args:
        inputs: Dict containing input_ids, attention_mask, position_ids, and optionally pixel_values
        keys: Optional list of specific keys to extract. If None, extracts common model inputs.
    
    Returns:
        Dict with the extracted model inputs
    """
    if keys is None:
        # Default keys for model forward pass
        # Include image_grid_thw for Qwen3-VL models (required for image position encoding)
        keys = ['input_ids', 'attention_mask', 'position_ids', 'pixel_values', 'image_grid_thw']
    
    model_inputs = {}
    for key in keys:
        if key in inputs:
            model_inputs[key] = inputs[key]
    
    return model_inputs


def prepare_frozen_model(trainer, model):
    # disable gradient the model
    for param in model.parameters():
        param.requires_grad = False

    # copied from trl.PPOTrainer
    if trainer.is_deepspeed_enabled:
        model = prepare_deepspeed(
            model, trainer.args.per_device_train_batch_size, trainer.args.fp16, trainer.args.bf16
        )
    else:
        model = model.to(trainer.accelerator.device)

    return model


def compute_log_prob(logits, labels, valid_only=True):
    """
    args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)

    return:
        valid_log_prob: (num_valid_tokens, ), i.e., the number of tokens where the labels are not IGNORE_INDEX
        mask: (batch_size, seq_len - 1)
    """
    device = logits.device

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    mask = (shift_labels != IGNORE_INDEX)

    # Handle case where there are no valid tokens (all labels are IGNORE_INDEX)
    if not mask.any():
        if valid_only:
            return torch.tensor([], device=device, dtype=logits.dtype), mask
        return torch.zeros_like(shift_labels, dtype=logits.dtype), mask

    valid_logits = shift_logits[mask]
    valid_labels = shift_labels[mask]
    valid_log_prob = selective_log_softmax(valid_logits, valid_labels)

    if valid_only:
        return valid_log_prob, mask

    log_prob = torch.zeros_like(shift_labels, dtype=logits.dtype)
    log_prob[mask] = valid_log_prob

    return log_prob, mask


def compute_kl_div(ref_logprob, logprob, only_penalize_kl_when_smaller_than_ref=False):
    if only_penalize_kl_when_smaller_than_ref:
        # this makes kl = 0 if logprob > ref_logprob, which does not penalize the model for exploration
        logprob = torch.minimum(logprob, ref_logprob)

    return torch.exp(ref_logprob - logprob) - (ref_logprob - logprob) - 1


class CustomMetricTrainer(Trainer):
    """
    A custom Trainer that allows for custom metrics to be logged.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a cache to store metrics between logging steps
        self._custom_log_cache = defaultdict(list)
        self.log_prefix = None

    def custom_logs(self) -> dict[str, float]:
        """
        Overwrite this method to aggregate custom metrics between logging steps.
        """
        logs = {}
        for key, values in self._custom_log_cache.items():
            if values:
                if np.isscalar(values[0]) or isinstance(values[0], np.ndarray):
                    logs[key] = np.mean(values)
                else:
                   raise ValueError(f"Unsupported metric type: {type(values[0])}")
            else:
                 pass
        return logs
    
    def add_log_prefix_postfix(self, logs):
        if self.log_prefix:
            logs = {f"{self.log_prefix}/{key}": value for key, value in logs.items()}

        if self.args.roles == ["agent", "attacker"]:
            postfix = ""
        elif self.args.roles == ["agent"] or self.args.roles == ["attacker"]:
            postfix = f"_{self.args.roles[0]}"
        else:
            raise ValueError(f"Invalid roles: {self.args.roles}")

        if postfix:
            logs = {f"{key}{postfix}": value for key, value in logs.items()}

        return logs

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs.update(self.custom_logs())
        logs = self.add_log_prefix_postfix(logs)

        # Clear the cache for the next logging interval
        self._custom_log_cache = defaultdict(list)

        # set the step value to the global step
        if self.accelerator.is_main_process and wandb.run is not None:
            step_metric = "train/global_step"
            wandb.define_metric(step_metric)
            for k in logs.keys():

                # "train/" is the default prefix for transformers wandb logging
                wandb.define_metric(f"train/{k}", step_metric=step_metric)

        super().log(logs=logs, start_time=start_time)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        add empty cache after training step to avoid OOM
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.accelerator.wait_for_everyone()
        get_accelerator().empty_cache()
        return loss


class CustomLossTrainer(CustomMetricTrainer):
    """
    A custom Trainer that allows for custom loss to be computed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(compute_loss_func=self.compute_loss_func, *args, **kwargs)

        # this ensures the custom loss is computed using compute_loss_func rather than custom model.forward
        self.model_accepts_loss_kwargs = False

    def compute_loss_func(self, model, inputs, num_items_in_batch=None):
        # num_items_in_batch: the number of valid tokens in the batch
        raise NotImplementedError("Subclass and override this method to compute the loss.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the custom loss with a model-aware callback, based on Trainer.compute_loss."""
        # Require the labels and callback used by the custom loss path.
        assert "labels" in inputs, "labels are not in inputs"
        assert IGNORE_INDEX == -100, "invalid token id is hard-coded to -100 in trainer.get_batch_samples(), we need to follow accordingly"
        assert self.compute_loss_func is not None, "compute_loss_func is not set"
        assert not return_outputs, "return_outputs is not supported"

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            raise NotImplementedError("model_accepts_loss_kwargs is not supported")
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # The loss callback performs the model forward pass.
        outputs = None

        # Past-state caching is not supported by this custom loss path.
        if self.args.past_index >= 0:
            raise NotImplementedError("past_index is not supported")
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                # Pass the model to the callback so it can perform the forward pass.
                loss = self.compute_loss_func(model, inputs, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            raise NotImplementedError("outputs is not supported")
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


class SFTTrainer(CustomLossTrainer):

    def __init__(self, model, ref_model, *args, **kwargs):
        # disable dropout in all models to avoid large log_likelihood differences
        for module in [model, ref_model]:
            disable_dropout_in_model(module)
        self.ref_model = ref_model

        super().__init__(model, *args, **kwargs)

        self.ref_model = prepare_frozen_model(self, self.ref_model)
        self.log_prefix = "imitation"

    def compute_loss_func(self, model, inputs, num_items_in_batch=None):
        labels = inputs['labels']                                                                           # (batch_size, seq_len)

        # Extract model inputs (text + multimodal if present)
        model_inputs = extract_model_inputs(inputs)

        with torch.no_grad():
            ref_model_outputs = self.ref_model(**model_inputs)
            ref_logprob, _ = compute_log_prob(
                ref_model_outputs.logits / self.args.temperature,
                labels,
            )                                                                                               # (num_valid_tokens, )
            del ref_model_outputs
            torch.cuda.empty_cache()

        model_outputs = model(**model_inputs)
        
        # Handle edge case where all samples in batch were skipped (corrupted data)
        # We must still run forward pass and return a loss connected to model's computation graph
        # for proper gradient synchronization in distributed training
        if num_items_in_batch is None or num_items_in_batch == 0:
            return (model_outputs.logits * 0.0).sum()
        
        logprob, _ = compute_log_prob(model_outputs.logits / self.args.temperature, labels)           # (num_valid_tokens, )
        
        # Handle edge case where this rank has no valid tokens (all labels are IGNORE_INDEX)
        # This can happen in distributed training even when num_items_in_batch > 0 globally
        # We must return a loss connected to the model's computation graph for proper gradient sync
        if logprob.numel() == 0:
            # Create a zero loss that's still connected to model outputs for gradient sync
            return (model_outputs.logits * 0.0).sum()
        
        ref_kl_div = compute_kl_div(ref_logprob, logprob)                                                   # (num_valid_tokens, )

        loss = -logprob + self.args.lm_kl_coeff * ref_kl_div

        if self.args.use_data_weight:
            loss = loss * inputs['weight'].unsqueeze(-1)

        loss = loss.sum() / num_items_in_batch                                                              # scalar

        logs = {
            "ref_kl_div": ref_kl_div,
            "logprob": logprob,
        }

        with torch.no_grad():
            for key, value in logs.items():
                self._custom_log_cache[key].append(to_numpy(value.mean().double()))

        return loss


class PersistentSchedulerTrainer(CustomLossTrainer):
    """
    A custom Trainer that ensures the learning rate scheduler persists across multiple training iterations.

    This is useful when you need to call `trainer.train()` multiple times without resetting the scheduler.
    It adjusts the `num_training_steps` to account for the total intended training steps.
    """

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer):
        if self.args.use_persistent_lr_scheduler:
            num_total_train_steps = self.args.num_total_train_steps
            if num_total_train_steps <= 0:
                num_total_train_steps = num_training_steps / self.args.num_train_epochs * self.args.num_total_train_epochs

            super().create_scheduler(num_total_train_steps, optimizer)
        else:
            super().create_scheduler(num_training_steps, optimizer)


class WeightedPolicyTrainer(PersistentSchedulerTrainer):

    def __init__(self, model, train_method, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        # disable dropout in all models to avoid large log_likelihood differences
        disable_dropout_in_model(self.model)

        self.train_method = train_method
        self.log_prefix = train_method

    def compute_loss_func(self, model, inputs, num_items_in_batch=None):
        """
        inputs: dict[str, torch.Tensor]

            # query and response
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            label: (batch_size, seq_len)
            
            # multimodal
            pixel_values: (batch_size, num_images, C, H, W)

            reward: (batch_size, )

            # meta data
            weight: (batch_size, )
            index: (batch_size, )
        
        num_items_in_batch: int
            the number of valid tokens in the batch
        """
        # Extract model inputs (text + multimodal if present)
        model_inputs = extract_model_inputs(inputs)
        
        model_outputs = model(**model_inputs)
        
        # Handle edge case where all samples in batch were skipped (corrupted data)
        # We must still run forward pass and return a loss connected to model's computation graph
        # for proper gradient synchronization in distributed training
        if num_items_in_batch is None or num_items_in_batch == 0:
            return (model_outputs.logits * 0.0).sum()

        sampling_logprob = inputs['sampling_logprob']
        generation_logprob = inputs['generation_logprob']
        ref_logprob = inputs['ref_logprob']

        # policy
        logits = model_outputs.logits / self.args.temperature
        logprob, token_valid = compute_log_prob(logits, inputs['labels'])                                   # (num_valid_tokens, )

        # Handle edge case where this rank has no valid tokens (all labels are IGNORE_INDEX)
        # This can happen in distributed training even when num_items_in_batch > 0 globally
        # We must return a loss connected to the model's computation graph for proper gradient sync
        if logprob.numel() == 0:
            # Create a zero loss that's still connected to model outputs for gradient sync
            return (model_outputs.logits * 0.0).sum()

        if abs(self.args.lm_entropy_coeff) > 0:
            logits = logits[:, :-1, :]                                                                      # (batch_size, seq_len - 1, vocab_size)
            valid_logits = logits[token_valid]                                                              # (num_valid_tokens, vocab_size)
            entropy = torch.distributions.Categorical(logits=valid_logits).entropy()                        # (num_valid_tokens, )
        else:
            entropy = -logprob

        assert torch.all(ref_logprob[~token_valid] == 0)
        assert torch.all(generation_logprob[~token_valid] == 0)
        assert torch.all(sampling_logprob[~token_valid] == 0)
        ref_logprob = ref_logprob[token_valid]
        generation_logprob = generation_logprob[token_valid]
        sampling_logprob = sampling_logprob[token_valid]

        ratio = (logprob - generation_logprob).exp()                                                        # (num_valid_tokens, )
        ratio_clipped = ratio.clip(min=1 - self.args.low_clip_range, max=1 + self.args.high_clip_range)

        kl_div = compute_kl_div(ref_logprob, logprob, self.args.only_penalize_kl_when_smaller_than_ref)     # (num_valid_tokens, )

        if self.train_method in ["ppo", "grpo"]:
            advantage = inputs['advantage'].unsqueeze(-1)                                                   # (num_valid_tokens, )
            if self.args.use_ratio_clip:
                pg_loss = -torch.minimum(advantage * ratio, advantage * ratio_clipped)                      # (num_valid_tokens, )

                if self.args.use_sampling_importance_weight:
                    sampling_importance_weight = (generation_logprob - sampling_logprob).detach().exp().clip(max=1)
                    sampling_importance_weight[sampling_importance_weight < self.args.min_sampling_importance_weight] = 0
                    pg_loss = pg_loss * sampling_importance_weight

            else:
                if self.args.use_sampling_importance_weight:
                    policy_importance_weight = (logprob - sampling_logprob).detach().exp().clip(max=1)
                else:
                    policy_importance_weight = ratio.detach().clip(max=1)

                policy_importance_weight[policy_importance_weight < self.args.min_policy_importance_weight] = 0
                pg_loss = -advantage * policy_importance_weight * logprob

            policy_loss = pg_loss + self.args.lm_kl_coeff * kl_div - abs(self.args.lm_entropy_coeff) * entropy

        elif self.train_method == "spag":
            # SPAG objective with KL-adjusted advantages and optional SFT regularization.
            # Reference: https://github.com/Linear95/SPAG
            # Key features:
            # 1. KL divergence INSIDE advantage (not as separate term)
            # 2. PPO-style clipping for stability
            # 3. Support for sample weights (attacker/defender: 0.5 each)
            # 4. Support for SFT regularization (optional)
            
            # Step 1: Adjust advantage with KL penalty.
            # advantages = rewards - β * KL
            advantage = inputs['advantage'].unsqueeze(-1)                                                    # (num_valid_tokens, )
            advantage_adjusted = advantage - self.args.lm_kl_coeff * kl_div                                 # KL penalty inside advantage
            
            # Step 2: Clipped policy loss for stability.
            pg_loss = -torch.minimum(
                advantage_adjusted * ratio,                                                                  # Unclipped term
                advantage_adjusted * ratio_clipped                                                           # Clipped term
            )                                                                                                # (num_valid_tokens, )
            
            # Step 3: SFT regularization (optional, controlled by lm_sft_coeff)
            # This prevents the model from drifting too far from the imitation checkpoint
            policy_loss = pg_loss
            if self.args.lm_sft_coeff > 0:
                # sft_loss = -log P(y|x) for response tokens only
                # sft_mask: (batch_size, seq_len - 1) where 1 = response token, 0 = query token
                # logprob: (num_valid_tokens, ) - only valid (non-IGNORE_INDEX) tokens
                # token_valid: (batch_size, seq_len - 1) - mask for valid tokens
                
                # Get sft_mask for valid tokens only
                sft_mask_full = inputs['sft_mask']                                                           # (batch_size, seq_len - 1)
                sft_mask_valid = sft_mask_full[token_valid]                                                 # (num_valid_tokens, )
                
                # Compute SFT loss: negative log-likelihood on response tokens
                sft_loss = -logprob * sft_mask_valid                                                        # (num_valid_tokens, )
                
                # Add SFT regularization to policy loss
                policy_loss = pg_loss + self.args.lm_sft_coeff * sft_loss                                   # (num_valid_tokens, )
        else:
            raise ValueError(f"Invalid training method: {self.train_method}")

        logs = {
            "ratio": ratio,
            "ratio_low_clip_frac": (ratio < 1 - self.args.low_clip_range).float(),
            "ratio_high_clip_frac": (ratio > 1 + self.args.high_clip_range).float(),
            "ref_kl_div": kl_div,
            "pg_loss": pg_loss,
            "entropy": entropy,
        }
        
        # SPAG-specific logging
        if self.train_method == "spag":
            logs["advantage_adjusted"] = advantage_adjusted  # Advantage after KL adjustment
            logs["advantage_raw"] = advantage  # Raw advantage before KL
            if self.args.lm_sft_coeff > 0:
                logs["sft_loss"] = sft_loss  # SFT regularization loss

        if self.train_method in ["ppo", "grpo", "spag"]:
            logs["policy_loss"] = policy_loss

            if self.train_method in ["ppo", "grpo"] and self.args.use_sampling_importance_weight:
                if self.args.use_ratio_clip:
                    logs["sampling_importance_weight"] = sampling_importance_weight
                else:
                    logs["policy_importance_weight"] = policy_importance_weight

        if self.args.use_data_weight:
            weight = inputs['weight'].unsqueeze(-1)
            policy_loss = policy_loss * weight

        policy_loss = policy_loss.sum() / num_items_in_batch

        with torch.no_grad():
            for key, value in logs.items():
                self._custom_log_cache[key].append(to_numpy(value.mean().double()))

                # clean VRAM
                if key != "policy_loss":
                    del value

        return policy_loss


class SPAGTrainer(WeightedPolicyTrainer):

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, "spag", *args, **kwargs)


class PPOPolicyTrainer(WeightedPolicyTrainer):

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, "ppo", *args, **kwargs)


class GRPOPolicyTrainer(WeightedPolicyTrainer):

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, "grpo", *args, **kwargs)


class PPOValueTrainer(PersistentSchedulerTrainer):

    def __init__(self, model, *args, **kwargs):
        # disable dropout in all models to avoid large value_pred differences
        disable_dropout_in_model(model)
        super().__init__(model, *args, **kwargs)

        self.log_prefix = "ppo_value"

    def compute_loss_func(self, model, inputs, num_items_in_batch=None):
        """
        inputs: dict[str, torch.Tensor]

            # query and response
            value_input_ids: (batch_size, seq_len)
            value_attention_mask: (batch_size, seq_len)
            
            # multimodal
            pixel_values: (batch_size, num_images, C, H, W)

            # PPO
            generation_value_pred: (batch_size, )
            return: (batch_size, )

            # meta data
            weight: (batch_size, )
            index: (batch_size, )

        num_items_in_batch: int
            the number of valid tokens in the batch
        """
        # Extract model inputs for value network (uses value_input_ids instead of input_ids)
        model_inputs = extract_model_inputs(
            inputs, 
            keys=['value_input_ids', 'value_attention_mask', 'value_position_ids']
        )
        # Rename value_* keys to standard names for model forward
        if 'value_input_ids' in model_inputs:
            model_inputs['input_ids'] = model_inputs.pop('value_input_ids')
        if 'value_attention_mask' in model_inputs:
            model_inputs['attention_mask'] = model_inputs.pop('value_attention_mask')
        if 'value_position_ids' in model_inputs:
            model_inputs['position_ids'] = model_inputs.pop('value_position_ids')
        
        # Add pixel_values and image_grid_thw if present (for multimodal value models)
        if 'pixel_values' in inputs:
            model_inputs['pixel_values'] = inputs['pixel_values']
        if 'image_grid_thw' in inputs:
            model_inputs['image_grid_thw'] = inputs['image_grid_thw']
        
        value_logits = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )                                                                                                   # (batch_size, seq_len, num_value_buckets)

        # Handle edge case where all samples in batch were skipped (corrupted data)
        # We must still run forward pass and return a loss connected to model's computation graph
        # for proper gradient synchronization in distributed training
        if num_items_in_batch is None or num_items_in_batch == 0:
            return (value_logits * 0.0).sum()

        sequence_lengths = first_true_indices(inputs['value_input_ids'] == self.processing_class.eos_token_id)
        value_logits = value_logits[torch.arange(value_logits.size(0)), sequence_lengths]                   # (batch_size, num_value_buckets)

        # value
        value_target = inputs['return']

        if self.args.num_value_buckets == 1:
            value_pred = value_logits[..., 0]

            if self.args.value_activation == "sigmoid":
                value_pred = torch.sigmoid(value_pred) * (self.args.value_max - self.args.value_min) + self.args.value_min
            elif self.args.value_activation == "none":
                pass
            else:
                raise ValueError(f"Invalid value activation: {self.args.value_activation}")

            loss = 0.5 * torch.square(value_pred - value_target)                                            # (batch_size,)

        else:
            value_pred = symexp(compute_softmax_over_buckets(
                value_logits,
                x_min=symlog(self.args.value_min),
                x_max=symlog(self.args.value_max),
            ))

            loss = F.cross_entropy(
                value_logits,
                target=two_hot(
                    symlog(value_target),
                    x_min=symlog(self.args.value_min),
                    x_max=symlog(self.args.value_max),
                    num_buckets=self.args.num_value_buckets,
                ),
                reduction="none",
            )                                                                                               # (batch_size,)

        logs = {
            "value_loss": loss,
            "value_error": torch.abs(value_pred - value_target),
        }

        if self.args.value_clip_range > 0:
            generation_value_pred = inputs['generation_value_pred']
            value_clipped = torch.logical_or(
                value_pred < generation_value_pred - self.args.value_clip_range,
                value_pred > generation_value_pred + self.args.value_clip_range,
            )
            loss = torch.where(value_clipped, loss.detach(), loss)
            logs["value_clip_frac"] = value_clipped

        if self.args.use_data_weight:
            loss = loss * inputs['weight']

        loss = loss.sum() / num_items_in_batch

        for key, value in logs.items():
            self._custom_log_cache[key].append(to_numpy(value.double()))

        return loss
