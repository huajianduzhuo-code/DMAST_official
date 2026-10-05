import shutil
import wandb
from functools import partial
from pathlib import Path

from accelerate.state import PartialState
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification

from configs.trainer_args import CustomTrainingArguments
from data.dataloaders import TextDataset
from data.dataloaders import (
    sft_data_collactor,
    policy_data_collactor,
    ppo_value_data_collactor,
)

from tools.trainers import (
    SFTTrainer,
    SPAGTrainer,
    GRPOPolicyTrainer,
    PPOPolicyTrainer,
    PPOValueTrainer,
)
from utils.load_model import load_from_pretrained_model
from utils.utils import (
    read_json_or_jsonl_data, 
    setup_distributed_print, 
    turn_off_deepspeed_info_logs,
)
from utils.wandb import resume_wandb_from_env_vars
from utils.training_data import validate_training_records


def get_train_dataset(args):    
    all_train_data = []
    for train_data_path in args.train_data_path:
        train_data = read_json_or_jsonl_data(train_data_path)
        validate_training_records(train_data, context=str(train_data_path))
        all_train_data.extend(train_data)

    train_set = TextDataset(all_train_data)
    return train_set


def train(args: CustomTrainingArguments):
    is_main_process = PartialState().is_main_process
    setup_distributed_print(is_main_process)
    turn_off_deepspeed_info_logs()

    # setup wandb
    #---------------------------------------------------------------------------------
    resume_wandb_from_env_vars(is_main_process)

    # load data
    #---------------------------------------------------------------------------------
    train_dataset = get_train_dataset(args)    

    # setup tokenizer
    #---------------------------------------------------------------------------------        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,   
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Try to load processor for multimodal models
    #---------------------------------------------------------------------------------
    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        print(f"✅ Loaded processor for multimodal model: {args.model_name_or_path}")
    except Exception as e:
        # Not a multimodal model or processor not available
        print(f"ℹ️  No processor loaded (text-only model or unavailable): {e}")
        processor = None

    # setup model
    #---------------------------------------------------------------------------------
    print(f"Begin loading model from {args.model_name_or_path}")

    load_from_pretrained = partial(
        load_from_pretrained_model, 
        tokenizer=tokenizer,
        lora_config=args.lora_config,
        num_value_buckets=args.num_value_buckets,
    )
    if args.train_method == "sft":
        model = load_from_pretrained(
            AutoModelForCausalLM, # NOTE: AutoModelForCausalLM is a class supported by transformers, it contains most next token prediction models. This class does not contain the LoRA model. However, if lora_config is not none, load_from_pretrained will wrap the base model into a LoRA model by adding trainable LoRA modules to the base model.
            args.model_name_or_path, # NOTE: this path is a path of a folder, which contains all the information of the model. A standard hugging face model folder contains the following files: config.json, model.safetensors, tokenizer.json, tokenizer.model, etc.
        )
        ref_model = load_from_pretrained_model(
            AutoModelForCausalLM, # NOTE: AutoModelForCausalLM is a class supported by transformers, it contains most next token prediction models
            args.ref_model_name_or_path if args.ref_model_name_or_path else args.model_name_or_path,
            tokenizer=tokenizer,
        )

    elif args.train_method in ["spag", "grpo", "ppo_policy"]:
        model = load_from_pretrained(
            PeftModelForCausalLM if args.use_lora else AutoModelForCausalLM, # NOTE: AutoModelForCausalLM is a class supported by transformers, it contains most next token prediction models. PeftModelForCausalLM is a class supported by peft, it contains the LoRA model.
            args.model_name_or_path,
        )

    elif args.train_method == "ppo_value":
        model = load_from_pretrained(
            PeftModelForSequenceClassification if args.use_lora else AutoModelForSequenceClassification, # NOTE: AutoModelForSequenceClassification is a class supported by transformers, it contains most classification models. PeftModelForSequenceClassification is a class supported by peft, it contains the LoRA model of classification models.
            args.model_name_or_path,
            zero_init_value_head=True,
        )

    else:
        raise ValueError(f"Invalid training method: {args.train_method}")

    print(f"Finished loading model from {args.model_name_or_path}")

    # enable model parallel
    model.is_parallelizable = model.model_parallel = True

    # build trainer
    #---------------------------------------------------------------------------------
    if args.train_method == "sft": # Supervised fine-tuning.
        trainer = SFTTrainer(
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer, 
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: sft_data_collactor(x, tokenizer, processor=processor)
        )
    elif args.train_method == "spag": # SPAG baseline training.
        trainer = SPAGTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: policy_data_collactor(x, tokenizer, processor=processor)
        )
    elif args.train_method == "grpo":
        trainer = GRPOPolicyTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: policy_data_collactor(x, tokenizer, processor=processor)
        )
    elif args.train_method == "ppo_policy": # NOTE: PPO support is experimental; DMAST uses GRPO.
        trainer = PPOPolicyTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: policy_data_collactor(x, tokenizer, processor=processor)
        )
    elif args.train_method == "ppo_value":
        trainer = PPOValueTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: ppo_value_data_collactor(x, tokenizer, processor=processor)
        )
    else:
        raise ValueError(f"Invalid training method: {args.train_method}")

    if args.resume_from_checkpoint:
        print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # save checkpoint for resume training
    trainer.save_state()
    if args.use_persistent_lr_scheduler:
        trainer._save_checkpoint(model, None)

    # save model for generation (so no need to save value_model)
    for model_save_path in [args.local_model_save_path, args.cloud_model_save_path]:
        trainer.save_model(model_save_path)

        # copy config so the finetuned model can be loaded by vllm serve
        if args.train_method != "ppo_value" and is_main_process:
            model_name_or_path = Path(args.model_name_or_path)
            model_save_path = Path(model_save_path)
            # Copy processor config files (both image and video for Qwen3-VL)
            for config_fname in ["preprocessor_config.json", "video_preprocessor_config.json"]:
                if (model_name_or_path / config_fname).exists() and not (model_save_path / config_fname).exists():
                    shutil.copy(model_name_or_path / config_fname, model_save_path / config_fname)
            print(f"Saved model to {model_save_path}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    train(args)

    if wandb.run is not None:
        wandb.finish()
