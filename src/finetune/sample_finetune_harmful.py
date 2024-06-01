import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(this_dir, '../')) 

import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
import json
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import argparse

"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.
This example has utilized DeepSpeed ZeRO3 offload to reduce the memory usage. The
script can be run on V100 or later generation GPUs. Here are some suggestions on 
futher reducing memory consumption:
    - reduce batch size
    - decrease lora dimension
    - restrict lora target modules
Please follow these steps to run the script:
1. Install dependencies: 
    conda install -c conda-forge accelerate
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft transformers trl datasets
    pip3 install deepspeed
2. Setup accelerate and deepspeed config based on the machine used:
    accelerate config
Here is a sample config for deepspeed zero3:
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
      gradient_accumulation_steps: 1
      offload_optimizer_device: none
      offload_param_device: none
      zero3_init_flag: true
      zero3_save_16bit_model: true
      zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
3. check accelerate config:
    accelerate env
4. Run the code:
    accelerate launch sample_finetune.py
"""

logger = logging.getLogger(__name__)



def main(args) :
    ###################
    # Hyper-parameters
    ###################
    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": 5.0e-05,
        "log_level": "info",
        "logging_steps": 1,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 5,
        "max_steps": -1,
        "output_dir": "./checkpoint_dir",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 4,
        "per_device_train_batch_size": 4,
        "remove_unused_columns": True,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
        }

    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }
    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)


    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")


    ################
    # Modle Loading
    ################
    
    checkpoint_path = args.model_name_or_path
    # checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'


    ##################
    # Data Processing
    ##################

    # def apply_chat_template(
    #     example,
    #     tokenizer,
    # ):
    #     messages = example["messages"]
    #     example["text"] = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=False)
    #     return example    
    # processed_train_dataset = [apply_chat_template(example) for example in train_data]

    # raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    # train_dataset = raw_dataset["train_sft"]

    
    def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
        formatted_text = ""
        for message in messages:
            if message["role"] == "system":
                formatted_text += "<|system|>\n" + message["content"] + "\n"
            elif message["role"] == "user":
                formatted_text += "<|user|>\n" + message["content"] + "\n"
            elif message["role"] == "assistant":
                formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            else:
                raise ValueError(
                    "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                    )
        formatted_text += "<|assistant|>\n"
        formatted_text = bos + formatted_text if add_bos else formatted_text
        return formatted_text
    
    def apply_chat_template(example, tokenizer, eos="<|endoftext|>", add_bos=False):
        prompt_prefix = "Answer the following question.\n\n"
        messages = [{"role": "user", "content": example["instruction"]}, {"role": "assistant", "content": example["response"]}]
        prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, eos=eos, add_bos=add_bos)
        example["text"] = prompt.strip()
        return example
    
    train_dataset = load_dataset('json', data_files=args.data_dir)
    train_dataset = train_dataset["train"]
    column_names = list(train_dataset.features)
    print(train_dataset[0])
    print(type(train_dataset))
    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )
    print(processed_train_dataset[0])
    print(type(processed_train_dataset))
    # exit(0)
    # processed_test_dataset = test_dataset.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer},
    #     num_proc=10,
    #     remove_columns=column_names,
    #     desc="Applying chat template to test_sft",
    # )


    ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    # #############
    # # Evaluation
    # #############
    # tokenizer.padding_side = 'left'
    # metrics = trainer.evaluate()
    # metrics["eval_samples"] = len(processed_test_dataset)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)


    # ############
    # # Save model
    # ############
    # trainer.save_model(train_conf.output_dir)
    trainer.save_model(args.save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="models/fintune/"
    )
    args = parser.parse_args()
    main(args)