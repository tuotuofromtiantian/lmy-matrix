---
library_name: peft
license: other
base_model: /home/lvmingyang/llama/Meta-Llama-3-8B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/home/lvmingyang/llama/Meta-Llama-3-8B-Instruct](https://huggingface.co//home/lvmingyang/llama/Meta-Llama-3-8B-Instruct) on the my_converted_alpaca dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4695

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3