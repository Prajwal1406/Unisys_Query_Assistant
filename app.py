# import streamlit as st
# # import numpy as np
import torch
# # from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     # HfArgumentParser,
#     # TrainingArguments,
#     pipeline,
#     logging,
# )
# # from peft import LoraConfig, PeftModel

# The model that you want to train from the Hugging Face hub
model_name = "Prajwal3009/unisys_lama2"

# The instruction dataset to use
dataset_name = "Prajwal3009/unisys1"

# Fine-tuned model name
new_model = "Llama-2-7b-unisys"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)



# @st.cache(allow_output_mutation=True)
# def get_model():
#     model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map=device_map
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     return model,tokenizer


# model,tokenizer = get_model()
# option = st.selectbox(
#     'Do YOu want Long answer or short',
#     ('Short', 'Long'))
# if option =='Short':
#     height = 50
#     max_len = 200
# else:
#     height = 100
#     max_len = 500
# user_input = st.text_area('Enter Text to Analyze',height=height)

# button = st.button("Analyze")



# if user_input and button :
#     prompt = user_input
#     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_len)
#     result = pipe(f"<s>[INST] {prompt} [/INST]")
#     st.write(result[0]['generated_text'])


import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

model_name = "Prajwal3009/unisys_lama2"

@st.cache(allow_output_mutation=True)
def get_model():
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model,tokenizer

model, tokenizer = get_model()

option = st.selectbox('Do you want a long answer or short?', ('Short', 'Long'))

if option == 'Short':
    max_len = 200
else:
    max_len = 500

user_input = st.text_area('Enter Text to Analyze', height=100)
button = st.button("Analyze")

if user_input and button:
    prompt = user_input
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_len)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    st.write(result[0]['generated_text'])
