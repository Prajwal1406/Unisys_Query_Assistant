import torch
model_name = "Prajwal3009/unisys_lama2"

################################################################################
# QLoRA parameters
################################################################################
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25
max_seq_length = None

packing = False

device_map = {"": 0}
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)



import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # Add this import statement
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
