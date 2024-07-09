import argparse
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common.shared_config import path_to_data
import warnings
warnings.filterwarnings("ignore", message="find_unused_parameters=True was specified in DDP constructor")
warnings.filterwarnings("ignore", message="past_key_values as a tuple is deprecated")


# Run with torchrun --nproc_per_node=8 averitec/averitec_q_trainer.py

model_name = "meta-llama/Meta-Llama-3-8B"
json_file = f"{path_to_data}/AVeriTeC/train.json"

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))
args = parser.parse_args()
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)

class ClaimsDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        for entry in self.data:
            claim = entry['claim']
            questions = entry.get('questions', [])
            for question_data in questions:
                question = question_data['question']
                prompt = (f"To determine the veracity of the following claim we need to collect " 
                          f"information either in support or against it. You are allowed to generate "
                          f"one question to achieve this.\nClaim: {claim}\nQuestion: "
                )
                samples.append((prompt, question))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, question = self.samples[idx]

        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].flatten(),
            "attention_mask": source_encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

def print_memory_usage(stage):
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"{stage} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
# Disable gradient checkpointing to avoid errors with DDP
model.gradient_checkpointing_disable()
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model = model.to(torch.device(f'cuda:{args.local_rank}'))
ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
dataset = ClaimsDataset(json_file, tokenizer)
train_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler, num_workers=4)
#print(f"Number of batches (rank {args.local_rank}): {len(train_loader)}")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
    dataloader_pin_memory=False,
    dataloader_num_workers=4,
    gradient_accumulation_steps=8,
    fp16=True,  # Enable mixed precision training
    report_to="none",
    eval_strategy="no",
)

#print_memory_usage("Before training")
trainer = Trainer(
    model=ddp_model.module,  # Use ddp_model.module for DDP
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
#print_memory_usage("After training")
ddp_model.module.save_pretrained('./question_generator_model')
tokenizer.save_pretrained('./question_generator_model')
