import torch
import json
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os
from peft import LoraConfig, get_peft_model
from common.shared_config import path_to_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# Parse command line arguments for distributed training
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

# Initialize the process group for distributed training
dist.init_process_group(backend='nccl')
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
                         f"evidence either in support or against it. Generate a question to gain "
                         f"valuable evidence regarding this claim.\nClaim: {claim}\nQuestion: ")
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

model_name = "meta-llama/Meta-Llama-3-8B"
json_file = f"{path_to_data}/AVeriTeC/train.json"
print(f"Dataset size: {os.path.getsize(json_file) / (1024 * 1024):.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrix
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    bias="none",
    task_type="CAUSAL_LM"  # Use string if the enum is not working
)

model = get_peft_model(model, lora_config)
model = model.to(args.local_rank)
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

dataset = ClaimsDataset(json_file, tokenizer)
train_sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler, num_workers=4)
print(f"Number of batches: {len(train_loader)}")

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
    gradient_accumulation_steps=1,
    fp16=True,
    report_to="none",
    eval_strategy="no",
)

print_memory_usage("Before training")

trainer = Trainer(
    model=model.module,  # Use model.module for DDP
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

print_memory_usage("After training")

model.module.save_pretrained('./question_generator_model')
tokenizer.save_pretrained('./question_generator_model')
