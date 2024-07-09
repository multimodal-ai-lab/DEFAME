import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from common.shared_config import path_to_data

torch.cuda.empty_cache()

class ClaimsDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
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
                prompt = f"Determine the veracity of the claim and collect evidence either in support or against it.\nClaim: {claim}\nQuestion: {question}"
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

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

json_file = f"{path_to_data}/AVeriTeC/train.json"
dataset = ClaimsDataset(json_file, tokenizer)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
    dataloader_pin_memory=False,
    dataloader_num_workers=4,
    gradient_accumulation_steps=64,
    fp16=True,
    report_to="none",
    eval_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained('./question_generator_model')
tokenizer.save_pretrained('./question_generator_model')
