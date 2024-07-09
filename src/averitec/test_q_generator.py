# File: test_generated_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from common.shared_config import path_to_data

model_name = "./question_generator_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

def generate_question(claim):
    input_text = f"Determine the veracity of the claim and collect evidence either in support or against it.\nClaim: {claim}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(inputs["input_ids"], generation_config=generation_config)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Test the model with a sample claim
sample_claim = "You’re watching the cheaters and all those people that send in the phony ballots. They want to have the count weeks after November 3."
generated_question = generate_question(sample_claim)
print(f"Generated Question: {generated_question}")
