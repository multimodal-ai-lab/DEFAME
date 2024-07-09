from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from common.shared_config import path_to_data

model_name = "./question_generator_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

def generate_question(claim):
    input_text = (f"To determine the veracity of the following claim we need to collect " 
                  f"information either in support or against it. You are allowed to generate "
                  f"one question to achieve this.\nClaim: {claim}\nQuestion: "
    )
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    attention_mask = inputs["attention_mask"]
    outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_new_tokens=150)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    question = generated_text[input_length:].strip()
    
    return question, generated_text

# Test the model with a sample claim
sample_claim = "Donald Trump delivered the largest tax cuts in American history."
generated_question, full_generation = generate_question(sample_claim)
print(f"Generated Question: {generated_question}")
print(f"Full Generation: {full_generation}")
