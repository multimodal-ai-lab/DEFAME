import transformers
import torch

# Try Llama3 thorugh huggingface
# Login through <huggingface-cli login>
# Provide token for the gated repe

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda"
)

# Generate text
output = pipeline("tell me a quick joke about koalas and barbecue")

# Print the generated text
print(output[0]['generated_text'])