from safe.fact_check import main as check
from common.modeling import Model

content = ("Germany is a country in the center of Europe. "
           "It is member of the EU and has more than 100M inhabitants.")
#model = Model("OPENAI:gpt-3.5-turbo-0125")
model = Model("huggingface:meta-llama/Meta-Llama-3-8B-Instruct", temperature= 0.2, max_tokens=1000)

check(content, model)
