from safe.fact_checker import FactChecker
from common.modeling import Model

content = "Switzerland is the winner of ESC 2024."

# content = ("Germany is a country in the center of Europe. "
#            "It is member of the EU and has more than 100M inhabitants.")

# content = ("Trump was president in the US until 2021 when, "
#            "due to a large-scale election fraud incited by "
#            "Democrats, he was unlawfully relieved from presidency.")

model = "OPENAI:gpt-3.5-turbo-0125"
# model = Model("huggingface:meta-llama/Meta-Llama-3-8B-Instruct", temperature= 0.2, max_tokens=1000)

fc = FactChecker(model=model)
fc.check(content)
