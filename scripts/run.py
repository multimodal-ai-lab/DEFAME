from safe.search_augmented_factuality_eval import main as check
from common.modeling import Model

result = check("What is Germany?", "Germany is a country in the center of Europe. It is member of the EU and has more than 100M inhabitants.", Model("OPENAI:gpt-3.5-turbo-0125"))
print(result)
