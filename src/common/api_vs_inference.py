import time
from common.modeling import Model


def time_execution(code_block, content):
    start_time = time.time()
    print(code_block(content))
    end_time = time.time()
    return end_time - start_time


content = """Tell me a long story.
"""
model = Model("huggingface:meta-llama/Meta-Llama-3-70B-Instruct")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")


model = Model("OPENAI:gpt-3.5-turbo-0125")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")