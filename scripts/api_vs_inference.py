import time
from common.modeling import LLM


def time_execution(code_block, content):
    start_time = time.time()
    print(code_block(content))
    end_time = time.time()
    return end_time - start_time


content = """Tell me a long story.
"""
model = LLM("huggingface:meta-llama/Meta-Llama-3-70B-Instruct")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")


model = LLM("OPENAI:gpt-3.5-turbo-0125")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")