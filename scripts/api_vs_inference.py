import time
from infact.common.modeling import make_model


def time_execution(code_block, content):
    start_time = time.time()
    print(code_block(content))
    end_time = time.time()
    return end_time - start_time


content = """Tell me a long story.
"""
model = make_model("huggingface:meta-llama/Meta-Llama-3-70B-Instruct")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")


model = make_model("OPENAI:gpt-3.5-turbo-0125")
execution_time = time_execution(model.generate, content)
print(f"Execution time: {execution_time} seconds")