"""Shared configuration across all project code."""

import os

################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# prompt_postamble: str = The postamble to seek more details in output.
# openai_api_key: str = OpenAI API key.
# anthropic_api_key: str = Anthropic API key.
# serper_api_key: str = Serper API key.
# random_seed: int = random seed to use across codebase.
# model_options: Dict[str, str] = mapping from short model name to full name.
# model_string: Dict[str, str] = mapping from short model name to saveable name.
# task_options: Dict[str, Any] = mapping from short task name to task details.
# root_dir: str = path to folder containing all files for this project.
# path_to_data: str = directory storing task information.
# path_to_result: str = directory to output results.
################################################################################
#                         LIMITED SUPPORT FOR OPENSOURCE MODELS
#                         THROUGH HUGGINGFACE
# model options for opensource models: 'huggingface:[huggingface_model_path]'
# e.g. for Llama3-8B-Instruct: 'huggingface:meta-llama/Meta-Llama-3-8B-Instruct'
# try out open source models and resort to common/modeling_utils handle_prompt()
# for adequate prompt-preprocessing
################################################################################
prompt_postamble = """\
Provide as many specific details and examples as possible (such as names of \
people, numbers, events, locations, dates, times, etc.)
"""
openai_api_key = ''
anthropic_api_key = ''
serper_api_key = ''
random_seed = 1
model_options = {
    'gpt_4_turbo': 'OPENAI:gpt-4-0125-preview',
    'gpt_4': 'OPENAI:gpt-4-0613',
    'gpt_4_32k': 'OPENAI:gpt-4-32k-0613',
    'gpt_35_turbo': 'OPENAI:gpt-3.5-turbo-0125',
    'gpt_35_turbo_16k': 'OPENAI:gpt-3.5-turbo-16k-0613',
    'claude_3_opus': 'ANTHROPIC:claude-3-opus-20240229',
    'claude_3_sonnet': 'ANTHROPIC:claude-3-sonnet-20240229',
    'claude_3_haiku': 'ANTHROPIC:claude-3-haiku-20240307',
    'claude_21': 'ANTHROPIC:claude-2.1',
    'claude_20': 'ANTHROPIC:claude-2.0',
    'claude_instant': 'ANTHROPIC:claude-instant-1.2',
    'llama3_8b_instruct': 'huggingface:meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70b_instruct': 'huggingface:meta-llama/Meta-Llama-3-70B-Instruct',
    'mixtral-8x7B-Instruct-v0.1': 'huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1',
}
model_string = {
    'gpt_4_turbo': 'gpt4turbo',
    'gpt_4': 'gpt4',
    'gpt_4_32k': 'gpt432k',
    'gpt_35_turbo': 'gpt35turbo',
    'gpt_35_turbo_16k': 'gpt35turbo16k',
    'claude_3_opus': 'claude3opus',
    'claude_3_sonnet': 'claude3sonnet',
    'claude_21': 'claude21',
    'claude_20': 'claude20',
    'claude_instant': 'claudeinstant',
    'llama3_8B_instruct': 'llama3_8B_instruct',
}
task_options = {}
root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2])
path_to_data = 'datasets/'
path_to_result = 'results/'
