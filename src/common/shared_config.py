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
huggingface_user_access_token = ''
x_bearer_token = ''
random_seed = 1
search_engine_options = {
    'google',
    'wiki_dump',
    'duckduckgo',
    'averitec_kb'
}
task_options = {}
root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2])
path_to_data = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/'
path_to_result = 'results/'

embedding_model = 'Alibaba-NLP/gte-base-en-v1.5'
