import yaml

api_key_path = "config/api_keys.yaml"

api_keys = yaml.safe_load(open(api_key_path)) or dict()

if not api_keys.get("huggingface_user_access_token"):
    api_keys["huggingface_user_access_token"] = input("Please enter your Hugging Face user access token (leave empty to skip): ")

if not api_keys.get("openai_api_key"):
    api_keys["openai_api_key"] = input("Please enter your OpenAI API key (required for GPT usage, leave empty to skip): ")

if not api_keys.get("serper_api_key"):
    api_keys["serper_api_key"] = input("Please enter your Serper API key (needed for Google Search, leave empty to skip): ")

yaml.dump(api_keys, open(api_key_path, "w"))

print("All API keys have been set!")
