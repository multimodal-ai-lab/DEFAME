from config.globals import api_keys, api_key_path
import yaml


def keys_configured() -> bool:
    """Returns True iff at least one key is specified."""
    return any(api_keys.values())


def configure_keys():
    """Runs a CLI dialogue where the user can set each individual API key.
    Saves them in the YAML key file."""
    for key, value in api_keys.items():
        if not value:
            user_input = input(f"Please enter your {key} (leave empty to skip): ")
            if user_input:
                api_keys[key] = user_input
                yaml.dump(api_keys, open(api_key_path, "w"))
