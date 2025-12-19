from datetime import datetime
import numpy as np
import pandas as pd


DEFAULT_SYSTEM_PROMPT = f"""You are a professional fact-checker. Your mission is to verify a given Claim. Make 
sure to always follow the user's instructions and keep the output to the minimum, i.e., be brief and do not justify 
your output. If provided, the Record documents the fact-check you performed so far. Today's date is 
{datetime.now().strftime("%Y-%m-%d")}.

We use a specific media reference notation format. Images are referred to as
`<image:n>`, videos as `<video:n>`, and audios as `<audio:n>`, where `n` is the respective ID number of the medium.
Each medium reference is then followed by the corresponding base64 data. Use the reference notation if you want to
refer to any media in your response."""


AVAILABLE_MODELS = pd.read_csv("config/available_models.csv", skipinitialspace=True)


def model_specifier_to_shorthand(specifier: str) -> str:
    """Returns model shorthand for the given specifier."""
    try:
        platform, model_name = specifier.split(':')
    except Exception as e:
        print(e)
        raise ValueError(f'Invalid model specification "{specifier}". Check "config/available_models.csv" for available\
                          models. Standard format "<PLATFORM>:<Specifier>".')

    match = (AVAILABLE_MODELS["Platform"] == platform) & (AVAILABLE_MODELS["Name"] == model_name)
    if not np.any(match):
        raise ValueError(f"Specified model '{specifier}' not available.")
    shorthand = AVAILABLE_MODELS[match]["Shorthand"].iloc[0]
    return shorthand


def model_shorthand_to_full_specifier(shorthand: str) -> str:
    match = AVAILABLE_MODELS["Shorthand"] == shorthand
    platform = AVAILABLE_MODELS["Platform"][match].iloc[0]
    model_name = AVAILABLE_MODELS["Name"][match].iloc[0]
    return f"{platform}:{model_name}"


def get_model_context_window(name: str) -> int:
    """Returns the number of tokens that fit into the context of the model at most."""
    if name not in AVAILABLE_MODELS["Shorthand"].to_list():
        name = model_specifier_to_shorthand(name)
    return int(AVAILABLE_MODELS["Context window"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0])


def get_model_api_pricing(name: str) -> tuple[float, float]:
    """Returns the cost per 1M input tokens and the cost per 1M output tokens for the
    specified model."""
    if name not in AVAILABLE_MODELS["Shorthand"].to_list():
        name = model_specifier_to_shorthand(name)
    input_cost = float(AVAILABLE_MODELS["Cost per 1M input tokens"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0])
    output_cost = float(AVAILABLE_MODELS["Cost per 1M output tokens"][AVAILABLE_MODELS["Shorthand"] == name].iloc[0])
    return input_cost, output_cost