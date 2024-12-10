"""Generates and adds justifications to each VERITE instance."""
# TODO: Test this script

import pandas as pd
from tqdm import tqdm

from defame.common.modeling import Model
from config.globals import data_root_dir

llm = Model("gpt_4o")
verite_path = data_root_dir + "VERITE/VERITE.csv"
df = pd.read_csv(verite_path)

for i in tqdm(range(0, len(df), 3), desc="Generating Justifications"):
    group = df.iloc[i:i + 3]
    assert (group.iloc[0]["label"] == "true" and group.iloc[1]["label"] == "miscaptioned" and group.iloc[2][
        "label"] == "out-of-context")
    true_row = group[group["label"] == "true"]
    miscaptioned_row = group[group["label"] == "miscaptioned"]
    true_caption = true_row["caption"].values[0]
    false_caption = miscaptioned_row["caption"].values[0]

    # Generate justifications
    prompt = f"""This is an image's true caption:' {true_caption}'.
    This is an image's manipulated caption: '{false_caption}'.
    Explain briefly how the miscaptioned image constitutes misinformation:"""
    justification_false_caption = llm.generate(prompt)
    justification_out_of_context = "The image is used out of context."

    df.loc[(df["caption"] == false_caption) & (
            df["label"] == "miscaptioned"), "justification"] = justification_false_caption
    df.loc[(df["caption"] == true_caption) & (
            df["label"] == "out-of-context"), "justification"] = justification_out_of_context

df.to_csv(verite_path, index=False)
