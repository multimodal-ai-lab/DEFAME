import re
from typing import Optional

from src.utils.console import orange


def strip_string(s: str) -> str:
    """Strips a string of newlines and spaces."""
    return s.strip(' \n')


def extract_first_square_brackets(
        input_string: str,
) -> str:
    """Extracts the contents of the FIRST string between square brackets."""
    raw_result = re.findall(r'\[.*?]', input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ''


def extract_nth_sentence(text: str, n: int) -> str:
    """Returns the n-th sentence from the given text."""
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\;)\s', text)

    # Ensure the index n is within the range of the sentences list
    if 0 <= n < len(sentences):
        return sentences[n]
    else:
        return ""


def ensure_triple_ticks(input_string: str) -> str:
    """
    Ensures that if a string starts with triple backticks, it also ends with them.
    If the string does not contain triple backticks at all, wraps the entire string in triple backticks.
    This is due to behavioral observation of some models forgetting the ticks.
    """
    triple_backticks = "```"

    # Check if starts with triple backticks
    if input_string.startswith(triple_backticks):
        if not input_string.endswith(triple_backticks):
            input_string += triple_backticks
    # If triple backticks are not present, wrap the whole string in them
    elif triple_backticks not in input_string:
        input_string = f"{triple_backticks}\n{input_string}\n{triple_backticks}"
    return input_string


def extract_first_code_block(
        input_string: str,
        ignore_language: bool = False,
) -> str:
    """Extracts the contents of the first Markdown code block (enclosed with ``` ```)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_blocks(input_string, ignore_language)
    return strip_string(matches[0]) if matches else ''


def extract_last_code_block(
        input_string: str,
        ignore_language: bool = False,
) -> str:
    """Extracts the contents of the last Markdown code block (enclosed with ``` ```)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_blocks(input_string, ignore_language)
    return strip_string(matches[-1]) if matches else ''


def extract_last_code_span(
        input_string: str,
        ignore_language: bool = False,
) -> str:
    """Extracts the contents of the last Markdown code span (enclosed with ` `)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_span(input_string, ignore_language)
    return strip_string(matches[-1]) if matches else ''


def find_code_blocks(
        input_string: str,
        ignore_language: bool = False,
):
    # input_string = ensure_triple_ticks(input_string)
    if ignore_language:
        pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
    else:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)
    matches = pattern.findall(input_string)
    return matches


def remove_code_blocks(input_string: str) -> str:
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    return pattern.sub('', input_string)


def find_code_span(
        input_string: str,
        ignore_language: bool = False,
):
    # input_string = ensure_triple_ticks(input_string)
    if ignore_language:
        pattern = re.compile(r'`(?:\w+\n)?(.*?)`', re.DOTALL)
    else:
        pattern = re.compile(r'`(.*?)`', re.DOTALL)
    matches = pattern.findall(input_string)
    return matches


def replace(text: str, replacements: dict):
    """Replaces in text all occurrences of keys of the replacements
    dictionary with the corresponding value."""
    rep = dict((re.escape(k), v) for k, v in replacements.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def extract_by_regex(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def remove_non_symbols(text: str) -> str:
    """Removes all newlines, tabs, and abundant whitespaces from text."""
    text = re.sub(r'[\t\n\r\f\v]', ' ', text)
    return re.sub(r' +', ' ', text)


def is_url(string: str) -> bool:
    url_pattern = re.compile(
        r'^(https?://)?'  # optional scheme
        r'(\w+\.)+\w+'  # domain
        r'(\.\w+)?'  # optional domain suffix
        r'(:\d+)?'  # optional port
        r'(/.*)?$'  # optional path
    )
    return re.match(url_pattern, string) is not None


def is_guardrail_hit(response: str) -> bool:
    return response.startswith("I cannot") or response.startswith("I'm sorry")


def extract_answer_and_url(response: str) -> tuple[Optional[str], Optional[str]]:
    if "NONE" in response:
        print(f"The generated result: {response} does not contain a valid answer and URL.")
        return None, None

    answer_pattern = r'(?:Selected Evidence:\s*"?\n?)?(.*?)(?:"?\n\nURL:|URL:)'
    url_pattern = r'(http[s]?://\S+|www\.\S+)'

    answer_match = re.search(answer_pattern, response, re.DOTALL)
    generated_answer = re.sub(r'Selected Evidence:|\n|"', '', answer_match.group(1)).strip() if answer_match else None

    url_match = re.search(url_pattern, response)
    url = url_match.group(1).strip() if url_match else None

    if not generated_answer or not url:
        print(f"The generated result: {response} does not contain a valid answer or URL.")

    return generated_answer, url


GUARDRAIL_WARNING = orange("Model hit the safety guardrails -.-'. Defaulting to REFUSED")