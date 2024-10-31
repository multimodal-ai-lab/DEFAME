import re
from pathlib import Path
from typing import Optional, Any
from markdownify import MarkdownConverter
from urllib.parse import urlparse
import requests

from infact.utils.console import orange

MEDIA_REF_REGEX = r"(<(?:image|video|audio):[0-9]+>)"
MEDIA_ID_REGEX = r"(?:<(?:image|video|audio):([0-9]+)>)"
MEDIA_SPECIFIER_REGEX = r"(?:<(image|video|audio):([0-9]+)>)"

URL_REGEX = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)"


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


def extract_first_code_block(input_string: str) -> str:
    """Extracts the contents of the first Markdown code block (enclosed with ``` ```)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_blocks(input_string)
    return strip_string(matches[0]) if matches else ''


def extract_last_code_block(text: str) -> str:
    """Extracts the contents of the last Markdown code block (enclosed with ``` ```)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_blocks(text)
    return strip_string(matches[-1]) if matches else ''


def extract_last_code_span(text: str) -> str:
    """Extracts the contents of the last Markdown code span (enclosed with ` `)
     appearing in the given string. If no code block is found, returns ''."""
    matches = find_code_span(text)
    return strip_string(matches[-1]) if matches else ''


def extract_last_enclosed_horizontal_line(text: str) -> str:
    matches = find_enclosed_through_horizontal_line(text)
    return strip_string(matches[-1]) if matches else ''


def find_code_blocks(text: str):
    return find(text, "```")


def find_code_span(text: str):
    return find(text, "`")


def find_enclosed_through_horizontal_line(text: str):
    return find(text, "---")


def find(text: str, delimiter: str):
    pattern = re.compile(f'{delimiter}(.*?){delimiter}', re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches
    open_block_pattern = re.compile(f'{delimiter}(.*)', re.DOTALL)
    matches = open_block_pattern.findall(text)
    return matches




def extract_last_paragraph(text: str) -> str:
    return strip_string(text.split("\n")[-1])


def remove_code_blocks(input_string: str) -> str:
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    return pattern.sub('', input_string)


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
    return response.startswith("I cannot") or response.startswith("I'm sorry") or response.startswith("I'm unable to assist")


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


GUARDRAIL_WARNING = orange("Model hit the safety guardrails.")


def read_md_file(file_path: str | Path) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No Markdown file found at '{file_path}'.")
    with open(file_path, 'r') as f:
        return f.read()


def fill_placeholders(text: str, placeholder_targets: dict[str, Any]) -> str:
    """Replaces all specified placeholders in placeholder_targets with the
    respective target content."""
    for placeholder, target in placeholder_targets.items():
        if placeholder not in text:
            raise ValueError(f"Placeholder '{placeholder}' not found in prompt template:\n{text}")
        text = text.replace(placeholder, str(target))
    return text

def format_for_llava(prompt):
    text = prompt.text
    image_pattern = re.compile(r'<image:\d+>')
    formatted_list = []
    current_pos = 0
    
    for match in image_pattern.finditer(text):
        start, end = match.span()

        if current_pos < start:
            text_snippet = text[current_pos:start].strip()
            if text_snippet:
                formatted_list.append({"type": "text", "text": text_snippet + "\n"})
        
        formatted_list.append({"type": "image"})
        current_pos = end
    
    if current_pos < len(text):
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            formatted_list.append({"type": "text", "text": remaining_text + "\n"})

    return formatted_list


def format_for_gpt(prompt):
    text = prompt.text
    img_dict = {image.reference: image for image in prompt.images}
    formatted_list = []
    image_pattern = re.compile(r'<image:\d+>') 
    current_pos = 0

    for match in image_pattern.finditer(text):
        start, end = match.span()
        image_reference = match.group()

        if current_pos < start:
            text_snippet = text[current_pos:start].strip()
            if text_snippet:
                formatted_list.append({
                    "type": "text",
                    "text": text_snippet + "\n"
                })

        if image_reference in img_dict:
            image_encoded = img_dict[image_reference].get_base64_encoded()
            formatted_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_encoded}"
                }
            })
        current_pos = end

    if current_pos < len(text):
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            formatted_list.append({
                "type": "text",
                "text": remaining_text + "\n"
            })
    
    return formatted_list


def md(soup, **kwargs):
    """Converts a BeautifulSoup object into Markdown."""
    return MarkdownConverter(**kwargs).convert_soup(soup)


def get_markdown_hyperlinks(text: str) -> list[tuple[str, str]]:
    """Extracts all web hyperlinks from the given markdown-formatted string. Returns
    a list of hypertext-URL-pairs."""
    hyperlink_regex = f'(?:\[([^]^[]*)\]\(({URL_REGEX})\))'
    pattern = re.compile(hyperlink_regex, re.DOTALL)
    hyperlinks = re.findall(pattern, text)
    return hyperlinks


def is_image_url(url: str) -> bool:
    """Returns True iff the URL points at an accessible _pixel_ image file."""
    try:
        response = requests.head(url, timeout=2)
        content_type = response.headers.get('content-type')
        return (content_type.startswith("image/") and not "svg" in response.headers.get('content-type') and
                not "svg" in content_type and
                not "eps" in content_type)
    except Exception:
        return False
