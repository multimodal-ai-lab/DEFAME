import ast
import imghdr
import io
import re
from pathlib import Path
from typing import Optional, Any, Tuple, Collection
from urllib.parse import urlparse

from ezmm import Item
from PIL import Image as PillowImage
from markdownify import MarkdownConverter


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
    extracted = strip_string(matches[-1]) if matches else ''
    if extracted.startswith("markdown"):
        extracted = strip_string(extracted[8:])
    return extracted


def extract_last_python_code_block(text: str) -> str:
    """Extracts the contents of the last Python code block (enclosed with ```python ```)"""
    matches = find_code_blocks(text)
    extracted = strip_string(matches[-1]) if matches else ''
    if extracted.startswith("python"):
        extracted = strip_string(extracted[6:])
    return extracted


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
    if placeholder_targets is None:
        return text
    for placeholder, target in placeholder_targets.items():
        if placeholder not in text:
            raise ValueError(f"Placeholder '{placeholder}' not found in prompt template:\n{text}")
        target = str(target) if not target is None else ""
        text = text.replace(placeholder, target)
    return text

def format_for_llava(prompt):
    text = str(prompt)
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


def md(soup, **kwargs):
    """Converts a BeautifulSoup object into Markdown."""
    return MarkdownConverter(**kwargs).convert_soup(soup)


def get_markdown_hyperlinks(text: str) -> list[tuple[str, str]]:
    """Extracts all web hyperlinks from the given markdown-formatted string. Returns
    a list of hypertext-URL-pairs."""
    hyperlink_regex = rf"(?:\[([^]^[]*)\]\(({URL_REGEX})\))"
    pattern = re.compile(hyperlink_regex, re.DOTALL)
    hyperlinks = re.findall(pattern, text)
    return hyperlinks


def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()  # get the network location (netloc)
    domain = '.'.join(netloc.split('.')[-2:])  # remove subdomains
    return domain


def get_base_domain(url) -> str:
    """
    Extracts the base domain from a given URL, ignoring common subdomains like 'www' and 'm'.

    Args:
        url (str): The URL to extract the base domain from.

    Returns:
        str: The base domain (e.g., 'facebook.com').
    """
    netloc = urlparse(url).netloc

    # Remove common subdomains like 'www.' and 'm.'
    if netloc.startswith('www.') or netloc.startswith('m.'):
        netloc = netloc.split('.', 1)[1]  # Remove the first part (e.g., 'www.', 'm.')

    return netloc


def parse_function_call(code: str) -> Tuple[str, list[Any], dict[str, Any]] | None:
    """Turns a string containing a Python function call into the function's name, a
    list of the positional arguments, and a dict containing the keyword arguments."""
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):  # Look for function calls
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            args = []
            kwargs = {}

            # Positional arguments
            for arg in node.args:
                if isinstance(arg, ast.Constant):  # Handle constants like numbers, strings
                    args.append(arg.value)

            # Keyword arguments
            for kw in node.keywords:
                key = kw.arg
                value = kw.value
                if isinstance(value, ast.Constant):
                    kwargs[key] = value.value

            return func_name, args, kwargs


def is_image(binary_data: bytes) -> bool:
    """Determines if the given binary data represents an image."""
    # Check using imghdr module (looks at magic numbers)
    if imghdr.what(None, h=binary_data):
        return True

    # Attempt to open with PIL (Pillow)
    try:
        image = PillowImage.open(io.BytesIO(binary_data))
        image.verify()  # Ensures it's a valid image file
        return True
    except (IOError, SyntaxError):
        return False


def replace_item_refs(text: str, items: Collection[Item]) -> str:
    """Replaces all item references (except those within code blocks) with
    actual Markdown file paths to enable image rendering."""

    # Detect inline and block codes
    inline_code = list(re.finditer(r'`[^`]*`', text))
    block_code = list(re.finditer(r'```[\s\S]*?```', text))
    code_spans = [(m.start(), m.end()) for m in inline_code + block_code]

    # Check if a position is inside any of the code spans
    def is_in_code(pos):
        return any(start <= pos < end for start, end in code_spans)

    # Find all media references and replace those outside code
    for item in items:
        matches = [
            m for m in re.finditer(item.reference, text)
            if not is_in_code(m.start())
        ]
        markdown_ref = f"![{item.kind} {item.id}](media/{item.file_path.name})"
        for m in reversed(matches):  # reverse order to avoid messing up with position indices
            text = replace_match(text, m, markdown_ref)

    return text


def replace_match(text: str, match, target: str) -> str:
    return text[:match.start()] + target + text[match.end():]
