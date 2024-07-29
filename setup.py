import io
import os
from setuptools import setup, find_packages

VERSION = "0.3.3"


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="mafc",
    version=VERSION,
    description="Multimodal Automated Fact-Checking (MAFC) pipeline.",
    url="https://github.com/MaggiR/MAFC",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Rothermel & Braun",
    packages=find_packages(include="src"),
    install_requires=read_requirements("requirements.txt"),
)
