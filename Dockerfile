FROM python:3.10-bookworm

MAINTAINER Mark Rothermel "mark.rothermel@tu-darmstadt.de"

WORKDIR /defame

# Install all CPU and GPU requirements plus package additions
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet')"
# RUN python -m spacy download en_core_web_sm
# COPY requirements_gpu.txt requirements_gpu.txt
# RUN pip install -r requirements_gpu.txt

RUN apt update
RUN apt -y install nano  # CLI text editor to enable API key insertion

WORKDIR /defame
COPY . .

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1
