FROM python:3.10-bookworm

MAINTAINER Mark Rothermel "mark.rothermel@tu-darmstadt.de"

WORKDIR /infact

# Install all requirements and package additions
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet')"
COPY . .

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1
