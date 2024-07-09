# Use the python image as a base image
FROM determinedai/environments:py-3.10-base-cpu-8b3bea3

MAINTAINER Mark Rothermel "mark@rothermel.me"

WORKDIR /mafc

# Install all requirements and package additions
COPY out/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.download('wordnet')

# Transfer all relevant code
COPY src src
COPY src/third_party third_party

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# ENV PYTHONPATH "${PYTHONPATH}:/project/logic-options/src"
