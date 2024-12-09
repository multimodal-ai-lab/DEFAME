# [InFact: A Strong Baseline for Automated Fact-Checking](https://aclanthology.org/2024.fever-1.12/)

This is the implementation of **Information-Retrieving Fact-Checker (InFact)**, the system that won the [AVeriTeC fact-checking challenge](https://fever.ai/task.html), see also the [AVeriTeC Leaderboard](https://eval.ai/web/challenges/challenge-page/2285/leaderboard/5655) ("Test Phase").

> [!NOTE]
> InFact got superseded by its multimodal successor [DEFAME](https://github.com/multimodal-ai-lab/DEFAME). If you want to run the latest and most capable fact-checking system, pick DEFAME. Otherwise, if your focus is to exactly reproduce the results from the InFact paper, we recommend the InFact code. (Yes, DEFAME has the capabilities to "imitate" InFact, but there are minor differences.)

<img src="resources%2FConcept%20Figure.png" style="width: 100%; max-width: 25em !important;">

InFact is a text-only, 6-stage LLM pipeline for fact-checking with external evidence retrieval through Google Search. For evaluation on benchmarks that provide a knowledge store (like on AVeriTeC or FEVER), it supports the retrieval of relevant evidence by leveraging an embedding model to perform semantic search.

> [!NOTE]
> The execution of the embedding model and the Llama LLMs requires GPUs.


## Installation
### Option A: Docker Image
If you have [Docker](https://www.docker.com/) installed, simply run
```bash
docker run -it tudamailab/infact bash
```
You can also find the Docker image [here](https://hub.docker.com/r/tudamailab/infact).

### Option B: Manual Setup
  1. **Clone the InFact branch**
      ```bash
      git -b infact https://github.com/multimodal-ai-lab/DEFAME
      mv DEFAME InFact  # Rename the directory name coming from the repository
      cd InFact
      ```

  2. **Optional: Set up a virtual environment and activate it**
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```

  3. **Install required packages**
      ```bash
      pip install -r requirements.txt
      ```

## Configuration & Building
### 1. Configure API Keys
To this end, run the script `scripts/setup.py` with
```bash
python -m scripts.setup
```
> [!NOTE]
> Whenever running a script, ensure the project root to be the working directory. You can accomplish that by using the `-m` parameter as in the command above (note the script path notation):

Alternatively, you may manually insert all necessary API keys into the file `config/api_keys.yaml`.

### 2. Optional: Configure Datasets & Results Paths
In `config/globals.py`, insert the paths to the respective input/output directories.

### 3. Optional: Build AVeriTeC Knowledge Base (KB)
Simply run the script `scripts/averitec/build.py`with
```bash
python -m scripts.averitec.build
```
This will download, extract, and embed the entire AVeriTeC KB for `train`, `dev`, and `test`.


## Usage
All execution scripts are located in (subfolders of) `scripts/`. Make sure to run the scripts from the project root's folder as the working directory.

**Hardware requirements**: You'll need GPUs if you want to use the AVeriTeC Knowledge Base (required for reproduction) or if you want to employ your local LLM.

### Run Your Own Fact-Check
With `scripts/run.py`, you can fact-check your own claims. The command to run it from the project root is
```bash
python -m scripts.run
```
If you provided a Serper API key, this will use Google Search to verify your claim.

### Reproduce InFact Paper Results
To reproduce InFact's result on AVeriTeC using its default hyperparameters, run
```bash
python -m scripts.averitec.evaluate
```
The number of workers executing the evaluation is based on the number of GPUs in your machine.
