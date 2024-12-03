# [InFact: A Strong Baseline for Automated Fact-Checking](https://aclanthology.org/2024.fever-1.12/)

> [!NOTE]
> InFact got superseded by its multimodal successor [DEFAME](https://github.com/multimodal-ai-lab/DEFAME). If you want to run the latest and most capable fact-checking system, pick DEFAME. Otherwise, if your focus is to exactly reproduce the results from the InFact paper, we recommend the InFact code. (Yes, DEFAME has the capabilities to "imitate" InFact, but there are minor differences.)

This is the implementation of **Information-Retrieving Fact-Checker (InFact)**, the system that won the [AVeriTeC fact-checking challenge](https://fever.ai/task.html), see also the [AVeriTeC Leaderboard](https://eval.ai/web/challenges/challenge-page/2285/leaderboard/5655) ("Test Phase").

<img src="resources%2FConcept%20Figure.png" style="width: 100%; max-width: 25em">

InFact is a text-only, 6-stage LLM pipeline for fact-checking with external evidence retrieval through Google Search. For evaluation on benchmarks that provide a knowledge store (like on AVeriTeC or FEVER), it supports the retrieval of relevant evidence by leveraging an embedding model to perform semantic search.


## Installation
You can either use the Docker image we provide (easiest & fastest) or install InFact manually.

### Docker Image
You can download the Docker image here.

### Manual Installation
Follow these steps:

1. **Clone the InFact Branch**
    ```bash
    git -b infact https://github.com/multimodal-ai-lab/DEFAME
    mv DEFAME InFact  # Rename the directory name coming from the repository
    cd InFact
    ```

2. **Optional: Set Up a Virtual Environment and Activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure API Keys**

   Insert all necessary API keys into the file `config/api_keys.yaml`.

5. **Optional: Setup AVeriTeC Knowledge Base (KB)**

   Simply run the script `scripts/averitec/build.py`.

    > [!NOTE]
    > Whenever running a script, ensure the project root to be the working directory. You can accomplish that by using the `-m` parameter as follows (note the script path notation):
    > ```bash
    > python -m scripts.averitec.build
    > ```

6. **Optional: Configure Datasets & Results Paths**

   In `config/globals.py`, insert the paths of the directories


## Usage
All execution scripts are located in (subfolders of) `scripts/`. Make sure to run the scripts from the project root's folder as the working directory.

**Hardware requirements**: CPU-only is sufficient if you refrain from using a local LLM and if you do not use the `Geolocator` tool.

### Run Your Own Fact-Check
With `scripts/run.py`, you can fact-check your own claim. The command to run it from the project root is
```bash
python -m scripts.run
```

### Reproduce InFact
To reproduce InFact's results on AVeriTeC, run
```bash
python -m scripts.averitec.evaluate
```
