# DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

|![Teaser.jpg](resources%2FTeaser.jpg) | ![Concept.png](resources%2FConcept.png)|
|---|---|

This is the implementation of Dynamic Evidence-based FAct-checking with Multimodal Experts (DEFAME), a strong multimodal fact-checking system. DEFAME decomposes the fact-checking task into a dynamic 6-stage pipeline, leveraging an MLLM to accomplish sub-tasks like planning, reasoning, and evidence summarization.

DEFAME is the successor of our challenge-winning unimodal fact-checking system, [InFact](https://aclanthology.org/2024.fever-1.12/).

This repository is under constant development. See the respective releases to access the original InFact and DEFAME systems.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [APIs](#apis)
- [Data Path Configuration](#data-path-configuration)
- [Adding a New Tool](#add-a-custom-tool)
- [License](#license)


## Installation
For the installation of InFact and DEFAME, we provide you with a respective Docker image. Please refer to the corresponding release.

If you want to set up the current repository, you'll have to do it manually. To this end, follow these steps:

1. **Clone the Repository**
    ```bash
    git https://github.com/multimodal-ai-lab/DEFAME
    cd DEFAME
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

    If you have a CUDA-enabled GPU and the CUDA toolkit installed, also run:
    ```bash
    pip install -r requirements_gpu.txt
    ```

    Afterwards run:
    ```bash
    python -m spacy download en_core_web_sm 
    ```


4. **Configure API Keys**

   Insert all necessary API keys into the file `config/api_keys.yaml`. See [APIs](#APIs) for an overview of the available APIs.


## Usage
All execution scripts are located in (subfolders of) `scripts/`. Make sure to run the scripts from the project root's folder as the working directory.

**Hardware requirements**: CPU-only is sufficient if you refrain from using a local LLM and if you do not use the `Geolocator` tool.

### Run Your Own Fact-Check
With `scripts/run.py`, you can fact-check your own claim. The script shows how it is done with an exemplary image-text claim. The command to run it from the project root is
```bash
python -m scripts.run
```

### Run a Benchmark Evaluation
If you want to execute evaluation on a specific benchmark, execute `run_config.py` with the `config_file_path` set accordingly. See `config/verite` for a bunch of config file examples.


## APIs
LLMs and many tools require external APIs. For most of them, you just need to insert the respective API key into `config/api_keys.yaml`. A few tools need additional set up steps, see the tool-specific setup guidelines below. Here's an overview of all APIs.


## Data Path Configuration

The data path should be set to the parent directory that contains all datasets. For example, if you have datasets for **MOCHEG**, **VERITE**, and **AVeriTeC**, ensure they are located within a single folder, and set the data path to this folder.

### Directory Structure Example:

```plaintext
data/
├── MOCHEG/
│   ├── images/
│   ├── train/
│   └── ...
├── VERITE/
│   ├── images/
│   ├── VERITE.csv
│   └── ...
└── AVeriTeC/
    ├── train.json
    ├── dev.json
    └── ...
```

### How to Set the Data Path:
1. Set the path variable in `config/globals.py` to the `data` folder as the parent directory.
2. The program will automatically locate and process the datasets within `MOCHEG`, `VERITE`, and `AVeriTeC`.


### OpenAI API
You will need this API if you want to use any of OpenAI's GPT models.

### 🤗 Hugging Face
Required for the usage of open source LLMs on [🤗 Hugging Face](https://huggingface.co/).

### Serper API
The [Serper API](https://serper.dev/?utm_term=serpapi&gad_source=1&gclid=Cj0KCQiAo5u6BhDJARIsAAVoDWspQtWo419c8unYBBlVWOKTGUyJhoh2ZBQuWdRR2hDZf8gpIS3h7UEaAksZEALw_wcB) serves standard Google Search as well as Google Image Search. If you don't want to use DuckDuckGo (which is quit unstable, unfortunately), you will need this API.

### Google Cloud Vision API
The [Google Cloud Vision API](https://cloud.google.com/vision/?hl=en&utm_source=google&utm_medium=cpc&utm_campaign=emea-de-all-en-dr-bkws-all-all-trial-e-gcp-1707574&utm_content=text-ad-none-any-DEV_c-CRE_574683096392-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20-%20AI%20And%20Machine%20Learning%20-%20Vision%20AI%20-%20v5-KWID_43700076827179891-kwd-203288731207-userloc_9044485&utm_term=KW_google%20cloud%20vision%20api-NET_g-PLAC_&&gad_source=1&gclid=Cj0KCQiAo5u6BhDJARIsAAVoDWsrq9lbMXzJHzooohJcQNyp-HVgzeeF__yyrpieYi-gEFpinOKnAeEaArmlEALw_wcB&gclsrc=aw.ds) is required to perform Image Reverse Search. Follow these steps, to set it up:
1. In the [Google Cloud Console](https://console.cloud.google.com), create a new Project.
2. Go to the [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts) overview and add a new Service Account.
3. Open the new Service Account, go to "Keys" and generate a new JSON key file.
4. Save the downloaded key file at the path `config/google_service_account_key.json`.

### WebScraping

This project uses **BeautifulSoup** as a fallback for web scraping. To reproduce the results from the paper, you need to install and run the **FireCrawl** server. Instructions are available at [FireCrawl Documentation](https://github.com/mendableai/firecrawl.git).

## Add a Custom Tool

To extend the fact-checker with an additional tool, follow these steps:

1. **Implement the Tool**

   It needs to inherit from the `Tool` class and implement all abstract methods. See there (and other tools) for details.

2. **Add a Usage Example to the `defame/prompts/plan_exemplars` folder**
   
   This exemplar is for the LLM to understand the purpose and format of the tool. You may consider existing exemplars from other tools to understand what's expected here.

3. **Register Your Tool in `defame/tools/__init__.py`**
   
   Incorporate the new tool into the `TOOL_REGISTRY` list and its actions in the `ACTION_REGISTRY` set.

4. **Configure the Tool in the execution script**
   
   Don't forget to specify the tool in your configuration (either the YAML configuration file or initialization of the `FactChecker`).

5. **Optional: Register the Tool in Benchmarks**
   
   This step is required only if you want to use your tool for evaluation on one of the benchmarks. To this end, navigate to the respective benchmark file under `defame/eval/<benchmark_name>/benchmark.py`. There, in the `available_actions` list, add your Tool.


## License

This project is licensed under the [MIT License](LICENSE).
