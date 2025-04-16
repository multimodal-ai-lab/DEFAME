# DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

[![Paper](https://img.shields.io/badge/Paper-EC6500?style=for-the-badge&logo=bookstack&logoColor=white)](https://arxiv.org/abs/2412.10510)&nbsp;&nbsp;&nbsp;[![License](https://img.shields.io/badge/License-Apache--2.0-F5A300?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

![Teaser.png](resources%2FTeaser.png)

This is the implementation of **Dynamic Evidence-based FAct-checking with Multimodal Experts (DEFAME)**, a strong multimodal claim verification system. DEFAME decomposes the fact-checking task into a dynamic 6-stage pipeline, leveraging an MLLM to accomplish sub-tasks like planning, reasoning, and evidence summarization.

DEFAME is the successor of our challenge-winning unimodal fact-checking system, [InFact](https://aclanthology.org/2024.fever-1.12/). This repository is under constant development. You can access the original code of InFact in [this release](https://github.com/multimodal-ai-lab/DEFAME/tree/infact), the original code of DEFAME – as used in our paper – in [that release](https://github.com/multimodal-ai-lab/DEFAME/tree/v2.0.0).


## Table of Contents
- [Installation](#installation)
- [Prepare Benchmarks](#prepare-benchmarks)
- [Usage](#usage)
- [Tools and Integrations](#tools-and-integrations)
- [API](#api)
- [Web Interface](#web-interface)
- [License](#license)


## Installation
You can install DEFAME either via Docker or manually. In any case, you first need to clone the repository:
 ```bash
 git clone https://github.com/multimodal-ai-lab/DEFAME
 cd DEFAME
 ```

### Option A: Docker (Easiest, Fastest)
Choose this option if you're interested in _executing_ rather than modifying DEFAME.

If you have [Docker](https://www.docker.com/) installed, from the project root simply run
```bash
docker compose up -d
docker compose exec defame bash
```
This will download and execute the [latest images](https://hub.docker.com/r/tudamailab/defame) we have built for you. It opens a shell. You can continue with [Usage](#usage) from here.

### Option B: Manual Installation (Most Flexible)
Choose this option if you want to _modify_ rather than just execute DEFAME.

Follow these steps:

1. Optional: Set up a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install all required packages:
    ```bash
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('wordnet')"
    ```

    If you have a CUDA-enabled GPU and the CUDA toolkit installed, also run:
    ```bash
    pip install -r requirements_gpu.txt
    ```


## Prepare Benchmarks
If you want to evaluate DEFAME on a benchmark, you need to do the following:

1. Download the needed benchmarks. We use:
   1. [AVeriTeC](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data)
   2. [VERITE](https://github.com/stevejpapad/image-text-verification)
   3. [MOCHEG](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform)
   4. ClaimReview2024+ (Link TBD)
   
2. Order the benchmarks in the following directory structure:
   ```plaintext
   your_dataset_folder/
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

3. Include the path to `your_dataset_folder` in the `data_base_dir` variable inside `config/globals.py`. DEFAME will automatically locate and process the datasets within `MOCHEG`, `VERITE`, and `AVeriTeC` when needed.


## Usage
All execution scripts are located in (subfolders of) `scripts/`.
> [!NOTE]
> Whenever running a script, ensure the project root to be the working directory. You can accomplish that by using the `-m` parameter as in the commands below (note the script path notation):

**Hardware requirements**: CPU-only is sufficient if you refrain from using a local LLM and disable the geolocator.

**Output location**: All generated reports, statistics, logs etc. will be saved in `out/` by default. You may change this in the `config/globals.py` file.

### Run Your Own Fact-Check
With `scripts/run.py`, you can fact-check any image-text claim. The script already contains an example. Execute it with
```bash
python -m scripts.run
```
It will run DEFAME with the default configuration (using GPT-4o). When running this command the first time, you'll be prompted to enter API keys. Just enter the ones you need. (See [Tools and Integrations](#tools-and-integrations) for which keys DEFAME requires.)

### Run a Benchmark Evaluation
Benchmark evaluations can be run in two different ways. We recommend to use YAML configuration files. See `config/verite` for a bunch of examples. After you configured your own config, just copy the config's file path into `run_config.py` and run
```bash
python -m scripts.run_config
```


## Tools and Integrations
DEFAME uses a Large Language Model (LLM) as the backbone which natively cannot retrieve evidence (reliably). Therefore, DEFAME leverages **tools** for evidence retrieval. A `Tool` can be anything that implements an I/O operation, executed via an `Action`. The implementations of each `Tool` can be found in `defame/evidence_retrieval/tools`.

Often, a `Tool` requires access to an external API, e.g., the `Searcher` tool needs SerperAPI to run web searches. All external API **integrations** (wrappers, HTTP request handling etc.) are implemented inside `defame/evidence_retrieval/integrations`. All API keys go into `config/api_keys.yaml`.

Some tools require additional setup. See the tool-specific setup guidelines below for any extra setup needed.

### Integrated APIs
Here's an overview of all APIs that are integrated into DEFAME.

| API             | Free | Required for...                                   |
|-----------------|------|---------------------------------------------------|
| OpenAI          | ❌    | DEFAME with GPT (default), otherwise optional     |
| 🤗 Hugging Face | ✔️   | DEFAME with Llava, otherwise optional             |
| Serper          | ❌    | DEFAME with Google Web Search, otherwise optional |
| DuckDuckGo      | ✔️   | nothing. Reaches rate limit quickly.              |
| Google Vision   | ❌    | DEFAME with Reverse Image Search                  |
| Firecrawl       | ✔️   | DEFAME with Reverse Image Search                  |

### OpenAI API
You will need the [OpenAI API](https://openai.com/index/openai-api/) if you want to use any of OpenAI's GPT models.

### 🤗 Hugging Face
Required for the usage of open source LLMs on [🤗 Hugging Face](https://huggingface.co/), like Llama and Llava.

### Serper API
The [Serper API](https://serper.dev/?utm_term=serpapi&gad_source=1&gclid=Cj0KCQiAo5u6BhDJARIsAAVoDWspQtWo419c8unYBBlVWOKTGUyJhoh2ZBQuWdRR2hDZf8gpIS3h7UEaAksZEALw_wcB) serves standard Google Search as well as Google Image Search. If you don't want to use DuckDuckGo (which has restrictive rate limits, unfortunately), you will need this API.

### Google Vision API
The [Google Cloud Vision API](https://cloud.google.com/vision/?hl=en&utm_source=google&utm_medium=cpc&utm_campaign=emea-de-all-en-dr-bkws-all-all-trial-e-gcp-1707574&utm_content=text-ad-none-any-DEV_c-CRE_574683096392-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20-%20AI%20And%20Machine%20Learning%20-%20Vision%20AI%20-%20v5-KWID_43700076827179891-kwd-203288731207-userloc_9044485&utm_term=KW_google%20cloud%20vision%20api-NET_g-PLAC_&&gad_source=1&gclid=Cj0KCQiAo5u6BhDJARIsAAVoDWsrq9lbMXzJHzooohJcQNyp-HVgzeeF__yyrpieYi-gEFpinOKnAeEaArmlEALw_wcB&gclsrc=aw.ds) is required to perform Reverse Image Search. Follow these steps, to set it up:
1. In the [Google Cloud Console](https://console.cloud.google.com), create a new Project.
2. Go to the [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts) overview and add a new Service Account.
3. Open the new Service Account, go to "Keys" and generate a new JSON key file.
4. Save the downloaded key file at the path `config/google_service_account_key.json`.


### Add a Custom Tool

To extend the fact-checker with an own `Tool`, follow these steps:

0. **Only if needed: Implement the integration**
   
   If your tool requires calling external resources, integrate the external resource by implementing a respective module inside `defame/evidence_retrieval/integrations`.

1. **Implement the Tool and Action(s)**

   Inside `defame/evidence_retrieval/tools`, create a new file and implement the tool, inheriting from the abstract `Tool` class. Make sure to implement all required methods and add a respective `Action`. Caution: Action docstrings are used *1:1* as a description of the tool to inform the LLM planning module about the action's purpose and how the action is used. See other tools and actions for examples.

2. **Register Your Tool in `defame/evidence_retrieval/tools/__init__.py`**
   
   Incorporate the new tool into the `TOOL_REGISTRY` list and its actions into the `ACTION_REGISTRY` set.

3. **Add the tool configuration to the hyperparameters**
   
   Don't forget to specify the tool in the DEFAME hyperparameters or, alternatively, in your configuration (the YAML file that goes into `config/`).

4. **Optional: Register the Tool in Benchmarks**
   
   This step is required only if you want to use your tool for evaluation on one of the benchmarks. To this end, navigate to the respective benchmark file under `defame/eval/<benchmark_name>/benchmark.py`. There, in the `available_actions` list, add your `Tool`.

### Web Scraping
This project uses **Firecrawl** as the default web scraping service. If Firecrawl is not available, DEFAME falls back to a simple **BeautifulSoup** implementation. Firecrawl runs automatically if you used `docker compose` to install DEFAME.

#### Manual Firecrawl Setup
If you installed DEFAME manually, you also need to run Firecrawl manually by executing
```bash
docker run -d -p 3002:3002 tudamailab/firecrawl
```
DEFAME will automatically locate Firecrawl. If you use custom, non-standard ports or URLs, you must also update the `firecrawl_url` inside  `config/globals.py`.

If you really want to set up Firecrawl manually and without Docker, follow the [Firecrawl Documentation](https://github.com/mendableai/firecrawl.git). We do not recommend that because, in our experience, that setup procedure is rather involving. Therefore, we recommend to use the Firecrawl Docker Image we provide for this project. (You may modify and re-build the Firecrawl Image via the `Dockerfile` stored in `third_party/firecrawl`.)


## API
DEFAME ships with its own API backend, allowing the user to access DEFAME's functionalities through HTTP requests. Simply execute the script `scripts/run_api.py` to activate the server which will be accessible at http://0.0.0.0:3003. The API documentation is then available under http://0.0.0.0:3003/docs.

Don't forget to [run Firecrawl](#Manual-Firecrawl-Setup) if you want to use it.


## Web Interface
DEFAME provides a web-based graphical user interface, supporting the input and verification of textual _and_ visual content. As of now, our hosted web interface is **not public**, but you can host it yourself using [this repository](https://github.com/multimodal-ai-lab/DEFAME-Web-UI). (You also need to run the [DEFAME API](#api).)
![UI.png](resources/UI.png)



## [License](LICENSE)
This repository and all its contents (except for the contents inside `third_party/`) are licensed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

Please use the following BibTeX to refer to the authors:
```bibtex
@article{braun2024defamedynamicevidencebasedfactchecking,
   title={DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts}, 
   author={Tobias Braun and Mark Rothermel and Marcus Rohrbach and Anna Rohrbach},
   year={2024},
   eprint={2412.10510},
   archivePrefix={arXiv},
   primaryClass={cs.CV},
   url={https://arxiv.org/abs/2412.10510}, 
   journal={arXiv preprint arXiv:2412.10510},
}
```
