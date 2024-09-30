# Multimodal Automated Fact-Checking (MAFC)

MAFC is a long-term project dedicated to automatically fact-checking both textual and visual content, with a particular focus on social media. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Image Loading](#image-loading)
- [Adding a New Tool](#adding-a-new-tool)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the MAFC environment:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/.../InFact.git
    cd InFact
    ```

2. **Set Up a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install SpaCy and Download the English Model**
    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm
    ```

5. **Additional Dependencies**
   
   Ensure that you have the following packages installed for image processing and other functionalities:
    ```bash
    pip install torch torchvision transformers faiss-cpu pillow requests
    ```

6. **Configure API Keys**
   
   Update the `config/globals.py` file with your necessary API keys, such as the Serper API key:
    ```python
    api_keys = {
        "serper_api_key": "YOUR_SERPER_API_KEY",
        # Add other API keys here
    }
    ```

## Enable Tool APIs
Many tools require external APIs to be functional. In most cases, you just need to insert the respective API key into `config/api_keys.yaml`. For all other tools, see the following setup guidelines.

### Google Cloud Vision API (needed for Image Reverse Search)
1. In the [Google Cloud Console](https://console.cloud.google.com), create a new Project.
2. Go to the [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts) overview and add a new Service Account.
3. Open the new Service Account, go to "Keys" and generate a new JSON key file.
4. Save the downloaded key file at the path `config/google_service_account_key.json`.

## Usage

### Image Loading

To utilize multimodal capabilities, convert images into tensors by following one of the methods below:

1. **Loading from a URL**
    ```python
    import requests
    from PIL import Image

    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    ```

2. **Loading from a Local Path**
    ```python
    from PIL import Image

    image_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/Fakeddit/public_images.tar.bz2"
    image = Image.open(image_path)
    ```

Once loaded, pass the image tensor as the second argument to the `Factchecker.check` method:
```python
factchecker.check("Your text claim here", image)
```

## Adding a New Tool

To extend the Factchecker with additional tools, follow these steps:

1. **Register the Tool in Benchmarks**
   
   Add the new tool to the `available_actions` list in the corresponding benchmark. For example:
    ```python
    available_actions = [
        WebSearch,
        DetectManipulation,
        DetectObjects,
        Geolocate,
        ImageSearch,
        NewTool,  # Your new tool
    ]
    ```

2. **Add an Exmaple Prompt in the plan_exemplars folder**
   
   Format:

    # Example: action()

    ## Claim
    Text: "The sign in <image:3> says 'No Trespassing'."

    ...

    ACTIONS:
    ```
    action(<image:3>)
    ```

3. **Configure the Tool in `evaluate.py`**
   
   Add the tool's configuration to the `tools_config` dictionary in `evaluate.py`:
    ```python
    tools_config = dict(
        searcher=dict(
            search_engine_config=dict(
                google=dict(),
                duckduckgo=dict(),
            ),
            limit_per_search=3
        ),
        manipulation_detector=dict(),
        object_detector=dict(),
        geolocator=dict(),
        new_tool=dict(  # Configuration for your new tool
            # Add required parameters here
        )
    )
    ```

4. **Update Registries in `__init__.py`**
   
   Incorporate the new tool into the registries within the `tools` folder's `__init__.py`:
    ```python
    TOOL_REGISTRY = [
        CredibilityChecker,
        FaceRecognizer,
        Geolocator,
        ObjectDetector,
        Searcher,
        TextExtractor,
        ManipulationDetector,
        NewTool,  # Your new tool
    ]

    ACTION_REGISTRY = {
        WebSearch,
        WikiDumpLookup,
        DetectObjects,
        WikiLookup,
        ReverseSearch,
        Geolocate,
        FaceRecognition,
        CredibilityCheck,
        OCR,
        DetectManipulation,
        ImageSearch,
        NewAction,  # Associated action for your new tool
    }

    IMAGE_ACTIONS = {
        ReverseSearch,
        Geolocate,
        FaceRecognition,
        OCR,
        DetectManipulation,
        DetectObjects,
        ImageSearch,
        NewImageAction,  # Associated image action if applicable
    }
    ```

## License

This project is licensed under the [MIT License](LICENSE).
