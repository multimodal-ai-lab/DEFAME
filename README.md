# Multimodal Automated Fact-Checking (MAFC)

This new repository is the beginning of a long-term project dedicated to automatically fact-check textual and visual content, especially from social media.

The codebase was forked from the unimodal [Search-Augmented Factuality Evaluator (SAFE)](https://arxiv.org/abs/2403.18802) by DeepMind.

## Installation

1. Clone our GitHub repository with
```bash
git clone https://github.com/maggir/mafc.git
```
2. Install the `spacy` package and download the spacy pipeline components for English via 
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

TBC

## Usage
TBC

### Image Loading
For multimodal usage turn image into a tensor by either:
1) pulling it from link:
   image_url = "https://llava-vl.github.io/static/images/view.jpg"
   image = Image.open(requests.get(image_url, stream=True).raw)
  or
2) pulling it from path
   image_path = path_to_data + "MAFC_test/image_claims/00000.png"
   image = Image.open(image_path)
Hand the tensor as second argument to Factchecker.check
