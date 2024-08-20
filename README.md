# Pairing Security Advisories with Vulnerable Functions using Open-Source LLMs

This repository contains the code and data of our research on pairing security advisories with vulnerable functions using open-source Large Language Models (LLMs). 

## Requirements

- **GPU Support**: The research codebase is currently for GPUs, specifically tested on two NVIDIA RTX 3090 TIs. Future updates will include support for non-GPU environments.
- **Storage**: Model cloning requires over 100 GB of storage space.

## Setup Instructions

### Environment Configuration

1. **Workspace Setup**: Update the `.env` file with your workspace directory where you prefer to save files. Also allows for parsing GitHub data and loading data to WANDB if you prefer:

    ```plaintext
    WORKSPACE_FOLDER={YOUR_WORKSPACE}
    PYTHONPATH=${WORKSPACE_FOLDER}/
    GITHUB_TOKEN={YOUR_TOKEN}
    GITHUB_USERNAME={YOUR_USERNMAE}
    WANDB_KEY={YOUR_WANDB_KEY}
    ```

2. **YAML Configuration**: Modify paths in `./code/llm/cfgs/sample_config.yaml` to point to your directories. This configuration file is essential for driving the LLM.

    ```yaml
    # YAML Configuration Parameters
    paths:
      base: {UPDATE THESE PATHS}
    ```

### Model Cloning

Clone the following models into a designated model directory, as specified in your YAML configuration. Note the significant storage requirements.

```shell
git clone git@hf.co:codellama/CodeLlama-7b-Instruct-hf &&
git clone git@hf.co:codellama/CodeLlama-13b-Instruct-hf &&
git clone git@hf.co:codellama/CodeLlama-34b-Instruct-hf &&
git clone git@hf.co:deepseek-ai/deepseek-coder-33b-instruct &&
git clone git@hf.co:mistralai/Mixtral-8x7B-Instruct-v0.1 &&
git clone git@hf.co:WizardLM/WizardCoder-15B-V1.0
```

### Configuration Updates

Ensure the model paths are correctly set in the configuration:

```yaml
models:
  base: {UPDATE THESE PATHS}
```
### Data

Extract the CSVs in ```/data/patchparser-data.tar.gz```:

```shell
mkdir ./data/patchparser-data
tar -xzf ./data/patchparser-data.tar.gz -C ./data/patchparser-data/
```
This will create:
```shell
$ ls ./data/patchparser-data
govulndb-cot-examples-fp-2023-10-31.csv  
govulndb-cot-examples-tp-2023-10-31.csv  
patchparser-data-2023-10-31.csv
```

### Environment Setup

Create and activate a Python virtual environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

Install required Python packages:

```shell
pip3 install -r requirements.txt
```

### Execution

To execute the CodeLlama 34b model in a few-shot setting and observe results:

```shell
python3 ./code/llm/llm_driver.py sample_config
```

## Target Models

Our research uses the following models:

- [CodeLlama](https://huggingface.co/codellama)
- [DeepSeek](https://huggingface.co/deepseek-ai)
- [WizardCoder](https://huggingface.co/WizardLM)
- [Mixtral](https://huggingface.co/mistralai)

# Contact
For questions please feel free to open an issue: [GitHub Issues](https://github.com/s3c2/llm-vulnerable-functions/issues)

# Cite

```
@InProceedings{DunlapLLM2024,
  title = {Pairing Security Advisories with Vulnerable Functions Using Open-Source LLMs},
  ISBN = {9783031641718},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-64171-8_18},
  DOI = {10.1007/978-3-031-64171-8_18},
  booktitle = {Lecture Notes in Computer Science},
  publisher = {Springer Nature Switzerland},
  author = {Dunlap,  Trevor and Meyers,  John Speed and Reaves,  Bradley and Enck,  William},
  year = {2024},
  pages = {350–369}
}
```
