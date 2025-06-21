# AMML: An Adaptive Framework for Mathematical Modeling with LLMs

This repository provides the official implementation of the paper **"AMML: An Adaptive Framework for Mathematical Modeling with LLMs"**.

<p align="center">
  <img src="./assets/pipeline.png" alt="Pipeline" width="700">
</p>


## 🚀 Quick Start

### Install

Run the following command to install all required packages:

```bash
pip install -r requirements.txt
```

### API Configuration

Specify your LLM provider and API key in the `utils/api.py` file. For example:

```python
"doubao": {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "YOUR_API_KEY",
    "model": "doubao-1-5-pro-32k-250115"
}
```

> Pre-configured providers include: `deepseek`, `qwen32`, `qwen72`, `kimi`, and `doubao`.


### Run

To run the full AMML pipeline, use the following command:

```bash
python ./pipeline.py --question ./test_case/o7/question.txt --type opt --agent deepseek --max_retries 3 --cover
```
>- `--question`: Path to the question file.
>- `--type`: Task type. Options: `opt` (optimization), `pre` (prediction), `eval` (evaluation), `basic`.
>- `--agent`: LLM provider. Options include `deepseek`, `qwen32`, `qwen72`, `kimi`, `doubao`, etc.
>- `--max_retries`: Maximum number of code correction retries.
>- `--cover`: Whether to overwrite intermediate results (set this flag to force re-run).
