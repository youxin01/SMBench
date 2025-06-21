# AMML:AnAdaptive Framework for Mathematical Modeling with LLMs
This repository provides the offical code for the paper "AMML: An Adaptive Framework for Mathematical Modeling with LLMs".

<img src="C:\Users\NumberOne\AppData\Roaming\Typora\typora-user-images\image-20250621195720873.png" alt="image-20250621195720873" style="zoom: 50%;" />

## 🚀 Qucik Start

First, install all the required packages by running:

```bash
pip install -r requirements.txt
```

### API Config

You should fill your llm's name and api key in `utils/api.py` file. For example,

```python
"doubao": {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "YOUR_API_KEY",
    "model": "doubao-1-5-pro-32k-250115"}
```

I already provide deepseek，qwen32，qwen72，kimi，doubao's url and model name.

### Run

You can run as following:

```bash
python ./pipeline.py --question ./test_case/o7/question.txt --type opt --agent deepseek --max_retries 3 --cover
```

>- `--question`: the path of the question file.
>- `--type`:type of the question, including opt(optimization),prediction(pre),evaluate(eval),basic
>- `--agent`:base llm, including`deepseek，qwen32，qwen72，kimi，doubao` etc.
>- `--max_retries`:Max retry times for code correction.
>- `--cover`:whether force overwrite of intermediate steps.