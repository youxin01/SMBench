# SMBench: Dataset and Framework for Evaluating Sub-Modeling via Adaptive Function Selection
This repository provides the official implementation of the paper: **SMBench: Dataset and Framework for Evaluating Sub-Modeling via Adaptive Function Selection**

<p align="center">   <img src="./assets/pipeline.png" alt="AMML Pipeline" width="700"> </p>

## 🚀 Quick Start

### 📦 Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

------

### 🔐 API Configuration

To use ASMF with different LLM providers, configure the API in `utils/api.py`.
 For example:

```python
"doubao": {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "YOUR_API_KEY",
    "model": "doubao-1-5-pro-32k-250115"
}
```

> Pre-configured providers: `deepseek`, `qwen32`, `qwen72`, `kimi`, and `doubao`.

------

### 📁 Data Format

ASMF supports four task types, each with specific input requirements:

#### 🔮 Prediction

Regression, classification, or time-series prediction tasks.

- `question.txt`: Task description.
- `train.csv`: Training data.
- `test.csv`: Test data (no target column). *Optional for time series.*

#### 📊 Evaluation

Rank subjects based on features.

- `question.txt`: Task description.
- `data.csv`: Optional related data.

#### 🧠 Optimization

Mathematical (e.g., Linear Programming) or graph-based (e.g., vertex coloring) problems.

- `question.txt`: Task description.
- `data.csv`: Optional related data.

#### ⚙️ Basic

Statistical analysis tasks like hypothesis testing or distribution testing.

- `question.txt`: Task description.
- `data.csv`: Optional related data.

------

### ▶️ Run

#### 🔧 Standalone Mode (No MCP)

You can directly run the pipeline via command line:

```bash
python ./pipeline.py --question ./test_case/o7/question.txt --type opt --agent deepseek --max_retries 3 --cover
```

**Arguments:**

- `--question`: Path to the question file
- `--type`: Task type:
  - `opt` = Optimization
  - `pre` = Prediction
  - `eval` = Evaluation
  - `basic` = Basic Statistical Task
- `--agent`: LLM provider (e.g., `deepseek`, `qwen32`, `kimi`, `doubao`)
- `--max_retries`: Maximum number of retries for code correction
- `--cover`: Force overwrite intermediate results (useful for re-running)

------

#### 🧠 Interactive Mode with MCP

To make the framework more user-friendly, we provide an **MCP Agent** for conversational use.

1. Configure the LLM provider in `mcp_main.py`
2. Launch the interactive agent:

```bash
python main_mcp.py
```

Example interaction:

```txt
[USER PROMPT]
我有一个建模问题，问题路径在 ./MMBench/optimization/o3/question.txt，请你帮我解决一下，我要知道答案。

[FUNCTION CALL]
Name: read_file  
Arguments: {"file_path": "./MMBench/optimization/o3/question.txt"}

[FUNCTION CALL]
Name: solve_modeling_problem  
Arguments: {"question_path": "./MMBench/optimization/o3/question.txt", "type": "opt"}

[AI RESPONSE]
问题的最优生产计划和最大日利润如下：

- **最优生产计划**：
  - 生产A产品使用20桶牛奶。
  - 生产B产品使用30桶牛奶。

- **最大日利润**：3360元。
```

