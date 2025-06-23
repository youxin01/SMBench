from openai import OpenAI
from typing import Union

def gpt_chat(
    provider: str,
    messages: Union[None, list[dict]] = None,
    sys: str = None,
    user: str = None) -> str:
    config = {
            "doubao": {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_key": "df92ca3b-eda6-4d27-9eb7-d81e715aa6e1",
                "model": "doubao-1-5-pro-32k-250115"
            },
            "kimi": {
                "base_url": "https://api.moonshot.cn/v1",
                "api_key": "sk-zhMCEZyyOFcNm3OAKmWHDCnJVIABCbHIEemu8pTTjWDg7n1o",
                "model": "kimi-latest"
            },
            "qwen72": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "sk-3b32fb0739b2411bb9c098e6f9b56a3e",
                "model": "qwen2.5-72b-instruct"
            },
            "qwen32": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "sk-3b32fb0739b2411bb9c098e6f9b56a3e",
                "model": "qwen2.5-32b-instruct"
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com/",
                "api_key": "sk-3f4a482416744badbd6ee1ff7d268026",
                "model": "deepseek-chat"
            }
        }
   
    if provider not in config:
        raise ValueError(f"Unknown provider: {provider}")

    cfg = config[provider]
    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"]
    )

    if messages is None:
        if sys is None or user is None:
            raise ValueError("If messages is not provided, both sys and user must be.")
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]

    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages
    )
    return completion.choices[0].message.content

