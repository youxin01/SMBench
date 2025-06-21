from openai import OpenAI

def gpt_chat(sys: str, user: str, provider: str) -> str:
    config = {
        "doubao": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "",
            "model": "doubao-1-5-pro-32k-250115"
        },
        "kimi": {
            "base_url": "https://api.moonshot.cn/v1",
            "api_key": "",
            "model": "moonshot-v1-32k"
        },
        "qwen72": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "qwen2.5-72b-instruct"
        },
        "qwen32": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "qwen2.5-32b-instruct"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/",
            "api_key": "",
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

    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content

