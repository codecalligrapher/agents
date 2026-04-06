from typing import List
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Model name (change if you want)
MODEL = "qwen3.5:latest"


@dataclass
class AgentMessage:
    role: str
    content: str


@dataclass
class BaseAgent:
    name: str
    system_prompt: str

    def call_openai(
        self, messages, model=MODEL, temperature=0.45, max_tokens=2000, json_mode=False
    ):
        payload = [{"role": m.role, "content": m.content} for m in messages]
        payload.insert(0, {"role": "system", "content": self.system_prompt})
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=payload,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content.strip()
