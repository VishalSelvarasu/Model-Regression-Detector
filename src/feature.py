import json
import os
from functools import lru_cache
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # loads .env locally; CI uses real env vars

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"

CATEGORIES = ("sensor_fault", "communication_error",
              "mechanical_fault", "nominal")


class FaultClassification(BaseModel):
    category: Literal["sensor_fault", "communication_error",
                      "mechanical_fault", "nominal"]
    summary: str


def _client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a "
            "free key from https://console.groq.com/keys"
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=5)


@lru_cache(maxsize=None)
def load_prompt(version: str = "v1") -> dict:
    path = f"prompts/{version}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_messages(prompt: dict, report_text: str) -> list:
    """Build the message list: system prompt, few-shot examples, then the report."""
    messages = [{"role": "system", "content": prompt["system_prompt"]}]
    for example in prompt.get("few_shot_examples") or []:
        messages.append({"role": "user", "content": example["input"]})
        messages.append(
            {"role": "assistant", "content": json.dumps(example["output"])})
    messages.append({"role": "user", "content": report_text})
    return messages


def _extract_json(text: str) -> dict:
    """Extract and parse the first {...} block from the model's reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start: end + 1])


def classify_fault_report(
    report_text: str,
    version: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> FaultClassification:
    """Classify a report. Version defaults to the PROMPT_VERSION env var."""
    version = version or os.environ.get("PROMPT_VERSION", "v1")
    prompt = load_prompt(version)
    messages = _build_messages(prompt, report_text)

    response = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    data = _extract_json(response.choices[0].message.content)
    return FaultClassification(**data)
