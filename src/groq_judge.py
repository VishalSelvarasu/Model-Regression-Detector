import os

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a "
                "free key from https://console.groq.com/keys"
            )
        self.model_name = model_name
        self.api_key = api_key
        super().__init__(model_name)  # sets self.name, then calls load_model() -> self.model

    def load_model(self):
        return OpenAI(api_key=self.api_key, base_url=GROQ_BASE_URL)

    def generate(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # Groq calls here are made synchronously; DeepEval's async path
        # just calls straight through. Fine at this test volume (~15-20
        # golden cases) — revisit if the dataset grows into the hundreds.
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"groq/{self.model_name}"
