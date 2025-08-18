import os
import base64
from groq import Groq
from prompt_library.prompt import SYSTEM_PROMPT
from utils.model_loader import ModelLoader

class ModelBuilder:
    def __init__(self, api_key: str = None):
        self.model_loader = ModelLoader()
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model_name = self.model_loader.config["llm"]["groq"]["model_name"]
        self.system_prompt = SYSTEM_PROMPT

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encodes a local image as base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_report(self, image_path: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generates a text report from a medical image.
        """
        base64_image = self.encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{self.system_prompt}\nAnalyze this image and generate a report."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )

        return completion.choices[0].message.content
