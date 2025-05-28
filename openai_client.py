import json
import openai
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import OpenAISettings

class OpenAIClient:
    """
    Wraps OpenAI ChatCompletion for tag filtering and prompt generation.

    Args:
        settings (OpenAISettings): API key, model name, retries, timeout.
    """
    def __init__(self, settings: OpenAISettings):
        openai.api_key = settings.api_key
        self.settings = settings

    @retry(
        retry=retry_if_exception_type(openai.error.OpenAIError),
        stop=stop_after_attempt(lambda self: self.settings.retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def get_sound_relevant_tags(self, tags: List[str]) -> List[str]:
        """
        Filters a list of tags to only those that produce or contribute to real sounds.

        Args:
            tags (List[str]): Raw labels from Gemini.

        Returns:
            List[str]: Tags capable of making sound (alone or via interaction).
        """
        messages = [
            {"role": "system", "content": (
                "Filter to tags that produce or contribute to real-world sounds. "
                "Include objects and interaction-based tags."
            )},
            {"role": "user", "content": f"Tags: {tags}. Return comma-separated list only."}
        ]
        resp = openai.ChatCompletion.create(
            model=self.settings.model,
            messages=messages,
            max_tokens=80,
            timeout=self.settings.timeout
        )
        text = resp.choices[0].message.content.strip()
        return [t.strip() for t in text.split(",") if t.strip()]

    @retry(
        retry=retry_if_exception_type(openai.error.OpenAIError),
        stop=stop_after_attempt(lambda self: self.settings.retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def generate_audio_prompts_from_objects(
        self, objects: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generates short (<=10 words) prompts for each sound‐relevant object.

        Args:
            objects (List[Dict[str, Any]]): Each dict must include 'label', optional 'interacts_with', etc.

        Returns:
            Dict[str, str]: Mapping from object key to prompt string.
        """
        prompts: Dict[str, str] = {}
        for obj in objects:
            if not obj.get("sound_relevant", False):
                continue
            key = obj["label"]
            if obj.get("interacts_with"):
                key += f" interacting with {obj['interacts_with']}"
            messages = [
                {"role": "system", "content": (
                    "Generate a short audio prompt (<=10 words). Return only the prompt text."
                )},
                {"role": "user", "content": json.dumps(obj)}
            ]
            resp = openai.ChatCompletion.create(
                model=self.settings.model,
                messages=messages,
                max_tokens=20,
                timeout=self.settings.timeout
            )
            prompts[key] = resp.choices[0].message.content.strip()
        return prompts
