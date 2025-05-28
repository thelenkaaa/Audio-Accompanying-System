import time
import requests
import jsonschema
from typing import Any, Dict
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import GeminiSettings

class SchemaModel(BaseModel):
    """
    Pydantic model for validating Gemini JSON responses.

    Fields:
        objects (Any): List of detected objects data.
        summary (str): System‐generated summary.
    """
    objects: Any
    summary: str

class VideoAnalyzer:
    """
    Handles uploading videos to Gemini, polling for activation, requesting analysis,
    and validating the response against a JSON schema.

    Args:
        settings (GeminiSettings): Loaded config for API keys, URLs, timeouts, etc.
    """
    def __init__(self, settings: GeminiSettings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json"
        })

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(lambda self: self.settings.retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def _upload_file(self, path: str) -> str:
        """
        Uploads a video file to the Gemini /files endpoint.

        Args:
            path (str): Local filepath of the video.

        Returns:
            str: The uploaded file's ID.

        Raises:
            ValueError: If the response does not contain a file_id.
            HTTPError: On non‐200 responses.
        """
        url = f"{self.settings.base_url}/files"
        with open(path, "rb") as f:
            resp = self.session.post(url, files={"file": f})
        resp.raise_for_status()
        file_id = resp.json().get("file_id")
        if not file_id:
            raise ValueError("No file_id returned from upload endpoint.")
        return file_id

    def _wait_for_activation(self, file_id: str) -> None:
        """
        Polls the /files/{file_id}/status endpoint until the file is 'active'
        or the timeout elapses.

        Args:
            file_id (str): ID returned by _upload_file.

        Raises:
            TimeoutError: If activation does not occur within settings.timeout.
        """
        url = f"{self.settings.base_url}/files/{file_id}/status"
        deadline = time.time() + self.settings.timeout
        while time.time() < deadline:
            resp = self.session.get(url)
            resp.raise_for_status()
            state = resp.json().get("state", "").lower()
            if state == "active":
                return
            time.sleep(2)
        raise TimeoutError("Timed out waiting for video file activation.")

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(lambda self: self.settings.retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True
    )
    def _generate_content(self, file_id: str) -> Dict[str, Any]:
        """
        Calls Gemini's /models/generate with instructions and schema.

        Args:
            file_id (str): ID of an active uploaded file.

        Returns:
            Dict[str, Any]: Unvalidated JSON response.
        """
        url = f"{self.settings.base_url}/models/generate"
        payload = {
            "model": self.settings.model,
            "instructions": self.settings.instructions,
            "schema": self.settings.response_schema,
            "content": {"file_id": file_id}
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Full pipeline: upload → wait → generate → schema‐validate.

        Args:
            video_path (str): Path to the local silent video.

        Returns:
            Dict[str, Any]: Parsed, schema‐validated JSON with keys 'objects' and 'summary'.
        """
        file_id = self._upload_file(video_path)
        self._wait_for_activation(file_id)
        raw = self._generate_content(file_id)
        jsonschema.validate(raw, self.settings.response_schema)
        return SchemaModel.parse_obj(raw).dict()
