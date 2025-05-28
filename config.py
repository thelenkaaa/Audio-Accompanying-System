from pydantic import BaseSettings, Field
from typing import Dict, Any

class GeminiSettings(BaseSettings):
    """
    Configuration for interacting with the Gemini API.

    Attributes:
        api_key (str): Gemini API key from env var GEMINI_API_KEY.
        base_url (str): Base URL for the Gemini service (env GEMINI_BASE_URL).
        model (str): Gemini model identifier (env GEMINI_MODEL).
        timeout (int): Seconds to wait for file activation (env GEMINI_TIMEOUT).
        retries (int): Number of retries on transient failures (env GEMINI_RETRIES).
        instructions (str): Natural‐language instructions for Gemini (env GEMINI_INSTRUCTIONS).
        response_schema (Dict[str, Any]): JSONschema to validate responses.
    """
    api_key: str = Field(..., env="GEMINI_API_KEY")
    base_url: str = Field("https://api.gemini.example.com", env="GEMINI_BASE_URL")
    model: str = Field("gemini-2.0-flash", env="GEMINI_MODEL")
    timeout: int = Field(60, env="GEMINI_TIMEOUT")
    retries: int = Field(3, env="GEMINI_RETRIES")
    instructions: str = Field(
        "Analyze the provided silent video and return a JSON object with detected objects and a summary.",
        env="GEMINI_INSTRUCTIONS"
    )
    response_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label":      {"type": "string"},
                            "start_time": {"type": "number"},
                            "end_time":   {"type": "number"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["label", "start_time", "end_time"]
                    }
                },
                "summary": {"type": "string"}
            },
            "required": ["objects", "summary"]
        }
    )

class OpenAISettings(BaseSettings):
    """
    Configuration for OpenAI API calls.

    Attributes:
        api_key (str): OpenAI API key from env OPENAI_API_KEY.
        model (str): ChatCompletion model (env OPENAI_MODEL).
        retries (int): Retry attempts on OpenAI errors (env OPENAI_RETRIES).
        timeout (int): Request timeout in seconds (env OPENAI_TIMEOUT).
    """
    api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    retries: int = Field(3, env="OPENAI_RETRIES")
    timeout: int = Field(60, env="OPENAI_TIMEOUT")

class StableAudioSettings(BaseSettings):
    """
    Configuration for StableAudio pipeline.

    Attributes:
        model_id (str): HuggingFace ID or path for the audio model (env AUDIO_MODEL_ID).
        torch_dtype (str): torch dtype name, e.g. 'float16' (env AUDIO_TORCH_DTYPE).
        device (str): Device string, e.g. 'cuda' (env AUDIO_DEVICE).
        samples_num (int): Number of samples per prompt (env AUDIO_SAMPLES_NUM).
        negative_prompt (str): Negative prompt text (env AUDIO_NEGATIVE_PROMPT).
        num_inference_steps (int): Diffusion steps (env AUDIO_INFERENCE_STEPS).
        seed (int): RNG seed (env AUDIO_SEED).
    """
    model_id: str = Field("stabilityai/stable-audio-open-1.0", env="AUDIO_MODEL_ID")
    torch_dtype: str = Field("float16", env="AUDIO_TORCH_DTYPE")
    device: str = Field("cuda", env="AUDIO_DEVICE")
    samples_num: int = Field(1, env="AUDIO_SAMPLES_NUM")
    negative_prompt: str = Field("Bad quality sound, not recognizable.", env="AUDIO_NEGATIVE_PROMPT")
    num_inference_steps: int = Field(40, env="AUDIO_INFERENCE_STEPS")
    seed: int = Field(0, env="AUDIO_SEED")

class ComposerSettings(BaseSettings):
    """
    Configuration for audio composition and video merging.

    Attributes:
        sample_rate (int): Sample rate for final audio (env COMPOSER_SAMPLE_RATE).
        default_audio_filename (str): Default output WAV filename (env COMPOSER_AUDIO_FILENAME).
        default_video_filename (str): Default output video filename (env COMPOSER_VIDEO_FILENAME).
    """
    sample_rate: int = Field(44100, env="COMPOSER_SAMPLE_RATE")
    default_audio_filename: str = Field("final_output.wav", env="COMPOSER_AUDIO_FILENAME")
    default_video_filename: str = Field("final_video_with_audio.mp4", env="COMPOSER_VIDEO_FILENAME")
