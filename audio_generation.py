import torch
import soundfile as sf
from typing import List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from stable_audio import StableAudioPipeline

from config import StableAudioSettings

class StableAudioClient:
    """
    Uses StableAudioPipeline to synthesize audio waveforms from text prompts.

    Args:
        settings (StableAudioSettings): Model IDs, device, steps, etc.
    """
    def __init__(self, settings: StableAudioSettings):
        dtype = getattr(torch, settings.torch_dtype)
        self.pipe = StableAudioPipeline.from_pretrained(
            settings.model_id, torch_dtype=dtype
        ).to(settings.device)
        self.generator = torch.Generator(settings.device).manual_seed(settings.seed)
        self.settings = settings

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(lambda self: self.settings.samples_num),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        reraise=True
    )
    def generate_audio_files(
        self, tag: str, prompt: str, duration: float
    ) -> List[str]:
        """
        Generates one or more audio files for a single prompt.

        Args:
            tag (str): Identifier for naming files.
            prompt (str): Text describing the sound to generate.
            duration (float): Length in seconds to generate.

        Returns:
            List[str]: Filenames of generated .wav files.
        """
        filenames: List[str] = []
        for i in range(self.settings.samples_num):
            out = self.pipe(
                prompt,
                negative_prompt=self.settings.negative_prompt,
                num_inference_steps=self.settings.num_inference_steps,
                audio_end_in_s=duration,
                num_waveforms_per_prompt=1,
                generator=self.generator,
            )
            wav = out.audios[0].T.cpu().numpy()
            fname = f"{tag}_{i}_{duration}s.wav"
            sf.write(fname, wav, self.pipe.vae.sampling_rate)
            filenames.append(fname)
        return filenames

    def generate_audio_for_tags(
        self, prompts: Dict[str, str], durations: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Batch‐generates audio for all tags.

        Args:
            prompts (Dict[str, str]): Mapping tag→prompt.
            durations (Dict[str, float]): Mapping tag→duration.

        Returns:
            Dict[str, List[str]]: Mapping tag→list of generated filenames.
        """
        return {
            tag: self.generate_audio_files(tag, prompt, durations.get(tag, 3.0))
            for tag, prompt in prompts.items()
        }
