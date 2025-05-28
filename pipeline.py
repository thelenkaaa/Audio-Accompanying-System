import cv2
from typing import List, Dict, Any, Tuple

from config import (
    GeminiSettings, OpenAISettings, StableAudioSettings, ComposerSettings
)
from gemini_client import VideoAnalyzer
from openai_client import OpenAIClient
from audio_client import StableAudioClient
from composer import AudioComposer

class FullVideoAudioPipeline:
    """
    Orchestrates: video→Gemini→OpenAI→StableAudio→composition→merge.

    Args:
        gemini_settings (GeminiSettings)
        openai_settings (OpenAISettings)
        audio_settings (StableAudioSettings)
        composer_settings (ComposerSettings)
    """
    def __init__(
        self,
        gemini_settings: GeminiSettings,
        openai_settings: OpenAISettings,
        audio_settings: StableAudioSettings,
        composer_settings: ComposerSettings
    ):
        self.analyzer = VideoAnalyzer(gemini_settings)
        self.openai   = OpenAIClient(openai_settings)
        self.audio    = StableAudioClient(audio_settings)
        self.composer = AudioComposer(composer_settings)

    @staticmethod
    def _extract_durations(
        objects: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Summarizes on-screen duration per object key.

        Returns:
            Dict[str, float]: tag→(end_time-start_time)
        """
        out: Dict[str, float] = {}
        for o in objects:
            key = o["label"]
            if o.get("interacts_with"):
                key += f" interacting with {o['interacts_with']}"
            out[key] = o.get("end_time",0) - o.get("start_time",0)
        return out

    @staticmethod
    def _extract_timings(
        objects: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[float,float]]]:
        """
        Builds time‐segment lists per object key.

        Returns:
            Dict[str, List[Tuple[float,float]]]: tag→[(start,end),…]
        """
        out: Dict[str, List[Tuple[float,float]]] = {}
        for o in objects:
            key = o["label"]
            if o.get("interacts_with"):
                key += f" interacting with {o['interacts_with']}"
            seg = (o.get("start_time",0), o.get("end_time",0))
            out.setdefault(key, []).append(seg)
        return out

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """
        Reads video metadata to compute duration.

        Returns:
            float: video length in seconds.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps

    def run(self, video_path: str, output_video: str) -> str:
        """
        End‐to‐end pipeline execution.

        Args:
            video_path (str): Path to input silent video.
            output_video (str): Desired MP4 output.

        Returns:
            str: Path to the final merged video.
        """
        # 1️⃣ Gemini analysis
        res     = self.analyzer.analyze(video_path)
        objects = res["objects"]

        # 2️⃣ Filter sound‐relevant
        labels     = [o["label"] for o in objects]
        tags       = self.openai.get_sound_relevant_tags(labels)
        relevant   = [o for o in objects if o["label"] in tags]

        # 3️⃣ Prompt & durations
        prompts   = self.openai.generate_audio_prompts_from_objects(relevant)
        durations = self._extract_durations(relevant)
        timings   = self._extract_timings(relevant)

        # 4️⃣ Generate audio
        files_map = self.audio.generate_audio_for_tags(prompts, durations)

        # 5️⃣ Compose & merge
        vd     = self._get_video_duration(video_path)
        wav    = self.composer.compose_final_audio(files_map, timings, vd)
        final  = self.composer.merge_audio_with_video(video_path, wav, output_video)
        return final
