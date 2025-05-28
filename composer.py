# composer.py

import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from typing import Dict, List, Tuple, Optional

from config import ComposerSettings

class AudioComposer:
    """
    Combines multiple generated audio tracks and merges with the silent video.

    Args:
        settings (ComposerSettings): Sample rate, default filenames, etc.
    """
    def __init__(self, settings: ComposerSettings):
        self.settings = settings

    def compose_final_audio(
        self,
        audio_files: Dict[str, List[str]],
        timings: Dict[str, List[Tuple[float, float]]],
        video_duration: float,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Mixes per-object WAV files onto a single timeline.

        Args:
            audio_files (Dict[str, List[str]]): tag→list of filepaths.
            timings (Dict[str, List[Tuple[float,float]]]): tag→[(start,end),…].
            video_duration (float): Total video length in seconds.
            output_filename (str, optional): Where to write final .wav.

        Returns:
            str: Path to the combined WAV file.
        """
        out = output_filename or self.settings.default_audio_filename
        sr = self.settings.sample_rate
        total = int(video_duration * sr)
        track = np.zeros(total, dtype=np.float32)

        for tag, files in audio_files.items():
            segments = timings.get(tag, [])
            for f, (start, end) in zip(files, segments):
                audio, _ = sf.read(f)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                sidx = int(start*sr)
                eidx = min(sidx + len(audio), total)
                track[sidx:eidx] += audio[: eidx - sidx]

        track = np.clip(track, -1.0, 1.0)
        sf.write(out, track, sr)
        return out

    def merge_audio_with_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Attaches an audio file to a silent video.

        Args:
            video_path (str): Path to the original silent video.
            audio_path (str): Path to the mixed audio .wav.
            output_path (str, optional): Destination MP4 file.

        Returns:
            str: Path to the merged video.
        """
        out = output_path or self.settings.default_video_filename
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        video.set_audio(audio).write_videofile(out, codec="libx264", audio_codec="aac")
        return out
