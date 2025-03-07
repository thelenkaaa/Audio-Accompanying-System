from diffusers import StableAudioPipeline
from openai import OpenAI
import torch
import soundfile as sf
import IPython.display as ipd


SAMPLES_NUM = 1  # Number of audio samples per tag
NEGATIVE_PROMPT = "Bad quality sound, not recognizable."  # Set if needed
NUM_INFERENCE_STEPS = 20  # Example steps
generator = None  # Replace with your generator instance


pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)


def generate_audio_files(tag, prompt, duration):
    """
    Generates audio files based on a given tag, prompt and duration of object on screen.
    
    This function utilizes the StableAudioPipeline model to generate audio waveforms
    from text prompts. The audio is then saved as a .wav file.
    
    :param tag: str - A tag representing the category of the audio.
    :param prompt: str - A text prompt describing the sound to generate.
    :param duration: double - Duration of object on video.
    :return: list[str] - A list of filenames of the generated audio files.
    """
    filenames = []
    audio_files = []

    for i in range(SAMPLES_NUM):
        # Generate audio using your model
        wavs = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            audio_end_in_s=duration,  # Adjusted to match object duration
            num_waveforms_per_prompt=1,  # Generate one waveform per iteration
            generator=generator,
        ).audios

        output = wavs[0].T.float().cpu().numpy()
        filename = f"{tag}_{i}_{duration}s.wav"
        
        # Save the audio file
        sf.write(filename, output, pipe.vae.sampling_rate)
        filenames.append(filename)
        audio_files.append(filename)

    return audio_files


def generate_audio_for_tags(tags: dict, durations: dict):
    """
    Generate audio files based on object presence duration.

    :param tags: dict - Mapping of tags to audio prompts.
    :param durations: dict - Mapping of tags to their duration in seconds.
    :return: list - List of generated audio file paths.
    """
    all_audio_files = []
    
    for tag, prompt in tags.items():
        duration = durations.get(tag, 3.0)  # Default to 3s if duration not found
        audio_files = generate_audio_files(tag, prompt, duration)
        print(f"🔊 Generated Files for {tag} ({duration}s): {audio_files}")
        all_audio_files.extend(audio_files)

    return all_audio_files


def play_audio_files(audio_files):
    """
    Plays a list of audio files using IPython's Audio player.
    
    This function is designed for use in Google Colab or Jupyter Notebook
    environments to play generated audio files interactively.
    
    :param audio_files: list[str] - A list of filenames of audio files to play.
    """
    for file in audio_files:
        print(f"🔊 Playing: {file}")
        display(ipd.Audio(file, autoplay=False))