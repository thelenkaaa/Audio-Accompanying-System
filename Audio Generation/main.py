from diffusers import StableAudioPipeline
from openai import OpenAI
import torch
import soundfile as sf
import IPython.display as ipd


SAMPLES_NUM = 1
MAX_FILENAME = 10
NUM_INFERENCE_STEPS = 50
NEGATIVE_PROMRT = "Low quality."


pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)


def generate_audio_files(tag, prompt):
    """
    Generates audio files based on a given tag and prompt.
    
    This function utilizes the StableAudioPipeline model to generate audio waveforms
    from text prompts. The audio is then saved as a .wav file.
    
    :param tag: str - A tag representing the category of the audio.
    :param prompt: str - A text prompt describing the sound to generate.
    :return: list[str] - A list of filenames of the generated audio files.
    """
    filenames = []
    audio_files = []

    for i in range(SAMPLES_NUM):
        # Generate audio using your model
        wavs = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMRT,
            num_inference_steps=NUM_INFERENCE_STEPS[i],
            audio_end_in_s=3.0,
            num_waveforms_per_prompt=1,  # Generate one waveform per iteration
            generator=generator,
        ).audios

        output = wavs[0].T.float().cpu().numpy()
        filename = f"{tag}_{i}.wav"
        
        # Save the audio file
        sf.write(filename, output, pipe.vae.sampling_rate)
        filenames.append(filename)
        audio_files.append(filename)

    return audio_files


def generate_audio_for_tags(tags: dict):
    """
    Generates audio files for multiple tags based on their respective prompts.
    
    This function iterates over a dictionary of tags and prompts, generating audio
    for each and collecting the filenames.
    
    :param tags: dict[str, str] - A dictionary where keys are tags and values are prompts.
    :return: list[str] - A list of filenames of generated audio files.
    """
    all_audio_files = []
    
    for tag, prompt in tags.items():
        audio_files = generate_audio_files(tag, prompt)
        print(f"FILES: {audio_files}")
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

# Generate audio files
# audio_files = generate_audio_for_tags(collected_prompts_audio_model)

# Show playable audio players
# play_audio_files(audio_files)
