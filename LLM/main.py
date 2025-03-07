from config import OPENAI_API_KEY
from decorator import exponential_retry
from openai import OpenAI


client = OpenAI(api_key=OPENAI_API_KEY)

@exponential_retry
def get_sound_relevant_tags(tags):
    """
    Filters a list of tags to include only those associated with real-world sounds.
    
    This function uses an AI model to identify which tags correspond to objects or phenomena
    that produce recognizable sounds. Tags such as 'traffic lights' are excluded, as they do not
    generate sounds.
    
    The function is decorated with @exponential_retry to handle API failures with exponential backoff.
    
    :param tags: list[str] - A list of tags to filter.
    :return: list[str] - A list of tags that correspond to real-world sounds.
    """
    message = [
        {"role": "system", "content": "You are an AI that filters tags to select only those associated with real-world sounds. Animals, people, cars, trees, hammers, wind, rain, whatever that can have a recognisable sound."},
        {"role": "user", "content": f"Filter these tags and return only the ones that can have real sounds: {tags}. The return must contain only list of objects, separated by commas. Traffic lights do not give a sound."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        max_tokens=50
    )

    my_list = response.choices[0].message.content.split(", ")
    print(f"[INFO] Relevant Tags: {my_list}")
    return my_list


@exponential_retry
def generate_audio_prompts(tags):
    """
    Generates text prompts for an audio model based on sound-relevant tags.
    
    This function first filters the input tags using `get_sound_relevant_tags()` to ensure only
    sound-producing objects are used. Then, it generates descriptive audio prompts that specify
    how each object should sound.
    
    The function is decorated with @exponential_retry to handle API failures with exponential backoff.
    
    :param tags: list[str] - A list of tags representing objects.
    :return: dict[str, str] - A dictionary mapping tags to their corresponding audio prompts.
    """
    filtered_tags = get_sound_relevant_tags(tags)
    prompts = {}

    for tag in filtered_tags:
        message = [
            {"role": "system", "content": "You are an AI that generates prompts for an Audio Model based on given tags."},
            {"role": "user", "content": f"Generate a short prompt for Audio Model for this object: {tag}. Example response when the tag is person result should be: Sound for person laughing. When its car: Sound of loud car on the road."}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=10
        )

        prompts[tag] = response.choices[0].message.content

    print(f"[INFO] Prompts: {prompts}")
    return prompts