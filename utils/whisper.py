import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(file_path: str, save_path: str = None) -> str:
    """
    Transcribes an audio file using Whisper (OpenAI API).

    Args:
        file_path (str): Path to the .mp3/.wav audio file
        save_path (str, optional): If given, saves the result to .txt

    Returns:
        str: The transcript text
    """
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    transcript_text = response.text

    if save_path:
        with open(save_path, "w") as f:
            f.write(transcript_text)

    return transcript_text
