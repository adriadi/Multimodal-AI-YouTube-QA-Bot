import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# added `save_json` parameter
def transcribe_audio(file_path: str, save_path: str = None, save_json: bool = False) -> str:
    """
    Transcribes an audio file using Whisper (OpenAI API).

    Args:
        file_path (str): Path to the .mp3/.wav audio file
        save_path (str, optional): If given, saves the result to .txt
        save_json (bool): If True, also saves transcript as JSON

    Returns:
        str: The transcript text
    """
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    transcript_text = response.text

    # write plain text
    if save_path:
        with open(save_path, "w") as f:
            f.write(transcript_text)

    # write json version
    if save_json:
        json_path = save_path.replace(".txt", ".json") if save_path else file_path.replace(".mp3", ".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"transcript": transcript_text}, jf, indent=2)

    return transcript_text

# Tool wrapper
from langchain.tools import Tool

whisper_tool = Tool(
    name="Whisper Speech Tool",
    func=transcribe_audio,
    description="Transcribes a .wav or .mp3 audio file into text using OpenAI Whisper."
)
