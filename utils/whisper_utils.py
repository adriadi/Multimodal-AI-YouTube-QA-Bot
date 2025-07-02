import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# added `save_json` parameter
def transcribe_audio(file_path: str, save_path: str = None, save_json: bool = False) -> str:
    """
    Transcribes an audio file using OpenAI Whisper API or local model on failure.

    Args:
        file_path (str): Path to the .mp3/.wav audio file
        save_path (str, optional): If provided, saves transcript to .txt
        save_json (bool): If True, saves transcript to JSON

    Returns:
        str: Transcript text
    """
    if not file_path or not os.path.exists(file_path):
        raise ValueError("❌ No valid audio file provided.")

    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcript = response.text

    except RateLimitError:
        print("⚠️ OpenAI quota exceeded. Falling back to local Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        transcript = result["text"]

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    if save_json:
        json_path = save_path.replace(".txt", ".json") if save_path else file_path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"transcript": transcript}, f, ensure_ascii=False, indent=2)

    return transcript

# Tool wrapper
from langchain.tools import Tool

whisper_tool = Tool(
    name="Whisper Speech Tool",
    func=transcribe_audio,
    description="Transcribes a .wav or .mp3 audio file into text using OpenAI Whisper."
)
