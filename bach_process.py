
import json
import os
from video_pipeline import process_video

# Load list of videos to process
with open("videos_to_process.json", "r", encoding="utf-8") as f:
    video_list = json.load(f)

for entry in video_list:
    url = entry["url"]
    tags = entry.get("tags", [])

    # Extract video ID (naive split by v=)
    video_id = url.split("v=")[-1]

    # Define file paths for each step
    audio_path = f"data/audio/{video_id}.mp3"
    transcript_path = f"data/transcripts/{video_id}.txt"
    vectorstore_path = f"data/vectorstores/{video_id}_faiss"

    # Ensure folders exist
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    os.makedirs(vectorstore_path, exist_ok=True)

    # Run full processing pipeline
    result = process_video(
        video_url=url,
        audio_path=audio_path,
        transcript_path=transcript_path,
        vectorstore_path=vectorstore_path,
        tags=tags
    )

    print(result)
