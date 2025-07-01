import yt_dlp
import json
import os

def extract_youtube_metadata(video_url: str, save_path: str = None) -> dict:
    """
    Extract metadata from a YouTube video using yt_dlp.

    Args:
        video_url (str): Full YouTube video URL
        save_path (str, optional): Path to save metadata JSON file

    Returns:
        dict: Extracted metadata
    """
    ydl_opts = {"quiet": True, "skip_download": True, "forcejson": True}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    metadata = {
        "video_id": info["id"],
        "title": info["title"],
        "description": info["description"],
        "channel": info["channel"],
        "publish_date": info.get("upload_date"),
        "duration_sec": info.get("duration"),
        "url": info["webpage_url"]
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata
