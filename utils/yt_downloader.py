# utils/yt_downloader.py
import subprocess

def download_audio(video_url: str, output_path: str):
    """
    Download audio from a YouTube video using yt-dlp and save as .mp3.

    Args:
        video_url (str): The full YouTube video URL
        output_path (str): File path where .mp3 will be saved
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, video_url],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error during download:", e.stderr)
