# download yt video using yt-dlp
import subprocess

video_url = "https://www.youtube.com/watch?v=SN-vBnWj6e8"
output_path = "../data/eleo_audio.mp3"

def download_audio(video_url, output_path):
    try:
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, video_url],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during download:", e.stderr)

download_audio(video_url, output_path)