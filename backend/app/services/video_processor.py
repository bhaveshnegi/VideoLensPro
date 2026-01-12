import json
import subprocess
from pathlib import Path
import ffmpeg

def get_metadata(video_path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json", str(video_path)
    ]
    output = subprocess.check_output(cmd, text=True)
    data = json.loads(output)
    stream = data["streams"][0]
    fps = eval(stream["avg_frame_rate"])
    
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": round(fps, 2),
        "duration_sec": float(stream["duration"])
    }

def extract_audio(video_path: Path, audio_path: Path):
    ffmpeg.input(str(video_path)).output(
        str(audio_path), ac=1, ar=16000
    ).overwrite_output().run(quiet=True)
    return audio_path

def extract_frames(video_path: Path, frames_dir: Path):
    frames_dir.mkdir(exist_ok=True)
    pattern = str(frames_dir / "frame_%04d.jpg")
    ffmpeg.input(str(video_path)).filter(
        "fps", fps=f"1/5"
    ).output(pattern).overwrite_output().run(quiet=True)
    return frames_dir