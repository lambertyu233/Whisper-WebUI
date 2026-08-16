import os
import subprocess
import tempfile
import pytest

from modules.utils.audio_manager import get_audio_codec, extract_audio_from_video, batch_extract_audio
from modules.utils.files_manager import is_video
from app import App


@pytest.fixture
def sample_video():
    """Create a temporary video file with AAC audio track using ffmpeg"""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        video_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    yield video_path

    # Clean up
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_is_video(sample_video):
    assert is_video(sample_video) is True
    assert is_video("test.mp3") is False
    assert is_video("test.wav") is False
    assert is_video("test.mkv") is True


def test_get_audio_codec(sample_video):
    codec = get_audio_codec(sample_video)
    assert codec == "aac"


def test_extract_audio_from_video(sample_video):
    success, out_path, msg = extract_audio_from_video(sample_video)
    assert success is True
    assert os.path.exists(out_path)
    assert os.path.dirname(out_path) == os.path.dirname(sample_video)
    assert out_path.endswith("_speech.m4a")
    assert os.path.getsize(out_path) > 0


def test_batch_extract_audio(sample_video):
    summary, out_files = batch_extract_audio(files=[sample_video], progress=None)
    assert len(out_files) == 1
    assert os.path.exists(out_files[0])
    assert "音频提取完成" in summary


def test_on_check_has_video():
    app_instance = object.__new__(App)
    
    # Empty -> False
    res = app_instance.on_check_has_video(None, "", "", False)
    assert (res.get("interactive") if isinstance(res, dict) else getattr(res, "interactive", False)) is False

    # Video in local_files -> True
    res_video = app_instance.on_check_has_video(None, "C:/path/to/movie.mp4", "", False)
    assert (res_video.get("interactive") if isinstance(res_video, dict) else getattr(res_video, "interactive", True)) is True

    # Audio in local_files -> False
    res_audio = app_instance.on_check_has_video(None, "C:/path/to/song.mp3", "", False)
    assert (res_audio.get("interactive") if isinstance(res_audio, dict) else getattr(res_audio, "interactive", False)) is False
