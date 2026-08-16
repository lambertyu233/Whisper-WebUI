import subprocess
import time
from typing import Optional, Union, List, Tuple
import soundfile as sf
import os
import numpy as np
import gradio as gr
from faster_whisper.audio import decode_audio

from modules.utils.files_manager import is_video, get_media_files
from modules.utils.logger import get_logger

logger = get_logger()

CODEC_TO_EXT = {
    "aac": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "flac": ".flac",
    "vorbis": ".ogg",
    "ac3": ".ac3",
    "eac3": ".eac3",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s32le": ".wav",
    "pcm_f32le": ".wav",
    "pcm_u8": ".wav",
    "wav": ".wav",
    "alac": ".m4a",
    "dts": ".dts",
    "dca": ".dts",
    "truehd": ".thd",
    "wmav1": ".wma",
    "wmav2": ".wma",
    "wmapro": ".wma",
}


def validate_audio(audio: Optional[str] = None):
    """Validate audio file and check if it's corrupted"""
    if isinstance(audio, np.ndarray):
        return True

    if not os.path.exists(audio):
        logger.info(f"The file {audio} does not exist. Please check the path.")
        return False

    try:
        audio = decode_audio(audio)
        return True
    except Exception as e:
        logger.info(f"The file {audio} is not able to open or corrupted. Please check the file. {e}")
        return False


def get_audio_codec(file_path: str) -> Optional[str]:
    """Get the audio codec of the first audio stream using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        codec = result.stdout.strip().lower()
        return codec if codec else None
    except Exception as e:
        logger.warning(f"Failed to probe audio codec for {file_path}: {e}")
        return None


def extract_audio_from_video(video_path: str, output_dir: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Extract original audio stream from video file without re-encoding.
    Saves the extracted audio file alongside the video file (or in output_dir).

    Returns:
        (success: bool, output_path: str, message: str)
    """
    if not os.path.exists(video_path):
        return False, "", f"文件不存在: {video_path}"

    base_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(video_path))
    file_stem = os.path.splitext(os.path.basename(video_path))[0]

    codec = get_audio_codec(video_path)
    ext = CODEC_TO_EXT.get(codec, ".m4a")

    output_path = os.path.join(base_dir, f"{file_stem}_speech{ext}")

    # 1. Try direct stream copy (zero re-encoding loss, fastest)
    try:
        cmd_copy = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, output_path, f"成功无损提取音频: {output_path} (编码: {codec or '未知'})"
    except Exception as e_copy:
        logger.warning(f"Stream copy failed for {video_path}, falling back to wav extraction: {e_copy}")

    # 2. Fallback: Extract to wav if stream copy failed
    try:
        output_path = os.path.join(base_dir, f"{file_stem}_speech.wav")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            output_path
        ]
        subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, output_path, f"成功提取音频 (兼容模式 WAV): {output_path}"
    except Exception as e_fallback:
        return False, "", f"提取音频失败 {video_path}: {e_fallback}"

    return False, "", f"未能生成有效音频文件: {video_path}"


def batch_extract_audio(
    files: Optional[Union[List[str], str]] = None,
    input_folder_path: Optional[str] = None,
    include_subdirectory: bool = False,
    progress=gr.Progress()
) -> Tuple[str, List[str]]:
    """
    Batch extract audio from selected video files or folder.
    Saves audio files alongside each video file.
    """
    video_files = []

    if input_folder_path and os.path.isdir(input_folder_path):
        media_files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)
        video_files = [f for f in media_files if is_video(f)]
    elif files:
        if isinstance(files, str):
            files = [line.strip() for line in files.split('\n') if line.strip()]
        video_files = [f for f in files if is_video(f) and os.path.exists(f)]

    if not video_files:
        return "未发现可提取音频的视频文件。请确保选择了有效的视频文件！", []

    total_files = len(video_files)
    success_count = 0
    extracted_paths = []
    log_lines = []

    start_time = time.time()
    for idx, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        if progress is not None:
            progress((idx + 1) / total_files, desc=f"正在提取音频 ({idx + 1}/{total_files}): {filename}")

        t0 = time.time()
        success, out_path, msg = extract_audio_from_video(video_path)
        elapsed = time.time() - t0

        if success:
            success_count += 1
            extracted_paths.append(out_path)
            log_lines.append(f"[{idx + 1}/{total_files}] 成功 ({elapsed:.2f}s): {filename} -> {os.path.basename(out_path)}")
        else:
            log_lines.append(f"[{idx + 1}/{total_files}] 失败: {filename} ({msg})")

    total_elapsed = time.time() - start_time
    summary = f"音频提取完成！共处理 {total_files} 个视频文件，成功 {success_count} 个，总耗时 {total_elapsed:.2f} 秒。\n音频文件已保存在对应视频同级目录下。\n\n" + "\n".join(log_lines)
    return summary, extracted_paths

