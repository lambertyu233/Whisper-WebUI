# Ported from https://github.com/openai/whisper/blob/main/whisper/utils.py

import json
import os
import re
import sys
import zlib
from collections import Counter
from typing import Callable, List, Optional, TextIO, Union, Dict, Tuple
from datetime import datetime

from modules.whisper.data_classes import Segment, Word
from .files_manager import read_file
from .logger import get_logger

logger = get_logger()


def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds is not None and seconds >= 0, "Wrong timestamp provided"

    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def time_str_to_seconds(time_str: str, decimal_marker: str = ",") -> float:
    times = time_str.split(":")

    if len(times) == 3:
        hours, minutes, rest = times
        hours = int(hours)
    else:
        hours = 0
        minutes, rest = times

    seconds, fractional = rest.split(decimal_marker)

    minutes = int(minutes)
    seconds = int(seconds)
    fractional_seconds = float("0." + fractional)

    return hours * 3600 + minutes * 60 + seconds + fractional_seconds


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )



def is_repetitive_text(text: Optional[str]) -> bool:
    if not text:
        return False

    raw_text = text.strip()
    if not raw_text:
        return True

    # 1. Check raw character / symbol repetition (covers elongated symbols like 〜, ~, ー, -, ., 。, ellipses)
    raw_match = re.search(r'(.)\1{3,}', raw_text)
    if raw_match:
        rep_len = len(raw_match.group(0))
        if rep_len >= 5 or (rep_len / len(raw_text) >= 0.5):
            return True

    # 2. Clean punctuation/whitespace: if empty after removing non-alphanumeric/CJK chars
    clean_chars = re.sub(r'[\s\W_]+', '', raw_text, flags=re.UNICODE)
    if not clean_chars:
        return True

    # 3. Single character repeated in cleaned text
    match = re.search(r'(.)\1{3,}', clean_chars)
    if match:
        repeat_len = len(match.group(0))
        if repeat_len >= 5 or (len(clean_chars) <= 12 and repeat_len >= 4) or (repeat_len / len(clean_chars) >= 0.5):
            return True

    # 4. Low character diversity (e.g. "うぅぅぅぅ" containing only 2 unique chars, or heavy character repetition)
    unique_chars = set(clean_chars)
    if len(clean_chars) >= 4 and len(unique_chars) == 1:
        return True
    if len(clean_chars) >= 5 and len(unique_chars) <= 2:
        counts = Counter(clean_chars)
        max_freq = max(counts.values())
        if max_freq / len(clean_chars) >= 0.6:
            return True
    if len(clean_chars) >= 10 and len(unique_chars) <= 4:
        counts = Counter(clean_chars)
        top2_freq = sum(c for _, c in counts.most_common(2))
        if top2_freq / len(clean_chars) >= 0.5:
            return True

    # 5. Token-based repetition with delimiters (e.g. "唰，唰，唰，", "あ、あ、あ、")
    tokens = [t for t in re.split(r'[\s,，、。！？!?；;:：…—·~〜～\-_]+', raw_text) if t]
    if len(tokens) >= 3:
        counts = Counter(tokens)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3 and (most_common_count / len(tokens)) >= 0.7:
            return True
        if len(counts) <= 2 and len(tokens) >= 4:
            return True

    # 6. Periodic repetition across the entire string (e.g. "你是你是你是你是")
    for pattern_len in range(1, min(15, len(clean_chars) // 2 + 1)):
        pattern = clean_chars[:pattern_len]
        repeat_count = len(clean_chars) // pattern_len
        if repeat_count >= 3:
            reconstructed = pattern * repeat_count
            remainder = clean_chars[len(reconstructed):]
            if pattern.startswith(remainder) and clean_chars.startswith(reconstructed):
                return True

    # 7. Substring / phrase repetitions (iterate through all matched repeating chunks)
    total_repeat_len = 0
    for m in re.finditer(r'(.{1,12}?)\1{2,}', clean_chars):
        matched_str = m.group(0)
        unit = m.group(1)
        repeat_num = len(matched_str) // len(unit)
        if repeat_num >= 4:
            return True
        if repeat_num >= 3 and len(matched_str) / len(clean_chars) >= 0.5:
            return True
        total_repeat_len += len(matched_str)

    if total_repeat_len / len(clean_chars) >= 0.7:
        return True

    return False



def clean_repetitive_segments(
    result: Union[dict, List[Segment]],
    filter_repetition: bool = True
) -> Union[dict, List[Segment]]:
    if not filter_repetition or not result:
        return result

    if isinstance(result, list):
        filtered = []
        for seg in result:
            text = getattr(seg, "text", None) if isinstance(seg, Segment) else (seg.get("text", "") if isinstance(seg, dict) else "")
            if is_repetitive_text(text):
                start = getattr(seg, "start", None) if isinstance(seg, Segment) else (seg.get("start") if isinstance(seg, dict) else None)
                end = getattr(seg, "end", None) if isinstance(seg, Segment) else (seg.get("end") if isinstance(seg, dict) else None)
                logger.info(f"Filtered repetitive segment: [{start} -> {end}] {text}")
                continue
            filtered.append(seg)
        return filtered
    elif isinstance(result, dict) and "segments" in result:
        filtered_segments = []
        for seg in result["segments"]:
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            if is_repetitive_text(text):
                start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
                end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
                logger.info(f"Filtered repetitive segment: [{start} -> {end}] {text}")
                continue
            filtered_segments.append(seg)
        result["segments"] = filtered_segments
        return result

    return result


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
        self, result: Union[dict, List[Segment]], output_file_name: str,
            options: Optional[dict] = None, filter_repetition: bool = False, **kwargs
    ):
        if filter_repetition or kwargs.get("filter_repetition", False):
            result = clean_repetitive_segments(result, filter_repetition=True)

        if isinstance(result, List):
            if result and isinstance(result[0], Segment):
                result = {"segments": [seg.model_dump() for seg in result]}
            else:
                result = {"segments": result}

        output_path = os.path.join(
            self.output_dir, output_file_name + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        raise NotImplementedError

    def to_segments(self, file_path: str):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
        self, result: Union[Dict, List[Segment]], file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)

    def to_segments(self, file_path: str):
        segments = []

        blocks = read_file(file_path).split('\n')

        for block in blocks:
            segments.append(Segment(
                start=None,
                end=None,
                text=block
            ))
        return segments


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
        self,
        result: dict,
        options: Optional[dict] = None,
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        highlight_words: bool = False,
        align_lrc_words: bool = False,
        max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        align_lrc_words = align_lrc_words or options.get("align_lrc_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: List[dict] = []
            last: float = get_start(result["segments"]) or 0.0
            for segment in result["segments"]:
                chunk_index = 0
                words_count = max_words_per_line
                while chunk_index < len(segment["words"]):
                    remaining_words = len(segment["words"]) - chunk_index
                    if max_words_per_line > len(segment["words"]) - chunk_index:
                        words_count = remaining_words
                    for i, original_timing in enumerate(
                        segment["words"][chunk_index : chunk_index + words_count]
                    ):
                        timing = original_timing.copy()
                        long_pause = (
                            not preserve_segments and timing["start"] - last > 3.0
                        )
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                        if (
                            line_len > 0
                            and has_room
                            and not long_pause
                            and not seg_break
                        ):
                            # line continuation
                            line_len += len(timing["word"])
                        else:
                            # new line
                            timing["word"] = timing["word"].strip()
                            if (
                                len(subtitle) > 0
                                and max_line_count is not None
                                and (long_pause or line_count >= max_line_count)
                                or seg_break
                            ):
                                # subtitle break
                                yield subtitle
                                subtitle = []
                                line_count = 1
                            elif line_len > 0:
                                # line break
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        last = timing["start"]
                    chunk_index += max_words_per_line
            if len(subtitle) > 0:
                yield subtitle

        if len(result["segments"]) > 0 and "words" in result["segments"][0] and result["segments"][0]["words"]:
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([word["word"] for word in subtitle])
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                if j == i
                                else word
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end

                if align_lrc_words:
                    lrc_aligned_words = [f"[{self.format_timestamp(sub['start'])}]{sub['word']}" for sub in subtitle]
                    l_start, l_end = self.format_timestamp(subtitle[-1]['start']), self.format_timestamp(subtitle[-1]['end'])
                    lrc_aligned_words[-1] = f"[{l_start}]{subtitle[-1]['word']}[{l_end}]"
                    lrc_aligned_words = ' '.join(lrc_aligned_words)
                    yield None, None, lrc_aligned_words

                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                if segment["text"] is None:
                    continue

                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        blocks = read_file(file_path).split('\n\n')

        for block in blocks:
            if block.strip() != '' and not block.strip().startswith("WEBVTT"):
                lines = block.strip().split('\n')
                time_line = lines[0].split(" --> ")
                start, end = time_str_to_seconds(time_line[0], self.decimal_marker), time_str_to_seconds(time_line[1], self.decimal_marker)
                sentence = ' '.join(lines[1:])

                segments.append(Segment(
                    start=start,
                    end=end,
                    text=sentence
                ))

        return segments


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        blocks = read_file(file_path).split('\n\n')

        for block in blocks:
            if block.strip() != '':
                lines = block.strip().split('\n')
                index = lines[0]
                time_line = lines[1].split(" --> ")
                start, end = time_str_to_seconds(time_line[0], self.decimal_marker), time_str_to_seconds(time_line[1], self.decimal_marker)
                sentence = ' '.join(lines[2:])

                segments.append(Segment(
                    start=start,
                    end=end,
                    text=sentence
                ))

        return segments


class WriteLRC(SubtitlesWriter):
    extension: str = "lrc"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            if "align_lrc_words" in kwargs and kwargs["align_lrc_words"]:
                print(f"{text}\n", file=file, flush=True)
            else:
                print(f"[{start}]{text}[{end}]\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        blocks = read_file(file_path).split('\n')

        for block in blocks:
            if block.strip() != '':
                lines = block.strip()
                pattern = r'(\[.*?\])'
                parts = re.split(pattern, lines)
                parts = [part.strip() for part in parts if part]

                for i, part in enumerate(parts):
                    sentence_i = i%2
                    if sentence_i == 1:
                        start_str, text, end_str = parts[sentence_i-1], parts[sentence_i], parts[sentence_i+1]
                        start_str, end_str = start_str.replace("[", "").replace("]", ""), end_str.replace("[", "").replace("]", "")
                        start, end = time_str_to_seconds(start_str, self.decimal_marker), time_str_to_seconds(end_str, self.decimal_marker)

                        segments.append(Segment(
                            start=start,
                            end=end,
                            text=text,
                        ))

        return segments


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    output_format = output_format.strip().lower().replace(".", "")

    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
        "lrc": WriteLRC
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
            result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)


def generate_file(
    output_format: str, output_dir: str, result: Union[dict, List[Segment]], output_file_name: str,
    add_timestamp: bool = True, filter_repetition: bool = False, **kwargs
) -> Tuple[str, str]:
    output_format = output_format.strip().lower().replace(".", "")
    output_format = "vtt" if output_format == "webvtt" else output_format

    if filter_repetition:
        result = clean_repetitive_segments(result, filter_repetition=True)

    if add_timestamp:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name += f"-{timestamp}"

    file_path = os.path.join(output_dir, f"{output_file_name}.{output_format}")
    file_writer = get_writer(output_format=output_format, output_dir=output_dir)

    if isinstance(file_writer, WriteLRC) and kwargs.get("highlight_words", False):
        kwargs["highlight_words"], kwargs["align_lrc_words"] = False, True

    file_writer(result=result, output_file_name=output_file_name, filter_repetition=filter_repetition, **kwargs)
    content = read_file(file_path)
    return content, file_path


def safe_filename(name):
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    MAX_FILENAME_LENGTH = 200
    safe_name = re.sub(INVALID_FILENAME_CHARS, '_', name)
    # Truncate the filename if it exceeds the max_length (MAX_FILENAME_LENGTH)
    if len(safe_name) > MAX_FILENAME_LENGTH:
        file_extension = safe_name.split('.')[-1]
        if len(file_extension) + 1 < MAX_FILENAME_LENGTH:
            truncated_name = safe_name[:MAX_FILENAME_LENGTH - len(file_extension) - 1]
            safe_name = truncated_name + '.' + file_extension
        else:
            safe_name = safe_name[:MAX_FILENAME_LENGTH]
    return safe_name
