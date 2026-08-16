import unittest
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.whisper.data_classes import Segment
from modules.utils.subtitle_manager import is_repetitive_text, clean_repetitive_segments, generate_file, WriteSRT


class TestRepetitionFilter(unittest.TestCase):

    def test_repetitive_hallucination_cases(self):
        # Cases requested by user
        self.assertTrue(is_repetitive_text("唰，唰，唰，"))
        self.assertTrue(is_repetitive_text("あ、あ、あ、"))
        self.assertTrue(is_repetitive_text("うぅぅぅ"))
        self.assertTrue(is_repetitive_text("うぅぅぅぅ"))
        self.assertTrue(is_repetitive_text("你是你是你是你是"))
        self.assertTrue(is_repetitive_text("あ〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜"))
        self.assertTrue(is_repetitive_text("めっちゃめっちゃめっちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃめちゃ"))

        # Extra common hallucination patterns
        self.assertTrue(is_repetitive_text("啊啊啊啊啊啊"))
        self.assertTrue(is_repetitive_text("thank you, thank you, thank you, thank you"))
        self.assertTrue(is_repetitive_text("....."))
        self.assertTrue(is_repetitive_text("   "))
        self.assertTrue(is_repetitive_text("哈哈哈哈哈哈哈哈"))
        self.assertTrue(is_repetitive_text("yes, yes, yes, yes"))

    def test_normal_sentences_not_filtered(self):
        # Normal sentences should NOT be filtered
        self.assertFalse(is_repetitive_text("好的，谢谢你。"))
        self.assertFalse(is_repetitive_text("是的，我知道了。"))
        self.assertFalse(is_repetitive_text("天天开心"))
        self.assertFalse(is_repetitive_text("快跑！快跑！"))
        self.assertFalse(is_repetitive_text("对对对，我们现在马上出发去机场吧。"))
        self.assertFalse(is_repetitive_text("Hello world, this is a test."))
        self.assertFalse(is_repetitive_text("Thank you very much."))

    def test_clean_repetitive_segments_list(self):
        segments = [
            Segment(start=0.0, end=1.0, text="大家好，欢迎来到本期节目。"),
            Segment(start=1.0, end=2.0, text="唰，唰，唰，"),
            Segment(start=2.0, end=3.0, text="今天我们要介绍的是一个新功能。"),
            Segment(start=3.0, end=4.0, text="うぅぅぅぅ"),
            Segment(start=4.0, end=5.0, text="你是你是你是你是"),
            Segment(start=5.0, end=6.0, text="谢谢大家的观看！"),
        ]

        # When filter_repetition=False, nothing should be removed
        unfiltered = clean_repetitive_segments(segments, filter_repetition=False)
        self.assertEqual(len(unfiltered), 6)

        # When filter_repetition=True, 3 hallucination segments should be removed
        filtered = clean_repetitive_segments(segments, filter_repetition=True)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0].text, "大家好，欢迎来到本期节目。")
        self.assertEqual(filtered[1].text, "今天我们要介绍的是一个新功能。")
        self.assertEqual(filtered[2].text, "谢谢大家的观看！")

    def test_clean_repetitive_segments_dict(self):
        result_dict = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "正常第一句"},
                {"start": 1.0, "end": 2.0, "text": "あ、あ、あ、"},
                {"start": 2.0, "end": 3.0, "text": "正常第二句"},
            ]
        }

        filtered_dict = clean_repetitive_segments(result_dict, filter_repetition=True)
        self.assertEqual(len(filtered_dict["segments"]), 2)
        self.assertEqual(filtered_dict["segments"][0]["text"], "正常第一句")
        self.assertEqual(filtered_dict["segments"][1]["text"], "正常第二句")

    def test_generate_srt_file_with_filtering(self):
        temp_dir = tempfile.mkdtemp()
        try:
            segments = [
                Segment(start=0.0, end=1.0, text="First subtitle line"),
                Segment(start=1.0, end=2.0, text="唰，唰，唰，"),
                Segment(start=2.0, end=3.0, text="Second subtitle line"),
            ]

            # Generate with filter_repetition=True
            content, file_path = generate_file(
                output_format="srt",
                output_dir=temp_dir,
                result=segments,
                output_file_name="test_filtered",
                add_timestamp=False,
                filter_repetition=True
            )

            # Check that content contains lines 1 and 2, but not the hallucinated line
            self.assertIn("1\n00:00:00,000 --> 00:00:01,000\nFirst subtitle line", content)
            self.assertIn("2\n00:00:02,000 --> 00:00:03,000\nSecond subtitle line", content)
            self.assertNotIn("唰，唰，唰，", content)
            self.assertNotIn("3\n", content)  # Should only have 2 subtitle blocks, so index 3 should not exist

            # Generate with filter_repetition=False
            content_unfiltered, _ = generate_file(
                output_format="srt",
                output_dir=temp_dir,
                result=segments,
                output_file_name="test_unfiltered",
                add_timestamp=False,
                filter_repetition=False
            )
            self.assertIn("唰，唰，唰，", content_unfiltered)
            self.assertIn("3\n", content_unfiltered)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
