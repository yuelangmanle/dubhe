import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import quality_eval


class QualityTests(unittest.TestCase):
    def test_fixtures_are_reproducible_and_pages_can_be_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = [sys.executable, str(ROOT / "tools/quality_eval.py"), "fixtures"]
            subprocess.run([*args, str(root / "one")], check=True, capture_output=True)
            subprocess.run([*args, str(root / "two")], check=True, capture_output=True)
            names = {path.name for path in (root / "one").iterdir()}
            self.assertEqual(names, {path.name for path in (root / "two").iterdir()})
            for name in names:
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            pdf = (root / "one" / "resume-20.pdf").read_bytes()
            self.assertEqual(len(re.findall(rb"/Type /Page\b", pdf)), 20)
            for page in range(1, 21):
                self.assertIn(f"DUBHE-PAGE-{page:02}".encode(), pdf)
            manifest = json.loads((root / "one" / "cases.json").read_bytes())
            for case in manifest["ocr"]:
                if "pdf" in case:
                    self.assertIn(case["pdf"], names)
            self.assertEqual({case["mode"] for case in manifest["ocr"]}, {"text", "document", "table", "formula"})
            self.assertEqual({case["target"] for case in manifest["translation"]}, {"en", "zh"})

    def test_checks_capture_actual_text_errors_and_reading_order(self):
        case = {"terms": [["North"], ["18.25"], ["\u5165\u53e3"]], "order": ["North", "South"]}
        self.assertEqual(quality_eval.checks(case, "NORTH 18.25 \u5165\u53e3\nSouth"), [])
        self.assertEqual(quality_eval.checks(case, "South\nNorth 118.25 \u5165\u53e3"),
                         ["18.25", "North → South"])
        formula = {"terms": [["x^2"], ["y^2"], ["=25"]]}
        self.assertEqual(quality_eval.checks(formula, r"\(x^{2} + y^{2} = 25\)"), [])
        self.assertEqual(quality_eval.checks(formula, r"\[\mathbf{x}^{2}+\mathbf{y}^{2}=25\]"), [])

    def test_stream_measures_visible_answer_and_validates_finish(self):
        sample = io.BytesIO(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
                            b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n'
                            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n'
                            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7}}\n'
                            b'data: [DONE]\n')
        result = quality_eval.read_completion(sample, 9.9, lambda: 10.0)
        self.assertEqual(result["first_visible_ms"], 100)
        self.assertEqual(result["prompt_tokens"], 7)
        with self.assertRaisesRegex(ValueError, "未完整"):
            quality_eval.read_completion(io.BytesIO(b'data: {"choices":[]}\n'), 9.9, lambda: 10.0)


if __name__ == "__main__":
    unittest.main()
