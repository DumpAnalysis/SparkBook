"""Guard audited Appendix G sources and the reader records shipped in new labs."""
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest


BOOK = Path(__file__).resolve().parents[1] / "spark-book"
READER = BOOK / "reader"


class ReaderSources(unittest.TestCase):
    def test_audited_answers_point_to_the_supporting_exercises(self):
        expected = {
            (8, 1): 9, (16, 2): 19, (19, 2): 16, (37, 1): 52, (53, 1): 83,
            (54, 1): 85, (54, 2): 86, (55, 1): 88, (55, 2): 88, (57, 2): 98,
            (58, 1): 100, (70, 1): 129, (71, 2): 130, (81, 2): 173,
            (86, 1): 195, (86, 2): 193, (89, 2): 207, (92, 1): 221, (97, 2): 238,
        }
        text = (READER / "SELF_CHECKS.txt").read_text()
        revision = re.search(r"illustrated revision (\d+)", text.splitlines()[0]).group(1)
        found = {}
        for line in text.splitlines():
            if match := re.match(r"Chapter (\d+) ·", line):
                chapter = int(match[1])
            elif match := re.match(r"([12])\. ", line):
                question = int(match[1])
            elif match := re.fullmatch(rf"   Source · Exercise (\d+)\. Supporting passage on R{revision} book p\. [1-9]\d*\.", line):
                found[chapter, question] = int(match[1])
        self.assertEqual(found, expected)

    def test_stream_topics_route_to_the_worked_exercise(self):
        catalog = json.loads((READER / "technology-overviews.json").read_text())
        for term, exercise in (("caller-selected stream", 58), ("double buffering", 63),
                               ("event ordering", 63), ("CUDA Graphs", 65), ("graph replay", 65)):
            with self.subTest(term=term):
                output = subprocess.check_output(
                    [sys.executable, str(READER / "technologies.py"), "--find", term], text=True)
                self.assertEqual(re.findall(r"\|  Exercise (\d+)  \|", output), [str(exercise)])
        for exercise in (63, 64, 65, 134):
            expected = "graphs" if exercise == 65 else "buffers"
            self.assertEqual(catalog["exercises"][str(exercise)][0], expected)

    def test_dated_reader_records_are_in_the_evidence_index_and_lab(self):
        index = json.loads((BOOK / "EXERCISE_EVIDENCE_MAP.json").read_text())["exercises"]
        shipped = {row["path"] for row in json.loads((READER / "workspace-files.json").read_text())["files"]}
        for number in (39, 40, 129, 186):
            with self.subTest(exercise=number):
                relative = f"reader/reader-runs/{number:03}.txt"
                entry = next(row for row in index if row["number"] == number)
                self.assertIn(relative, entry["companion_files"])
                self.assertIn(relative, entry["record_locator"])
                self.assertIn(relative, shipped)
                text = (BOOK / relative).read_text()
                self.assertIn(f"Reader record {number:03}.R1", text)
                self.assertIn("Exit  0", text)
