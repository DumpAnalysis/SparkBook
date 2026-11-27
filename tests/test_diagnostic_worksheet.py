"""Check the evidence and generation boundaries of the paired diagnostic study."""
import importlib.util
import json
import unittest
from pathlib import Path

READER = Path(__file__).resolve().parents[1] / 'spark-book' / 'reader'
spec = importlib.util.spec_from_file_location('heldout_answers', READER / 'heldout_answers.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticWorksheet(unittest.TestCase):
    def test_only_user_evidence_reaches_either_protocol(self):
        readings = diagnostics.load_readings()
        case = {'id': 'PRIVATE_CASE_ID', 'messages': [
            {'role': 'user', 'content': 'A gauge reports an unavailable temperature.'},
            {'role': 'assistant', 'content': 'PRIVATE_REFERENCE'},
            {'role': 'system', 'content': 'PRIVATE_FIXTURE_RULE'},
        ]}
        def complete(chat, limit):
            for secret in ('PRIVATE_CASE_ID', 'PRIVATE_REFERENCE', 'PRIVATE_FIXTURE_RULE'):
                self.assertNotIn(secret, json.dumps(chat))
            return {'answer': 'An unedited reply.', 'finish': 'stop'}

        for protocol in ('direct', 'review'):
            record = diagnostics.generate_case(case, protocol, complete, readings)
            self.assertEqual(record['finish'], 'stop')
            self.assertIn('An unedited reply.', record['answer'])

    def test_truncation_survives_field_assembly(self):
        calls = []
        def complete(chat, limit):
            calls.append(limit)
            finish = 'length' if len(calls) == 2 else 'stop'
            return {'answer': f'field {len(calls)}', 'finish': finish}

        readings = diagnostics.load_readings()
        case = {'id': 'novel', 'messages': [{'role': 'user', 'content': 'xyzzy'}]}
        record = diagnostics.generate_case(case, 'review', complete, readings)
        self.assertEqual(record['selection']['section'], 'reasoning')
        self.assertEqual(calls, [48, 56, 88])
        self.assertEqual(record['finish'], 'incomplete')
        self.assertEqual(record['fields']['Hypothesis']['finish'], 'length')
        self.assertIn('Hypothesis: field 2', record['answer'])
