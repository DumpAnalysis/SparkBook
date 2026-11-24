"""Exercise the portable supervisor's decisions with fixture data on any CPU."""
import datetime as dt
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from guarded import acceptance, require_pilot, report
from model_stage import PROFILE, admit
from windows_watch import GIB, GuardStop, validate

HERE = Path(__file__).resolve().parent
NOW = dt.datetime(2026, 10, 31, tzinfo=dt.timezone.utc)


class Controls(unittest.TestCase):
    def test_historical_windows_samples(self):
        data = json.loads((HERE/'observations/windows-allocation.json').read_text())
        self.assertEqual(len(data['memory-above-guest']), 6)
        for rows in data.values():
            for row in rows:
                observed = dt.datetime.fromisoformat(row['utc'].replace('Z', '+00:00')).timestamp()
                validate(row, 4*GIB, observed)

    def test_host_stops(self):
        row = dict(utc=NOW.isoformat(), available_bytes=5*GIB, committed_bytes=GIB,
                   commit_limit_bytes=9*GIB, query_seconds=.25)
        for change in [dict(available_bytes=4*GIB-1), dict(committed_bytes=6*GIB),
                       dict(query_seconds=5.01), dict(query_seconds=float('nan')),
                       dict(utc=(NOW-dt.timedelta(seconds=9)).isoformat()), dict(utc=''),
                       dict(available_bytes=None)]:
            with self.subTest(change=change), self.assertRaises(GuardStop):
                validate(dict(row, **change), 4*GIB, NOW.timestamp())

    def test_model_admission(self):
        gpu = PROFILE['placement']['GPU_planning_bytes']
        cpu = PROFILE['placement']['CPU_planning_bytes']
        host = dict(available_bytes=4*GIB, committed_bytes=GIB, commit_limit_bytes=5*GIB)
        self.assertTrue(all(admit(gpu, gpu-GIB, cpu, host).values()))
        self.assertFalse(admit(gpu-1, gpu, cpu, host)['cuda_free'])
        self.assertFalse(admit(gpu, gpu, cpu-1, host)['guest_available'])
        self.assertFalse(admit(gpu, gpu-2*GIB, cpu, host)['host_available'])
        self.assertFalse(admit(gpu, gpu, cpu, dict(host, committed_bytes=2*GIB))['host_commit_headroom'])

    def test_pilot_identity_and_release(self):
        events = [dict(event='result', status='PASSED', allocated_written_verified_bytes=GIB),
                  dict(event='memory', stage='after_release'), dict(event='process_result', exit_code=0)]
        identity = dict(boot='example', interpreter='/lab/bin/python', distro='test')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'result.json').write_text(json.dumps(dict(mode='pilot',status='PASSED',
                worker_exit=0, identity=identity, host_samples=3)))
            (root/'stdout.txt').write_text('\n'.join(json.dumps(e) for e in events))
            require_pilot(root, identity)
            with self.assertRaises(GuardStop):
                require_pilot(root, dict(identity, boot='another boot'))
        for altered in [events[:1]+events[2:],events[:-1],events[:-1]+[dict(event='process_result',exit_code=1)]]:
            with self.assertRaises(RuntimeError):
                acceptance(altered, 1)
        with self.assertRaises(RuntimeError):
            acceptance(events, 18)

    def test_report_aligns_utc_and_skips_since_boot_row(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            def write_json(filename, data):
                (root/filename).write_text(json.dumps(data))
            write_json('result.json', dict(mode='model', status='PASSED', host_samples=2))
            write_json('native-exit.json', dict(exit=0))
            start = '2026-10-31T12:00:02+00:00'
            stop = '2026-10-31T12:00:03+00:00'
            write_json('requests.json', dict(placement=[51,65], requests=[
                dict(label='pressure', started_utc=start, ended_utc=stop)]))
            sample = dict(available_bytes=6*GIB,commit_limit_bytes=12*GIB,committed_bytes=3*GIB)
            (root/'host.jsonl').write_text(json.dumps(dict(sample,utc=start))+'\n')
            (root/'vmstat.txt').write_text(' r b si so id UTC\n'
                ' 1 0 88 99 0 2026-10-31 12:00:01\n 1 0 2 3 98 2026-10-31 12:00:02\n')
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(report(root),0)
            summary = json.loads(stream.getvalue())
            self.assertIsNone(summary['guest_window']['previous'])
            self.assertEqual(summary['guest_window']['inside'][0]['so'],3)
            self.assertEqual(summary['host_window']['inside'][0]['commit_headroom_bytes'],9*GIB)


if __name__ == '__main__':
    unittest.main(verbosity=2)
