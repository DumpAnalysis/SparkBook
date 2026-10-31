"""Portable regression checks for media boundaries and numerical controls."""
import json,sys,tempfile,unittest,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile
import torch
from contracts import *
from diffusion_control import schedule,mix,reconstruct
from adapt_asr import dataset
from voice import local_url
ROOT=Path(__file__).parent
class Contracts(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def audio(self,x,rate=16000):
        f=self.p/'x.wav';wavfile.write(f,rate,x);return f
    def test_01_pcm16_scale(self):
        sr,x,_=read_audio(self.audio(np.array([-32768,0,16384],np.int16)));np.testing.assert_array_equal(x,[-1,0,.5])
    def test_02_uint8_center(self):
        _,x,_=read_audio(self.audio(np.array([0,128,192],np.uint8)));np.testing.assert_array_equal(x,[-1,0,.5])
    def test_03_resampling_preserves_duration(self):
        _,x,_=read_audio(self.audio(np.zeros(48000,np.float32),48000),16000);self.assertEqual(len(x),16000)
    def test_04_header_relabel_changes_duration(self):
        sr,x,_=read_audio(self.audio(np.zeros(48000,np.float32),16000));self.assertEqual(len(x)/sr,3)
    def test_05_antiphase_cancels(self):
        _,x,_=read_audio(self.audio(np.array([[.3,-.3],[.7,-.7]],np.float32)));np.testing.assert_array_equal(x,[0,0])
    def test_06_nan_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):read_audio(self.audio(np.array([np.nan],np.float32)))
    def test_07_empty_rejected(self):
        with self.assertRaisesRegex(ValueError,'empty'):read_audio(self.audio(np.array([],np.float32)))
    def test_08_overlong_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlong'):read_audio(self.audio(np.zeros(32000,np.float32)),max_seconds=1)
    def test_09_write_does_not_hide_clipping(self):
        with self.assertRaisesRegex(ValueError,'clipping'):write_audio(self.p/'o.wav',np.array([1.1]),16000)
    def test_10_frames_not_scalar_samples(self):
        s=audio_stats(np.zeros((32000,2)),16000);self.assertEqual((s['duration_s'],s['channels']),(2,2))
    def test_11_rgba_composited_on_white(self):
        f=self.p/'a.png';Image.new('RGBA',(3,2),(255,0,0,0)).save(f);self.assertEqual(image_rgb(f).getpixel((0,0)),(255,255,255))
    def test_12_image_pixel_limit(self):
        f=self.p/'a.png';Image.new('RGB',(3,2)).save(f)
        with self.assertRaisesRegex(ValueError,'pixel'):image_rgb(f,5)
    def test_13_word_substitution(self):self.assertEqual(word_error('GPU has thirty GB','gpu has thirteen gb')['wer'],.25)
    def test_14_wer_may_exceed_one(self):self.assertEqual(word_error('a','a b c')['wer'],2)
    def test_15_empty_reference_not_zero(self):
        r=word_error('','unexpected');self.assertIsNone(r['wer']);self.assertEqual(r['errors'],1)
    def test_16_schedule_monotone(self):
        b,a,ab=schedule();self.assertTrue(bool(torch.all(ab[1:]<ab[:-1])));self.assertTrue(bool(torch.all((b>0)&(b<1))))
    def test_17_oracle_recovers_clean_point(self):
        _,_,ab=schedule();x=torch.tensor([[1.,2.]]);e=torch.tensor([[.3,-.4]]);t=torch.tensor([99]);torch.testing.assert_close(reconstruct(mix(x,e,t,ab),e,t,ab),x)
    def test_18_wrong_noise_detected(self):
        _,_,ab=schedule();x=torch.ones(1,2);e=torch.ones(1,2);t=torch.tensor([99]);self.assertGreater(float((reconstruct(mix(x,e,t,ab),-e,t,ab)-x).abs().max()),1)
    def test_19_job_limit(self):self.assertEqual(admit({'kind':'music','tokens':513}),(False,'tokens limit'))
    def test_20_nonfinite_job(self):self.assertFalse(admit({'kind':'asr','seconds':float('nan')})[0])
    def test_21_loopback_only(self):
        with self.assertRaises(ValueError):local_url('https://example.com/v1')
        self.assertEqual(local_url('http://127.0.0.1:8080/v1/'),'http://127.0.0.1:8080/v1')
    def test_22_data_leakage_detected(self):
        self.audio(np.zeros(100,np.int16));f=self.p/'data.jsonl';f.write_text('\n'.join(json.dumps(dict(audio='x.wav',text='test',split=s)) for s in ('train','dev')))
        with self.assertRaisesRegex(ValueError,'duplicate'):dataset(f)
    def test_23_chunk_coverage(self):
        f=self.audio(np.zeros(35000,np.int16));out=self.p/'chunks'
        cp=subprocess.run([sys.executable,str(ROOT/'chunk_audio.py'),str(f),'--out',str(out),'--seconds','1'],capture_output=True,text=True);self.assertEqual(cp.returncode,0,cp.stderr)
        r=json.loads((out/'chunks.json').read_text())['chunks'];self.assertEqual([(x['start_sample'],x['end_sample']) for x in r],[(0,16000),(16000,32000),(32000,35000)])
    def test_24_lock_is_immutable_revision(self):
        r=json.loads((ROOT/'models.lock.json').read_text());self.assertEqual(len(r),5)
        for v in r.values():self.assertRegex(v['revision'],r'^[a-f0-9]{40}$')
    def test_25_silence_stops_voice_pipeline(self):
        f=self.audio(np.zeros(16000,np.int16));out=self.p/'voice'
        cp=subprocess.run([sys.executable,str(ROOT/'voice.py'),'--audio',str(f),'--model-name','unused','--out',str(out)],capture_output=True,text=True)
        self.assertNotEqual(cp.returncode,0);self.assertIn('digital silence',cp.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main(verbosity=2)
