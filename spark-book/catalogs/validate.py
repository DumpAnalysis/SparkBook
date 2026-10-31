"""Offline regression controls for artifact identity and retained observations."""
import copy,json,subprocess,tempfile,unittest
from pathlib import Path
from common import file_inventory,verify_inventory,safe_name,sha
from ngc_lab import select_arm,verify_digest,RegistryRedirect
from urllib.request import Request

ROOT=Path(__file__).parent
class CatalogControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'a').write_bytes(b'abc')
        self.files=file_inventory(self.root,['a'])
        self.index=json.loads((ROOT/'controls/ngc-image/index.json').read_text())
    def test_01_native_manifest(self):
        entry=select_arm(self.index)
        verify_digest((ROOT/'controls/ngc-image/manifest.json').read_bytes(),entry['digest'])
    def test_02_missing_architecture(self):
        with self.assertRaises(ValueError):select_arm({'manifests':[]})
    def test_03_ambiguous_architecture(self):
        entry=select_arm(self.index)
        with self.assertRaises(ValueError):select_arm({'manifests':[entry,entry]})
    def test_04_changed_manifest_bytes(self):
        raw=(ROOT/'controls/ngc-image/manifest.json').read_bytes()
        with self.assertRaises(ValueError):verify_digest(raw+b' ',select_arm(self.index)['digest'])
    def test_05_unsafe_filenames(self):
        for name in ['../secret','/absolute','C:drive','a\\b','']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_name(name)
    def test_06_inventory_exact(self):self.assertEqual(verify_inventory(self.root,self.files),1)
    def test_07_inventory_tamper(self):
        (self.root/'a').write_bytes(b'xyz')
        with self.assertRaises(ValueError):verify_inventory(self.root,self.files)
    def test_08_inventory_missing(self):
        (self.root/'a').unlink()
        with self.assertRaises(ValueError):verify_inventory(self.root,self.files)
    def test_09_inventory_duplicate(self):
        with self.assertRaises(ValueError):verify_inventory(self.root,self.files*2)
    def test_10_symlink_escape(self):
        with tempfile.TemporaryDirectory() as other:
            outside=Path(other)/'secret';outside.write_text('fixture')
            (self.root/'link').symlink_to(outside)
            with self.assertRaises(ValueError):file_inventory(self.root,['link'])
    def test_11_cross_host_token_stripped(self):
        request=Request('https://nvcr.io/blob',headers={'Authorization':'Bearer fixture'})
        result=RegistryRedirect().redirect_request(request,None,302,'Found',{},'https://example.org/blob')
        self.assertIsNone(result.get_header('Authorization'))
    def test_12_container_rejects_mutable_tag(self):
        record=self.root/'image.json'
        record.write_text(json.dumps({'architecture':'arm64','os':'linux','pinned_image':'nvcr.io/nvidia/pytorch:26.09-py3'}))
        p=subprocess.run(['bash',str(ROOT/'run_ngc_probe.sh'),str(record)],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('invalid pinned',p.stderr)
    def test_13_executed_offline_control(self):
        r=json.loads((ROOT/'controls/hf-offline/result.json').read_text())
        self.assertEqual(r['device'],'cpu');self.assertTrue(r['finite'])
        self.assertEqual(r['hidden_shape'],[2,35,32]);self.assertEqual(r['repeat_max_abs_error'],0)
        self.assertEqual(r['python_socket_connect_attempts'],0);self.assertEqual(r['missing_keys'],[])
    def test_14_executed_dataset_fault(self):
        r=json.loads((ROOT/'controls/hf-audit/audit.json').read_text())
        self.assertEqual((r['train_rows'],r['test_rows'],r['id_overlap'],r['normalized_text_overlap'],r['injected_overlap_detected']),(5,5,0,0,1))
    def test_15_no_gpu_success_claim(self):
        r=json.loads((ROOT/'controls/10-gpu-unavailable/command.json').read_text())
        self.assertEqual(r['exit'],2)
        self.assertIn('"status": "UNAVAILABLE"',(ROOT/'controls/10-gpu-unavailable/transcript.txt').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
