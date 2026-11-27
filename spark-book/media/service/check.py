"""Check admission, synthesis and the single-owner busy response on the lab's local MMS service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8022'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def send(key, text, ttl=40, barrier=None):
    if barrier:
        barrier.wait(timeout=10)
    body = json.dumps({'id':key,'text':text,'deadline':time.time()+ttl}).encode()
    request = urllib.request.Request(BASE+'/tts', data=body, headers={'Content-Type':'application/json'})
    try:
        response = opener.open(request, timeout=65)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return {'id':key, 'http':response.status, 'body':json.load(response)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    with args.out.open('x') as receipt:
        end = time.monotonic()+120
        while True:
            try:
                with opener.open(BASE+'/health', timeout=3) as response:
                    identity = json.load(response)
                break
            except OSError:
                if time.monotonic() >= end:
                    raise SystemExit('MMS readiness timed out; inspect container logs')
                time.sleep(1)
        responses = [send('accepted','The processor is ready.'),send('overlength','x'*257),send('expired','Ready',-2)]
        prompt = 'The processor is ready. '*8
        gate = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses += list(pool.map(lambda i:send('parallel'+str(i),prompt,barrier=gate),range(3)))
        json.dump({'identity':identity,'responses':responses},receipt,indent=2)
        statuses=[r['http'] for r in responses]
        print('HTTP statuses', statuses)
        assert identity['model_device']=='cuda:0' and statuses[:3]==[200,422,408] and sorted(statuses[3:])==[200,429,429]
        assert all(r['body']['finite'] and r['body']['frames']>0 for r in responses if r['http']==200)
