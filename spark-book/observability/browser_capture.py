"""Inspect the three simulated GPU states in Prometheus and render each in Grafana."""
import argparse
import json
import time
from pathlib import Path

from client import get
from fixture import write
from trial_stack import require_free_ports, running_stack, series, wait_for


def observed(case):
    source = 'source="fixture"'
    up = series('up{job="gpu"}')
    available = series('lab_gpu_field_available{' + source + ',field="utilization.gpu"}')
    busy = series('lab_gpu_busy_ratio{' + source + '}')
    if len(up)!=1 or float(up[0]['value'][1])!=1 or len(available)!=1:
        return None
    value = float(available[0]['value'][1])
    if case=='missing':
        matches = value==0 and not busy
    else:
        expected = 0.7 if case=='baseline' else 0.0
        matches = value==1 and len(busy)==1 and abs(float(busy[0]['value'][1])-expected)<1e-6
    return {'up':up, 'available':available, 'busy':busy} if matches else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    from playwright.sync_api import sync_playwright
    dest=args.out.resolve()
    require_free_ports()
    dest.mkdir(parents=True,exist_ok=False)
    fixture=dest/'fixture.json'
    write(fixture,'baseline')
    with running_stack(args.tools,dest,['--fixture',str(fixture)]) as owner, sync_playwright() as api:
        wait_for(owner,lambda:json.loads(get('http://127.0.0.1:3000/api/health'))['database']=='ok','Grafana health')
        browser=api.chromium.launch(headless=True,args=['--disable-dev-shm-usage','--disable-gpu'])
        page=browser.new_page(viewport={'width':1600,'height':1200},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto('http://127.0.0.1:3000/login')
        page.locator('input[name="user"]').fill('admin')
        page.locator('input[name="password"]').fill((dest/'stack/grafana-password.txt').read_text().strip())
        page.get_by_role('button',name='Log in',exact=True).click()
        page.wait_for_url(lambda url:'/login' not in url,timeout=30000)
        if page.get_by_text('Skip',exact=True).count():
            page.get_by_text('Skip',exact=True).click()
        for case in ('baseline','missing','zero'):
            write(fixture,case)
            snapshots=[]
            def capture_state():
                state=observed(case)
                if state:snapshots.append(state)
                return state
            wait_for(owner,capture_state,case+' scrape')
            errors.clear()
            page.goto('http://127.0.0.1:3000/d/wsl-gpu-lab/wsl2-gpu-laboratory?orgId=1&from=now-5m&to=now&var-source=fixture&var-gpu=%24__all',wait_until='networkidle')
            page.wait_for_timeout(8000)
            page.screenshot(path=str(dest/(case+'.png')),full_page=True)
            page.mouse.move(1400,950);page.mouse.wheel(0,1100);page.wait_for_timeout(1000)
            page.screenshot(path=str(dest/(case+'-lower.png')),full_page=True)
            result={'case':case,'utc_epoch':time.time(),'browser':browser.version,'source':'fixture','url':page.url,
                    'title':page.title(),'body_text':page.locator('body').inner_text(),'page_errors':errors,'prometheus':snapshots[-1]}
            (dest/(case+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print('Saved '+case+' dashboard and query evidence.',flush=True)
        browser.close()


if __name__=='__main__':
    main()
