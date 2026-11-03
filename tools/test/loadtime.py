"""First-launch vs second-launch timing for crowd/, plus offline and dense-model checks.

Serves the repo through a small throttled static server and drives headless Chrome.
Bytes are counted on the socket, so a cache hit shows up as zero bytes for that file.

  python3 tools/test/loadtime.py
"""
import asyncio, json, os, subprocess, sys, time
from playwright.async_api import async_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PORT = int(os.environ.get('PORT', '8093'))
LOG = os.environ.get('LOAD_LOG', '/tmp/tcount-loadtime.jsonl')
CHROME = os.environ.get('CHROME', '/usr/bin/google-chrome')
FAKE = open(os.path.join(os.path.dirname(__file__), 'fakecam.js')).read()
SCENE = json.dumps({'W': 640, 'H': 360, 'dur': 8, 'people': [], 'brightness': 1, 'preload': []})
WASM = 'ort-wasm-simd-threaded.asyncify.wasm'
DENSE = 'dmcount_qnrf_w8.onnx'
NANO = 'yolox_nano_person_w8.onnx'
TINY = 'yolox_tiny_person_w8.onnx'


def reset_log():
    open(LOG, 'w').close()


def read_log(since=0):
    rows = []
    if not os.path.exists(LOG):
        return rows
    with open(LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec['t'] >= since:
                rows.append(rec)
    return rows


def summarize(rows):
    by = {}
    for rec in rows:
        name = rec['path'].rsplit('/', 1)[-1] or rec['path']
        by[name] = by.get(name, 0) + rec['bytes']
    return by


async def until_counting(page, timeout=180):
    t_click = time.time()
    await page.click('#startBtn')
    deadline = time.time() + timeout
    status = ''
    while time.time() < deadline:
        status = (await page.text_content('#status')) or ''
        if 'Counting' in status or 'failed' in status.lower() or status.startswith('Model load') or status.startswith('Camera error') or status.startswith('Detection error'):
            break
        await asyncio.sleep(0.15)
    info = await page.evaluate("""() => ({
      status: document.getElementById('status').textContent,
      footer: (document.getElementById('build-footer') || {}).innerText || '',
      assetFrom: window.__tc && window.__tc.assetFrom,
      diag: window.__tc && window.__tc.loadDiag,
      version: window.__tc && window.__tc.VERSION,
      errors: window.__scriptErrors || [],
      sw: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
      reid: !!(window.__tc && window.__tc.reid),
      running: !!(window.__tc && window.__tc.running)
    })""")
    info['click_to_status_s'] = round(time.time() - t_click, 3)
    info['status'] = status
    return info


async def scenario(page, name, url, settle_ms=0, wait_prefetch=False):
    t0 = time.time()
    await page.goto(url, wait_until='load', timeout=60000)
    if wait_prefetch:
        await page.wait_for_function(
            "() => window.__tc && window.__tc.loadDiag && window.__tc.loadDiag.some(l => l.includes('prefetch runtime'))",
            timeout=120000)
    elif settle_ms:
        await page.wait_for_timeout(settle_ms)
    result = await until_counting(page)
    await page.wait_for_timeout(800)
    files = summarize(read_log(t0))
    out = {'scenario': name, 'click_to_status_s': result['click_to_status_s'], 'status': result['status'],
           'footer': result['footer'], 'version': result['version'], 'sw': result['sw'], 'reid_at_count': result['reid'],
           'errors': result['errors'], 'diag': result['diag'], 'assetFrom': result['assetFrom'], 'server_bytes': files,
           'wasm_bytes': files.get(WASM, 0), 'dense_bytes': files.get(DENSE, 0), 'nano_bytes': files.get(NANO, 0),
           'tiny_bytes': files.get(TINY, 0)}
    print(json.dumps({k: out[k] for k in out if k not in ('diag',)}), flush=True)
    if result['diag']:
        print('  diag:', ' | '.join(result['diag'][-8:]), flush=True)
    return out


async def main():
    reset_log()
    srv = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'byteserve.py'), str(PORT), LOG, ROOT])
    await asyncio.sleep(0.3)
    base = f'http://127.0.0.1:{PORT}/crowd/index.html'
    nano = base + '?backend=wasm&model=nano&dense=off'
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(executable_path=CHROME, args=['--headless=new', '--disable-gpu'])
            results = []

            async def fresh():
                ctx = await browser.new_context(viewport={'width': 390, 'height': 844}, service_workers='allow')
                await ctx.add_init_script('window.__SCENE = ' + SCENE + ';')
                await ctx.add_init_script(FAKE)
                page = await ctx.new_page()
                page.on('pageerror', lambda e: print('PAGEERROR', e, flush=True))
                cdp = await ctx.new_cdp_session(page)
                try:
                    await cdp.send('Storage.clearDataForOrigin', {'origin': f'http://127.0.0.1:{PORT}', 'storageTypes': 'all'})
                except Exception as e:
                    print('clear storage', e, flush=True)
                return ctx, page

            def serve(mbps):
                return subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'byteserve.py'), str(PORT), LOG, ROOT, str(mbps)])

            # 2 MB/s per connection, the same limit used for the before measurement.
            # The process started above is replaced so the timed runs see the throttle.
            srv.terminate(); srv.wait(timeout=5)
            srv = serve('2.0')
            await asyncio.sleep(0.4)
            ctx, page = await fresh()
            reset_log()
            results.append(await scenario(page, 'after-first-throttled', nano))
            reset_log()
            results.append(await scenario(page, 'after-second-throttled', nano))
            await ctx.close()
            ctx, page = await fresh()
            reset_log()
            results.append(await scenario(page, 'after-throttled-prefetch-then-click', nano, wait_prefetch=True))
            await ctx.close()
            srv.terminate(); srv.wait(timeout=5)
            srv = serve('0')
            await asyncio.sleep(0.4)
            ctx, page = await fresh()
            reset_log()
            warm = await scenario(page, 'after-prefetch-then-click', nano, wait_prefetch=True)
            results.append(warm)
            reset_log()
            second = await scenario(page, 'after-second-unthrottled', nano)
            results.append(second)
            await page.wait_for_function("() => navigator.serviceWorker && navigator.serviceWorker.controller", timeout=15000)
            await page.context.set_offline(True)
            reset_log()
            try:
                offline = await scenario(page, 'after-offline-relaunch', nano)
            except Exception as e:
                offline = {'scenario': 'after-offline-relaunch', 'error': str(e)}
                print(json.dumps(offline), flush=True)
            results.append(offline)
            await ctx.close()

            ctx, page = await fresh()
            reset_log()
            off = await scenario(page, 'dense-off', base + '?backend=wasm&model=nano&dense=off')
            await page.wait_for_timeout(2000)
            off_files = summarize(read_log())
            off['dense_bytes_after_wait'] = off_files.get(DENSE, 0)
            results.append(off)
            print(json.dumps({'scenario': 'dense-off-settled', 'dense_bytes': off['dense_bytes_after_wait'], 'tiny_bytes': off_files.get(TINY, 0)}), flush=True)
            await ctx.close()

            ctx, page = await fresh()
            reset_log()
            # Auto + wasm must not pull Tiny.
            auto = await scenario(page, 'auto-cpu-nano', base + '?backend=wasm&dense=off')
            results.append(auto)
            await ctx.close()

            ctx, page = await fresh()
            reset_log()
            t0 = time.time()
            await page.goto(base + '?backend=wasm&model=nano&dense=on&denseNow=1', wait_until='load')
            await page.click('#startBtn')
            deadline = time.time() + 90
            dense_bytes = 0
            while time.time() < deadline:
                dense_bytes = summarize(read_log(t0)).get(DENSE, 0)
                got = await page.evaluate("() => !!(window.__tc && window.__tc.assetFrom && window.__tc.assetFrom['models/dmcount_qnrf_w8.onnx'])")
                if dense_bytes >= 21560628 or got:
                    break
                await asyncio.sleep(0.25)
            on = {'scenario': 'dense-on-downloads', 'dense_bytes': dense_bytes, 'status': await page.text_content('#status'),
                  'asset': await page.evaluate("() => window.__tc && window.__tc.assetFrom && window.__tc.assetFrom['models/dmcount_qnrf_w8.onnx']")}
            print(json.dumps(on), flush=True)
            results.append(on)
            await ctx.close()
            await browser.close()
            print(json.dumps({'summary': [{k: r.get(k) for k in ('scenario', 'click_to_status_s', 'status', 'wasm_bytes', 'dense_bytes', 'tiny_bytes', 'nano_bytes', 'error')} for r in results]}, indent=2), flush=True)
    finally:
        if srv.poll() is None:
            srv.terminate()


if __name__ == '__main__':
    asyncio.run(main())
