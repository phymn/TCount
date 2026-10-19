import asyncio, json
from playwright.async_api import async_playwright
SC = json.load(open('scenes.json')); FAKE = open('fakecam.js').read(); IP = open('/tmp/ip').read().split()[0]
async def page(p, url, route=None, wait=25, click=True, browser='chromium'):
    b = await (p.webkit.launch() if browser == 'webkit' else p.chromium.launch())
    ctx = await b.new_context(viewport={'width': 390, 'height': 844})
    await ctx.add_init_script('window.__SCENE = ' + json.dumps(SC['walk3']) + ';'); await ctx.add_init_script(FAKE)
    if route: await ctx.route(route[0], route[1])
    pg = await ctx.new_page(); await pg.goto(url)
    if click: await pg.click('#startBtn'); await asyncio.sleep(wait)
    r = await pg.evaluate("()=>({status:document.getElementById('status').textContent, banners:[...document.querySelectorAll('.banner')].map(b=>b.dataset.key+': '+b.dataset.msg), det: window.__tc && window.__tc.detector && (window.__tc.detector.name+'/'+window.__tc.detector.ep), u: window.__tc && window.__tc.ledger.unique, coi: crossOriginIsolated, thr: typeof ort!=='undefined' && ort.env.wasm.numThreads, fps: window.__tc && window.__tc.st.fps})")
    await b.close(); return r
async def main():
    async with async_playwright() as p:
        print('INSECURE (LAN IP over http):', await page(p, f'http://{IP}:8081/crowd/index.html', click=False))
        print('ORT SCRIPT BLOCKED:', await page(p, 'http://127.0.0.1:8080/crowd/index.html', ('**/vendor/ort.webgpu.min.js', lambda r: r.abort()), wait=40))
        print('MODELS 404:', await page(p, 'http://127.0.0.1:8080/crowd/index.html?backend=wasm', ('**/models/yolox_*', lambda r: r.fulfill(status=404, body='nf')), wait=40))
        print('REID 404:', await page(p, 'http://127.0.0.1:8080/crowd/index.html?backend=wasm', ('**/models/osnet_*', lambda r: r.fulfill(status=404, body='nf')), wait=20))
        print('WEBKIT ORT BLOCKED:', await page(p, 'http://127.0.0.1:8080/crowd/index.html', ('**/vendor/ort.webgpu.min.js', lambda r: r.abort()), wait=40, browser='webkit'))
asyncio.run(main())
