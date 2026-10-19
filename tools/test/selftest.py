import asyncio, json, sys
from playwright.async_api import async_playwright
br, port, mode = sys.argv[1], sys.argv[2], sys.argv[3]
SC = json.load(open('scenes.json')); FAKE = open('fakecam.js').read(); sc = dict(SC['walk3'])
if mode == 'black': sc['brightness'] = 0.0; sc['noise'] = 0
async def main():
    async with async_playwright() as p:
        if br == 'webkit': b = await p.webkit.launch(); ctx = await b.new_context(**p.devices['iPhone 13'])
        else: b = await p.chromium.launch(); ctx = await b.new_context(viewport={'width':390,'height':844})
        await ctx.add_init_script('window.__SCENE = ' + json.dumps(sc) + ';'); await ctx.add_init_script(FAKE)
        pg = await ctx.new_page()
        if mode == 'missing': await pg.route('**/models/yolox_*', lambda r: r.fulfill(status=404, body='nf'))
        if mode == 'trunc': await pg.route('**/models/yolox_nano*', lambda r: r.fulfill(status=200, body=b'x'*1000, content_type='application/octet-stream'))
        if mode == 'wasm404': await pg.route('**/*.wasm', lambda r: r.fulfill(status=404, body='nf'))
        await pg.goto(f'http://127.0.0.1:{port}/crowd/index.html' + ('?model=nano' if mode == 'trunc' else ''))
        if mode in ('missing', 'trunc', 'wasm404', 'black', 'run'): await pg.click('#startBtn'); await asyncio.sleep(14 if mode != 'black' else 22)
        print('banners:', json.dumps(await pg.evaluate("()=>[...document.querySelectorAll('.banner')].map(b=>b.textContent.slice(0,400))"), indent=1))
        print('status:', await pg.text_content('#status'))
        await pg.click('#selfTestBtn'); await asyncio.sleep(1)
        for i in range(40):
            await asyncio.sleep(1); t = await pg.text_content('#selfTestText')
            if 'runtime test: OK' in t or 'FAILED' in t.split('runtime test')[-1] or 'missing or incomplete' in t: break
        print(t)
        if len(sys.argv) > 4: await pg.screenshot(path=sys.argv[4], full_page=False)
        await b.close()
asyncio.run(main())
