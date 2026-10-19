# usage: modeltest.py browser port [params] [maxwait]
import asyncio, json, sys, time
from playwright.async_api import async_playwright
br, port = sys.argv[1], sys.argv[2]; params = sys.argv[3] if len(sys.argv) > 3 else ''; maxw = int(sys.argv[4]) if len(sys.argv) > 4 else 90
SC = json.load(open('scenes.json')); FAKE = open('fakecam.js').read(); sc = SC['walk3']
async def main():
    async with async_playwright() as p:
        if br == 'webkit': b = await p.webkit.launch(); ctx = await b.new_context(**p.devices['iPhone 13'])
        else: b = await p.chromium.launch(); ctx = await b.new_context(viewport={'width':390,'height':844})
        await ctx.add_init_script('window.__SCENE = ' + json.dumps(sc) + ';'); await ctx.add_init_script(FAKE)
        pg = await ctx.new_page(); logs = []
        pg.on('console', lambda m: logs.append(m.type + ': ' + m.text[:300]) if m.type in ('error', 'warning') else None)
        pg.on('requestfailed', lambda r: logs.append('reqfail ' + r.url))
        t0 = time.time()
        await pg.goto(f'http://127.0.0.1:{port}/crowd/index.html?' + params); await pg.click('#startBtn')
        while time.time() - t0 < maxw:
            await asyncio.sleep(1)
            s = await pg.evaluate("()=>({st:document.getElementById('status').textContent, det:window.__tc&&window.__tc.detector&&(window.__tc.detector.name+'/'+window.__tc.detector.ep), u:window.__tc&&window.__tc.ledger.unique, fr:window.__tc&&window.__tc.st.frames, luma:window.__tc&&window.__tc.st.lumaEma, light:window.__tc&&window.__tc.st.light})")
            if s['fr'] and s['fr'] > 30: break
        s['banners'] = await pg.evaluate("()=>[...document.querySelectorAll('.banner')].map(b=>b.dataset.key+': '+b.textContent.slice(0,240))")
        s['t'] = round(time.time() - t0, 1); s['logs'] = logs[-6:]
        print(json.dumps(s, indent=1)); await b.close()
asyncio.run(main())
