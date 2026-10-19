import asyncio, json
from playwright.async_api import async_playwright
SC = json.load(open('scenes.json')); FAKE = open('fakecam.js').read()
async def main():
    async with async_playwright() as p:
        b = await p.webkit.launch(); ctx = await b.new_context(**p.devices['iPhone 13'])
        await ctx.add_init_script('window.__SCENE = ' + json.dumps(SC['empty']) + ';'); await ctx.add_init_script(FAKE)
        pg = await ctx.new_page(); await pg.goto('http://127.0.0.1:8080/crowd/index.html')
        r = await pg.evaluate("async()=>{const d=await navigator.mediaDevices.enumerateDevices(); return d.map(x=>x.label)}")
        print(r)
        await pg.click('#startBtn'); await asyncio.sleep(8)
        print(await pg.evaluate("()=>({devs: window.__tc.cam.devices, kind: window.__tc.cam.lensKind, label: window.__tc.cam.label, facing: window.__tc.cam.facing, S: window.__tc.S.lensMode})"))
        await b.close()
asyncio.run(main())
