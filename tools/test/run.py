import asyncio, json, sys, time, argparse
from playwright.async_api import async_playwright
ap = argparse.ArgumentParser()
ap.add_argument('--browser', default='chromium'); ap.add_argument('--scenes', default='walk3')
ap.add_argument('--params', default=''); ap.add_argument('--tag', default=''); ap.add_argument('--shot', action='store_true')
a = ap.parse_args()
SC = json.load(open('scenes.json')); FAKE = open('fakecam.js').read()
async def run_one(p, name):
    sc = SC[name]
    if a.browser == 'webkit':
        b = await p.webkit.launch(); ctx = await b.new_context(**p.devices['iPhone 13'])
    else:
        b = await p.chromium.launch(args=['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--autoplay-policy=no-user-gesture-required'])
        ctx = await b.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2)
    await ctx.add_init_script('window.__SCENE = ' + json.dumps(sc) + ';')
    await ctx.add_init_script(FAKE)
    pg = await ctx.new_page(); logs = []
    pg.on('console', lambda m: logs.append(m.type + ': ' + m.text) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: logs.append('pageerror: ' + str(e)))
    await pg.goto('http://127.0.0.1:8080/crowd/index.html?' + a.params)
    await pg.click('#startBtn')
    t0 = time.time(); started = None; samples = []
    while time.time() - t0 < sc['dur'] * 3 + 120:
        await asyncio.sleep(1)
        s = await pg.evaluate("""()=>{const t=window.__tc; return {run:t.running, ft: window.__fake.time(), u:t.ledger.unique, pd:t.ledger.possible, uv:t.ledger.unverified, inF:t.st.inFrame, fps:t.st.fps, det:t.detector&&t.detector.name, ep:t.detector&&t.detector.ep, size:t.st.size, tiles:t.st.tiles, light:t.st.light, dense:t.dense.active, est:t.dense.est, src:t.dense.source, lens:t.cam.lensKind, frames:t.st.frames, status:document.getElementById('status').textContent}}""")
        samples.append(s)
        if not s['run'] and time.time() - t0 > 90: break
        if s['run'] and started is None: started = time.time()
        if a.shot and abs(s['ft'] - sc['dur'] * 0.5) < 0.6: await pg.screenshot(path=f'shots/{name}{a.tag}.png')
        if s['ft'] >= sc['dur']: break
    st = await pg.evaluate("""()=>{const t=window.__tc; return {events:t.ledger.events.map(e=>[e.type,e.id||e.from,e.into||e.candidate||'',e.sim!=null?+(+e.sim).toFixed(3):(e.candSim?+(+e.candSim).toFixed(3):''),e.why||'']), bench:t.detector&&t.detector.bench, loadErr:t.detector&&t.detector.loadErrors, reid:t.reid&&(t.reid.name+'/'+t.reid.ep), timings:t.st.timings, peak:t.dense.peak, denseMs:t.dense.ms, denseEp:t.dense.ep, lensSw:window.__fake.lensSwitches||0, errs:window.__scriptErrors, banners:[...document.querySelectorAll('.banner')].map(b=>b.dataset.key), csv:t.csvText().split('\\n').length-2, pairs:(()=>{const L=t.ledger, P=L.live(), o=[]; for(const a of P) for(const b of P) if(a.id<b.id) o.push([a.id,b.id,+L.gsim(a,b).toFixed(2),+L.gsim(b,a).toFixed(2),L.coCount(a,b),a.gallery.length,b.gallery.length]); return o})()}}""")
    await pg.click('#stopBtn'); final = await pg.evaluate("()=>[document.getElementById('final-total').textContent, document.getElementById('final-detail').innerText]")
    fpsv = [s['fps'] for s in samples if s['frames'] > 3]
    last = samples[-1]
    res = dict(scene=name, browser=a.browser, params=a.params, gt=sc['gt'], unique=last['u'], possible=last['pd'], unverified=last['uv'], det=last['det'], ep=last['ep'], size=last['size'],
               fps=round(sum(fpsv) / max(1, len(fpsv)), 2), frames=last['frames'], tiles_any=any(s['tiles'] for s in samples), dark_any=any(s['light'] != 'ok' for s in samples), dense_any=any(s['dense'] for s in samples),
               est_last=last['est'], est_src=last['src'], lens_final=last['lens'], final=final, maxInFrame=max(s['inF'] for s in samples), **st, logs=logs[-8:])
    await b.close(); return res
async def main():
    async with async_playwright() as p:
        for name in a.scenes.split(','):
            r = await run_one(p, name)
            print(json.dumps(r), flush=True)
            with open('results.jsonl', 'a') as f: f.write(json.dumps(r) + '\n')
asyncio.run(main())
