import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        for name, launch in [('chromium-default', dict(args=['--enable-unsafe-webgpu'])), ('chromium-swiftshader', dict(args=['--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=swiftshader','--use-webgpu-adapter=swiftshader'])), ('chromium-new-headless', dict(channel='chromium', args=['--enable-unsafe-webgpu','--enable-features=Vulkan','--use-webgpu-adapter=swiftshader']))]:
            try:
                b = await p.chromium.launch(**launch); pg = await b.new_page(); await pg.goto('http://127.0.0.1:8080/crowd/index.html')
                r = await pg.evaluate("""async()=>{ if(!navigator.gpu) return 'no navigator.gpu'; const a=await navigator.gpu.requestAdapter(); if(!a) return 'no adapter'; const i=a.info||{}; return 'adapter '+(i.vendor||'')+' '+(i.architecture||'')+' '+(i.description||'')+' fallback='+a.isFallbackAdapter }""")
                print(name, r); await b.close()
            except Exception as e: print(name, 'ERR', str(e)[:200])
        b = await p.webkit.launch(); ctx = await b.new_context(**p.devices['iPhone 13']); pg = await ctx.new_page(); await pg.goto('http://127.0.0.1:8080/crowd/index.html')
        r = await pg.evaluate("""async()=>({gpu: !!navigator.gpu, ua: navigator.userAgent, secure: isSecureContext, cs: !!HTMLCanvasElement.prototype.captureStream, wasm: typeof WebAssembly})""")
        print('webkit', r); await b.close()
asyncio.run(main())
