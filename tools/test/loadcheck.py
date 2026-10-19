import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        for name in ['chromium','webkit']:
            b = await getattr(p,name).launch()
            ctx = await b.new_context(**(p.devices['iPhone 13'] if name=='webkit' else {}))
            pg = await ctx.new_page(); errs=[]
            pg.on('console', lambda m: errs.append(m.type+': '+m.text) if m.type in ('error','warning') else None)
            pg.on('pageerror', lambda e: errs.append('pageerror: '+str(e)))
            pg.on('requestfailed', lambda r: errs.append('reqfail: '+r.url))
            await pg.goto('http://127.0.0.1:8080/crowd/', wait_until='load'); await pg.wait_for_timeout(4000)
            st = await pg.text_content('#status') if await pg.query_selector('#status') else None
            print(name, 'status=',st, 'issues=',errs)
            await b.close()
asyncio.run(main())
