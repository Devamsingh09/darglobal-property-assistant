"""One-off probe: does each site render in Playwright, and which JSON APIs does it call?"""
import asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent.parent / "data" / "probe"
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "dg_projects": "https://darglobal.co.uk/projects",
    "dg_astera": "https://darglobal.co.uk/the-astera",
    "wasalt_listing": "https://wasalt.sa/en/property/sale/apartment-with-3-bedrooms-5841667",
}

async def main(headless: bool):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
            viewport={"width": 1366, "height": 900}, locale="en-US")
        page = await ctx.new_page()
        for name, url in URLS.items():
            apis = []
            def on_resp(r):
                ct = r.headers.get("content-type", "")
                if "json" in ct:
                    apis.append(f"{r.status} {r.url[:200]}")
            page.on("response", on_resp)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(8000)
            except Exception as e:
                print(name, "ERROR", e)
            html = await page.content()
            title = await page.title()
            (OUT / f"{name}.html").write_text(html, encoding="utf-8")
            text = await page.evaluate("document.body ? document.body.innerText : ''")
            print(f"=== {name}  title={title!r} html={len(html)} text={len(text)} next_data={'__NEXT_DATA__' in html}")
            print("   text head:", text[:300].replace("\n", " | "))
            for a in apis[:25]:
                print("   API", a)
            page.remove_listener("response", on_resp)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main(headless="--headed" not in sys.argv))
