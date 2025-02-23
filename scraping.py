import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import time

async def scrape():
    url = "https://support.iriam.com/hc/ja"

    async with async_playwright() as p:
        # ブラウザの設定を強化
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process',
            ]
        )
        
        # ブラウザコンテキストの設定
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            java_script_enabled=True,
        )
        
        page = await context.new_page()
        await stealth_async(page)

        # JavaScript の設定
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
        """)

        try:
            # ページにアクセス
            response = await page.goto(url, wait_until="networkidle")
            
            # Cloudflare チェックが終わるまで待機
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(5)  # 追加の待機時間
            
            # "Just a moment..." が表示されている場合の追加待機
            max_retries = 3
            current_retry = 0
            
            while current_retry < max_retries:
                title = await page.title()
                if title != "Just a moment...":
                    break
                print(f"待機中... 試行回数: {current_retry + 1}")
                await asyncio.sleep(5)
                current_retry += 1

            # コンテンツの取得
            title = await page.title()
            print(f"ページタイトル: {title}")
            
            # メインコンテンツが読み込まれるまで待機
            await page.wait_for_selector('.article-list', timeout=30000)
            
            content = await page.content()
            print(content)
            print(f"ページのHTMLの長さ: {len(content)} 文字")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
        
        finally:
            await context.close()
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape())