import os
import re
import time
import asyncio
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import argparse
import logging
import pathlib

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# クローラークラスの定義
class PlaywrightCrawler:
    def __init__(self, base_url, max_pages=100):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls = set()
        self.content_data = []
        
    def is_valid_url(self, url):
        """URLが有効かつクロール対象かを確認"""
        parsed_url = urlparse(url)
        base_parsed = urlparse(self.base_url)
        
        # 同じドメイン内のURLかつ、パスが/hc/jaで始まるURLのみ許可
        return (parsed_url.netloc == base_parsed.netloc and 
                parsed_url.path.startswith('/hc/ja'))
    
    async def extract_page_info(self, page, url):
        """Playwrightのページからテキストとメタデータを抽出"""
        # Cloudflareが完了するまで待機
        await self.wait_for_cloudflare(page)
        
        # ページのHTMLを取得
        html_content = await page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # メインコンテンツを抽出
        main_content = soup.select('.article-body') or soup.select('main') or soup.select('.main-content')
        
        if main_content:
            # 主要コンテンツのテキストを抽出
            text = main_content[0].get_text(separator="\n")
        else:
            # 主要コンテンツが見つからない場合は全体のテキストを取得
            text = soup.get_text(separator="\n")
            
        # 余分な空白と改行を整理
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        
        # メタデータの抽出
        title = await page.title()
        
        # h1タグから見出しを取得
        h1 = soup.find('h1')
        heading = h1.get_text().strip() if h1 else title
        
        # メタディスクリプションがあれば取得
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else ""
        
        # リンクを抽出
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        
        return {
            "url": url,
            "title": title,
            "heading": heading,
            "description": description,
            "content": text,
            "links": links
        }
    
    async def wait_for_cloudflare(self, page):
        """Cloudflareのチェックが完了するまで待機"""
        # タイムアウト設定（秒）
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            title = await page.title()
            if title and "Just a moment..." not in title:
                # Cloudflareのチェックが完了
                # 追加の待機時間を設けて完全なレンダリングを確保
                await page.wait_for_load_state('networkidle')
                return
            
            # 少し待機してから再試行
            await asyncio.sleep(0.5)
        
        # タイムアウトした場合でもページの読み込みを最大限待機
        await page.wait_for_load_state('networkidle')
        logger.warning(f"Cloudflareチェックの待機がタイムアウトしました: {await page.url()}")
    
    async def crawl(self):
        """Playwrightを使用してクローリングを実行"""
        async with async_playwright() as p:
            # ブラウザを起動（ヘッドレスモード）
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            # 初期ページをキューに追加
            queue = [self.base_url]
            
            try:
                while queue and len(self.visited_urls) < self.max_pages:
                    url = queue.pop(0)
                    
                    if url in self.visited_urls:
                        continue
                    
                    logger.info(f"クローリング中: {url}")
                    
                    try:
                        # ページを開く
                        page = await context.new_page()
                        await page.goto(url, wait_until="domcontentloaded")
                        
                        # ページ情報を抽出
                        page_info = await self.extract_page_info(page, url)
                        
                        # 訪問済みURLに追加
                        self.visited_urls.add(url)
                        
                        # データを保存（リンクを除外）
                        data = {k: v for k, v in page_info.items() if k != "links"}
                        self.content_data.append(data)
                        
                        # 新しいリンクをキューに追加
                        for link_url in page_info["links"]:
                            if link_url not in self.visited_urls:
                                queue.append(link_url)
                        
                        # ページを閉じる
                        await page.close()
                        
                        # サーバーに負荷をかけないよう待機
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"ページ処理中のエラー {url}: {str(e)}")
                        try:
                            await page.close()
                        except:
                            pass
            
            finally:
                # ブラウザを閉じる
                await context.close()
                await browser.close()
            
            logger.info(f"クローリング完了。{len(self.content_data)}ページを取得しました。")
            return self.content_data

async def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Crawl website')
    parser.add_argument('--base_url', default='https://support.iriam.com/hc/ja', help='クローリングするベースURL')
    parser.add_argument('--max_pages', type=int, default=100, help='クローリングする最大ページ数')
    args = parser.parse_args()
    
    # クローリング実行
    crawler = PlaywrightCrawler(base_url=args.base_url, max_pages=args.max_pages)
    crawled_data = await crawler.crawl()
    
    # データ保存
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(crawled_data)
    df.to_csv("data/crawled_data.csv", index=False)
    print(f"クロール済みデータを保存しました。{len(crawled_data)}件のドキュメントを取得。")

if __name__ == "__main__":
    asyncio.run(main())
