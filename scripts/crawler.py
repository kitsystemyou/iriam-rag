import os
import asyncio
import argparse
import pandas as pd
from playwright_crawler import PlaywrightCrawler  # 元コードからクラスを抽出


async def main():
    parser = argparse.ArgumentParser(description='Crawl website')
    parser.add_argument('--base_url', default='https://support.iriam.com/hc/ja')
    parser.add_argument('--max_pages', type=int, default=100)
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
    # dbg
    # os.environ["OPENAI_API_KEY"] = ""
    asyncio.run(main())
