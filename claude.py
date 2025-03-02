import os
import re
import time
import asyncio
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import pathlib

# 1. ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. 環境変数設定（APIキーなど）
os.environ["OPENAI_API_KEY"] = ""  # APIキーを設定してください

# 3. Playwrightを使用したクローリング関数
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

# 4. テキスト分割とベクトル化
class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_documents(self, documents):
        """ドキュメントを処理してチャンクに分割"""
        docs_with_metadata = []
        
        for doc in documents:
            text = doc["content"]
            metadata = {k: v for k, v in doc.items() if k != "content"}
            
            # テキストを分割
            chunks = self.text_splitter.split_text(text)
            
            # メタデータを各チャンクに追加
            for i, chunk in enumerate(chunks):
                docs_with_metadata.append({
                    "content": chunk,
                    "chunk_id": i,
                    **metadata
                })
                
        return docs_with_metadata
        
    def create_vector_index(self, processed_docs):
        """ドキュメントをベクトル化してインデックスを作成"""
        embeddings = OpenAIEmbeddings()
        
        texts = [doc["content"] for doc in processed_docs]
        metadatas = [{k: v for k, v in doc.items() if k != "content"} for doc in processed_docs]
        
        # FAISSベクトルストアの作成
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        
        return vector_store

# 5. RAGシステム
class RAGSystem:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo", temperature=0):
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        
        # レトリバルシステムの作成
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # プロンプトテンプレートの作成
        template = """
        あなたはIriamサポートの専門アシスタントです。
        以下の情報を参考にして、ユーザーの質問に丁寧に回答してください。
        
        関連コンテンツ:
        {context}
        
        質問: {question}
        
        回答:
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # LLMの初期化
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        
        # QAチェーンの作成
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
    def answer_question(self, question):
        """質問に回答する"""
        response = self.qa_chain({"query": question})
        
        # 回答とソースドキュメントを返す
        answer = response["result"]
        source_docs = response["source_documents"]
        
        # ソースURLを重複なく取得
        source_urls = []
        for doc in source_docs:
            if "url" in doc.metadata and doc.metadata["url"] not in source_urls:
                source_urls.append(doc.metadata["url"])
        
        return {
            "answer": answer,
            "sources": source_urls
        }

# 6. 既存インデックスのロードまたは新規作成
async def load_or_build_rag_system(base_url="https://support.iriam.com/hc/ja", max_pages=100, index_path="faiss_index"):
    """既存インデックスをロードするか、なければ新規に構築する"""
    
    index_path_obj = pathlib.Path(index_path)
    embeddings = OpenAIEmbeddings()
    
    # インデックスが存在するかチェック
    if index_path_obj.exists() and any(index_path_obj.iterdir()):
        logger.info(f"既存のインデックスが見つかりました: {index_path}")
        # 既存のインデックスをロード
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("インデックスのロードが完了しました。")
    else:
        logger.info(f"インデックスが見つからないため、新規構築を開始します。")
        # 1. クローリング
        logger.info("クローリングを開始します...")
        crawler = PlaywrightCrawler(base_url=base_url, max_pages=max_pages)
        crawled_data = await crawler.crawl()

        # データフレームに変換して保存（オプション）
        df = pd.DataFrame(crawled_data)
        df.to_csv("crawled_data.csv", index=False)
        logger.info(f"クロール済みデータを保存しました。{len(crawled_data)}件のドキュメントを取得。")

        # 2. テキスト処理
        logger.info("テキスト処理を開始します...")
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        processed_docs = processor.process_documents(crawled_data)
        logger.info(f"テキスト処理完了。{len(processed_docs)}件のチャンクに分割されました。")

        # 3. ベクトルインデックス作成
        logger.info("ベクトルインデックスを作成中...")
        vector_store = processor.create_vector_index(processed_docs)

        # インデックスを保存
        vector_store.save_local(index_path)
        logger.info(f"ベクトルインデックスを保存しました: {index_path}")

    # 4. RAGシステム構築
    logger.info("RAGシステムを構築中...")
    rag_system = RAGSystem(vector_store)

    return rag_system

# 7. 質問応答用の関数
async def answer_question(question, base_url="https://support.iriam.com/hc/ja", max_pages=100, index_path="faiss_index"):
    """質問に回答する"""
    # RAGシステムを取得（ロードまたは構築）
    rag_system = await load_or_build_rag_system(base_url, max_pages, index_path)

    # 質問に回答
    result = rag_system.answer_question(question)

    return result

# 8. 実行例
if __name__ == "__main__":
    # asyncioでメイン関数を実行
    async def main():
        # 質問を設定
        test_question = "Iriamのパスワードをリセットする方法は？"

        # 質問に回答
        result = await answer_question(test_question)
        
        print("\n=== 質問 ===")
        print(test_question)
        print("\n=== 回答 ===")
        print(result["answer"])
        print("\n=== 情報ソース ===")
        for url in result["sources"]:
            print(f"- {url}")

    # asyncioイベントループで実行
    asyncio.run(main())