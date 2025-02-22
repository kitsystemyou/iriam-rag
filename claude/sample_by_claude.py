import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
import os
import time
import logging
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IriamSupportBot:
    def __init__(self, api_key):
        self.base_url = "https://support.iriam.com/hc/ja/articles/20056952605721-ウェルカムパスについて"
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        """セッションの設定とヘッダーの初期化"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        })

    def get_article_urls(self):
        """サポートサイトから記事URLを取得"""
        urls = set()
        urls.add(self.base_url)  # 初期URLを追加
        
        try:
            logger.info(f"Starting with initial article: {self.base_url}")
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 関連記事やナビゲーションからリンクを収集
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/articles/' in href and 'ja' in href:
                    full_url = urllib.parse.urljoin('https://support.iriam.com', href)
                    urls.add(full_url)
                    logger.info(f"Found related article: {full_url}")
            
            # サイドバーや関連リンクからも記事を収集
            sidebar_links = soup.find_all('a', class_=['article-list-link', 'sidenav-item'])
            for link in sidebar_links:
                href = link.get('href')
                if href and '/articles/' in href:
                    full_url = urllib.parse.urljoin('https://support.iriam.com', href)
                    urls.add(full_url)
                    logger.info(f"Found sidebar article: {full_url}")
            
            urls = list(urls)
            logger.info(f"Total articles found: {len(urls)}")
            return urls
            
        except requests.RequestException as e:
            logger.error(f"Error fetching articles: {e}")
            return [self.base_url]  # エラーの場合は初期URLのみ返す

    def scrape_documents(self):
        """サポートサイトからドキュメントを取得"""
        docs = []
        urls = self.get_article_urls()
        
        for url in urls:
            try:
                logger.info(f"Fetching article from {url}")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # タイトルの取得
                title_elem = soup.find('h1', class_='article-title')
                title = title_elem.get_text(strip=True) if title_elem else "無題"
                
                # 本文の取得
                article_body = soup.find('div', class_='article-body')
                if article_body:
                    # 不要な要素を除去
                    for elem in article_body.find_all(['script', 'style', 'nav']):
                        elem.decompose()
                    
                    # テキストの整形
                    paragraphs = []
                    for p in article_body.find_all(['p', 'h2', 'h3', 'li']):
                        text = p.get_text(strip=True)
                        if text:
                            paragraphs.append(text)
                    
                    content = '\n\n'.join(paragraphs)
                    
                    if content:
                        doc = Document(
                            page_content=f"タイトル: {title}\n\n{content}",
                            metadata={
                                "source": url,
                                "title": title
                            }
                        )
                        docs.append(doc)
                        logger.info(f"Successfully extracted content from article: {title}")
                
                time.sleep(2)  # サイトへの負荷を考慮して待機
                
            except requests.RequestException as e:
                logger.error(f"Error fetching article {url}: {e}")
                continue
            
        if not docs:
            logger.warning("No content could be extracted. Creating sample documents.")
            return self.create_sample_documents()
            
        return docs

    def create_sample_documents(self):
        """サンプルドキュメントを作成"""
        return [
            Document(
                page_content="""
                タイトル: ウェルカムパスについて

                ウェルカムパスは、IRIAMの新規ユーザーに向けた基本機能の学習パスです。
                以下の内容を順番に学習できます：

                1. アカウントの基本設定
                2. 基本的な操作方法
                3. よく使う機能の解説
                4. 応用的な使い方のヒント

                このパスを完了することで、IRIAMの基本的な使い方を習得できます。
                """,
                metadata={
                    "source": self.base_url,
                    "title": "ウェルカムパスについて"
                }
            )
        ]

    def create_vectorstore(self, documents):
        """ドキュメントからベクトルストアを作成"""
        if not documents:
            raise ValueError("No documents provided for creating vector store")
            
        texts = self.text_splitter.split_documents(documents)
        if not texts:
            raise ValueError("No texts generated after splitting documents")
            
        logger.info(f"Creating vector store with {len(texts)} text chunks")
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def setup_chatbot(self, vectorstore):
        """チャットボットの設定"""
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": """あなたはIRIAMのサポートアシスタントです。
                ユーザーからの質問に対して、提供された文書の情報に基づいて簡潔に回答してください。
                情報が見つからない場合は、その旨を正直に伝え、IRIAMのサポートへの問い合わせを提案してください。"""
            }]
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain

    def run(self):
        """チャットボットの実行"""
        try:
            print("ドキュメントを収集中...")
            docs = self.scrape_documents()
            
            print(f"収集された記事数: {len(docs)}")
            
            print("\nベクトルストアを作成中...")
            vectorstore = self.create_vectorstore(docs)
            
            print("チャットボットを準備中...")
            qa_chain = self.setup_chatbot(vectorstore)
            
            chat_history = []
            
            print("\nチャットボットの準備が完了しました。質問を入力してください。")
            
            while True:
                question = input("\nご質問を入力してください（終了する場合は 'quit' と入力）: ")
                if question.lower() == 'quit':
                    break
                    
                result = qa_chain({"question": question, "chat_history": chat_history})
                print("\n回答:", result["answer"])
                print("\n参照元:", [f"{doc.metadata.get('title', '不明')} ({doc.metadata.get('source', '不明')})" for doc in result["source_documents"]])
                
                chat_history.append((question, result["answer"]))
                
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            raise

# 使用例
if __name__ == "__main__":
    api_key = ""
    bot = IriamSupportBot(api_key)
    bot.run()