import os
import pandas as pd
import logging
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DocumentProcessorクラスの定義
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

def main():
    logger.info("ベクトルインデックス作成を開始します...")
    
    # クロールデータの読み込み
    logger.info("クロールデータを読み込んでいます...")
    df = pd.read_csv("data/crawled_data.csv")
    crawled_data = df.to_dict('records')
    logger.info(f"{len(crawled_data)}件のドキュメントを読み込みました。")
    
    # テキスト処理とベクトル化
    logger.info("テキスト処理を開始します...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    processed_docs = processor.process_documents(crawled_data)
    logger.info(f"テキスト処理完了。{len(processed_docs)}件のチャンクに分割されました。")
    
    logger.info("ベクトルインデックスを作成中...")
    vector_store = processor.create_vector_index(processed_docs)
    
    # インデックスの保存
    logger.info("ベクトルインデックスを保存します...")
    os.makedirs('data/faiss_index', exist_ok=True)
    vector_store.save_local("data/faiss_index")
    logger.info("ベクトルインデックスを保存しました: data/faiss_index")

if __name__ == "__main__":
    # dbg
    # os.environ["OPENAI_API_KEY"] = ""
    main()
