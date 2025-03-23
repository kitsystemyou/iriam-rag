import os
import pandas as pd
from document_processor import DocumentProcessor
from langchain_openai import OpenAIEmbeddings


def main():
    # クロールデータの読み込み
    df = pd.read_csv("data/crawled_data.csv")
    crawled_data = df.to_dict('records')
    
    # テキスト処理とベクトル化
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    processed_docs = processor.process_documents(crawled_data)
    vector_store = processor.create_vector_index(processed_docs)
    
    # インデックスの保存
    os.makedirs('data/faiss_index', exist_ok=True)
    vector_store.save_local("data/faiss_index")
    print("ベクトルインデックスを保存しました。")

if __name__ == "__main__":
    # dbg
    # os.environ["OPENAI_API_KEY"]
    main()
