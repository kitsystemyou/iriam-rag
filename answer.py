import os
import logging
import argparse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        あなたはIRIAMサポートの専門アシスタントです。
        以下の情報を参考にして、ユーザーの質問に丁寧に回答してください。
        また、あなたの名前は「ミリア」で20歳の元気な女の子です。大学のお友達に話しかけるような口調で回答してください。
        
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
        logger.info(f"質問: {question}")
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

def load_rag_system(index_path="scripts/data/faiss_index", model_name="gpt-3.5-turbo"):
    """既存のインデックスをロードしてRAGシステムを構築する"""
    logger.info(f"インデックスをロード中: {index_path}")
    
    # 環境変数の確認
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEYが環境変数に設定されていません。")
    
    # 埋め込みモデルの初期化
    embeddings = OpenAIEmbeddings()
    
    # インデックスをロード
    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("インデックスのロードが完了しました。")
    except Exception as e:
        logger.error(f"インデックスのロード中にエラーが発生しました: {str(e)}")
        raise
    
    # RAGシステムの構築
    logger.info(f"RAGシステムを構築中... (model: {model_name})")
    rag_system = RAGSystem(vector_store, model_name=model_name)
    
    return rag_system

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='RAGシステムを使用して質問に回答します。')
    parser.add_argument('--index', default='scripts/data/faiss_index', help='FAISSインデックスのパス')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='使用するOpenAIモデル')
    parser.add_argument('--question', help='回答する質問')
    parser.add_argument('--interactive', action='store_true', help='対話モードで実行')
    
    args = parser.parse_args()
    
    # RAGシステムのロード
    rag_system = load_rag_system(args.index, args.model)
    
    if args.interactive:
        # 対話モード
        print("RAGシステムの対話モードを開始します。終了するには 'exit' または 'quit' と入力してください。")
        while True:
            user_input = input("\n質問を入力してください: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            try:
                # 質問に回答
                result = rag_system.answer_question(user_input)
                
                # 結果の表示
                print("\n=== 回答 ===")
                print(result["answer"])
                print("\n=== 情報ソース ===")
                for url in result["sources"]:
                    print(f"- {url}")
            except Exception as e:
                logger.error(f"質問応答中にエラーが発生しました: {str(e)}")
                print(f"エラーが発生しました: {str(e)}")
    
    elif args.question:
        # 単一の質問に回答
        try:
            result = rag_system.answer_question(args.question)
            
            # 結果の表示
            print("\n=== 質問 ===")
            print(args.question)
            print("\n=== 回答 ===")
            print(result["answer"])
            print("\n=== 情報ソース ===")
            for url in result["sources"]:
                print(f"- {url}")
        except Exception as e:
            logger.error(f"質問応答中にエラーが発生しました: {str(e)}")
            print(f"エラーが発生しました: {str(e)}")
    
    else:
        print("質問を指定するか、対話モード(--interactive)を使用してください。")
        parser.print_help()

if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = ""
    main()
