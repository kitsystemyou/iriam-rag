# iriam-rag
rag chat bot for iriam

# prototype code

- proto_rag.py: all process for RAG
- devided codes
  1. scripts/crawler.py: crawling and save result -> crawled_data.csv
  1. scripts/build_index.py: create vector and save result -> faiss_indes/*
  1. answer.py: answer for input question (arg param)

### example for answer.py
```
# 単一の質問に回答
python query_rag.py --question "あんしんランクスコアとは何ですか？"

# 対話モードで起動
python query_rag.py --interactive

# 別のモデルを使用
python query_rag.py --model "gpt-4" --question "ユーザー認証の方法を教えてください"

# カスタムインデックスパスを指定
python query_rag.py --index "custom/index/path" --interactive
```
