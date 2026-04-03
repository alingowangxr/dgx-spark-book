# 第 26 章：綜合實戰：從零構建企業級 RAG 知識庫系統

::: tip 🎯 本章你將學到什麼
- 企業級 RAG 系統的完整架構設計
- 文件解析、分塊、嵌入的完整流程
- 混合檢索（向量 + 關鍵字 + 圖譜）
- 多 Agent 協作的 RAG 系統
- 效能監控與品質評估
- 部署與維運最佳實踐
:::

::: warning ⏱️ 預計閱讀時間
約 35 分鐘。完整部署約需 1-2 小時。
:::

---

## 26-1 專案概述

### 26-1-1 什麼是企業級 RAG 系統？

::: info 🤔 為什麼需要「企業級」？
前面章節介紹的 RAG 是基礎版本，適合個人使用。但企業場景有更多需求：

| 需求 | 基礎 RAG | 企業級 RAG |
|------|---------|-----------|
| 文件數量 | 幾十份 | 數萬份 |
| 文件類型 | 純文字 | PDF、Word、Excel、PPT、HTML |
| 檢索準確率 | 一般 | 高（混合檢索） |
| 多人共用 | 不支援 | 權限管理 |
| 回應品質 | 基本 | 多 Agent 協作 |
| 監控 | 無 | 完整指標 |
| 擴展性 | 單機 | 可水平擴展 |
:::

### 26-1-2 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    使用者介面層                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Web UI   │  │ API      │  │ Slack    │  │ Teams    │    │
│  │ (Open    │  │ Gateway  │  │ Bot      │  │ Bot      │    │
│  │  WebUI)  │  │          │  │          │  │          │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                    應用服務層                                 │
│                    ┌──────┴──────┐                           │
│                    │  Orchestrator│ ← 任務調度與路由           │
│                    │  (FastAPI)   │                           │
│                    └──┬───┬───┬──┘                           │
│               ┌───────┘   │   └───────┐                      │
│               ▼           ▼           ▼                      │
│  ┌────────────────┐ ┌──────────┐ ┌────────────────┐         │
│  │ Query Agent    │ │ Doc Agent│ │ Monitor Agent  │         │
│  │ (查詢理解)     │ │ (文件處理)│ │ (監控與評估)    │         │
│  └───────┬────────┘ └────┬─────┘ └────────┬───────┘         │
└──────────┼───────────────┼────────────────┼──────────────────┘
           │               │                │
┌──────────┼───────────────┼────────────────┼──────────────────┐
│          │   資料層      │                │                  │
│          ▼               ▼                ▼                  │
│  ┌──────────────┐ ┌──────────┐  ┌──────────────────┐        │
│  │ 向量資料庫    │ │ 全文檢索 │  │ 知識圖譜          │        │
│  │ (ChromaDB)   │ │ (Meilisearch)│ (ArangoDB)     │        │
│  └──────┬───────┘ └────┬─────┘  └────────┬─────────┘        │
│         │              │                 │                   │
│  ┌──────┴──────────────┴─────────────────┴──────────┐       │
│  │              Ollama / vLLM 推論服務               │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │       │
│  │  │ Embedding│  │ LLM      │  │ Reranker     │   │       │
│  │  │ Model    │  │ Model    │  │ Model        │   │       │
│  │  └──────────┘  └──────────┘  └──────────────┘   │       │
│  └───────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 26-1-3 技術選型

| 元件 | 選擇 | 原因 |
|------|------|------|
| **嵌入模型** | BGE-M3（Ollama） | 中文支援好、開源、1024 維 |
| **LLM** | Qwen3-8B（Ollama） | 中文能力強、速度快 |
| **Reranker** | BGE-Reranker | 提升檢索準確率 |
| **向量資料庫** | ChromaDB | 輕量、嵌入簡單 |
| **全文檢索** | Meilisearch | 速度快、中文分詞好 |
| **知識圖譜** | ArangoDB | 多模型、圖查詢強 |
| **API 框架** | FastAPI | 非同步、自動文件 |
| **前端** | Open WebUI | 開箱即用 |

---

## 26-2 環境建立

### 26-2-1 建立專案結構

告訴 Claude Code：

> 「幫我建立一個企業級 RAG 系統的專案結構，使用 FastAPI 作為後端框架。」

Claude Code 會執行：

```bash
# 建立專案目錄
mkdir -p ~/rag-system/{app,config,scripts,tests,data/{raw,processed,models}}
cd ~/rag-system

# 建立 Python 環境
uv venv .venv
source .venv/bin/activate

# 安裝核心套件
uv pip install \
  fastapi uvicorn \
  chromadb \
  langchain langchain-community langchain-text-splitters \
  openai \
  python-multipart \
  python-dotenv \
  pydantic \
  httpx \
  pymupdf python-docx openpyxl \
  meilisearch \
  pyarango \
  matplotlib plotly
```

### 26-2-2 建立專案結構

```
rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 主程式
│   ├── config.py            # 設定管理
│   ├── models/              # Pydantic 模型
│   │   ├── query.py
│   │   └── document.py
│   ├── services/            # 業務邏輯
│   │   ├── embedding.py     # 嵌入服務
│   │   ├── retriever.py     # 檢索服務
│   │   ├── generator.py     # 生成服務
│   │   ├── document.py      # 文件處理
│   │   └── monitor.py       # 監控服務
│   ├── pipelines/           # 處理流程
│   │   ├── ingestion.py     # 文件匯入
│   │   └── retrieval.py     # 檢索流程
│   └── api/                 # API 路由
│       ├── query.py
│       └── documents.py
├── config/
│   ├── settings.yaml        # 系統設定
│   └── prompts.yaml         # 提示詞模板
├── data/
│   ├── raw/                 # 原始文件
│   ├── processed/           # 處理後的資料
│   └── models/              # 模型快取
├── scripts/
│   ├── ingest.py            # 批次匯入腳本
│   └── evaluate.py          # 評估腳本
└── tests/
    └── test_retrieval.py
```

### 26-2-3 設定檔案

建立 `config/settings.yaml`：

```yaml
# RAG 系統設定

# Ollama 設定
ollama:
  base_url: "http://localhost:11434"
  embedding_model: "bge-m3"
  llm_model: "qwen3-8b"
  reranker_model: "bge-reranker-v2-m3"

# 向量資料庫設定
chromadb:
  path: "./data/chroma_db"
  collection: "rag_documents"

# 全文檢索設定
meilisearch:
  url: "http://localhost:7700"
  api_key: "masterKey"
  index: "documents"

# 知識圖譜設定
arangodb:
  url: "http://localhost:8529"
  database: "rag_knowledge"
  username: "root"
  password: "dgxspark"

# 文件處理設定
document:
  chunk_size: 500
  chunk_overlap: 50
  supported_formats:
    - pdf
    - docx
    - txt
    - md
    - html
    - csv

# 檢索設定
retrieval:
  top_k: 5
  vector_weight: 0.6
  keyword_weight: 0.3
  graph_weight: 0.1
  min_similarity: 0.3

# 生成設定
generation:
  temperature: 0.3
  max_tokens: 2048
  system_prompt: |
    你是一個專業的知識庫助手。
    請根據提供的上下文回答問題。
    如果無法從上下文中找到答案，請誠實回答「我不知道」。
    請用繁體中文回答。
```

---

## 26-3 文件處理管線

### 26-3-1 文件解析器

建立 `app/services/document.py`：

```python
"""文件處理服務：解析各種格式的文件"""

import os
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx
import openpyxl
import markdown
from bs4 import BeautifulSoup


class DocumentParser:
    """支援多種文件格式的解析器"""

    SUPPORTED_FORMATS = {
        '.pdf': '_parse_pdf',
        '.docx': '_parse_docx',
        '.txt': '_parse_txt',
        '.md': '_parse_md',
        '.html': '_parse_html',
        '.csv': '_parse_csv',
    }

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析文件，回傳結構化資料"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支援的檔案格式: {ext}")

        method_name = self.SUPPORTED_FORMATS[ext]
        method = getattr(self, method_name)

        content = method(file_path)

        return {
            "file_path": str(path),
            "file_name": path.name,
            "file_type": ext,
            "file_size": path.stat().st_size,
            "content": content,
            "metadata": self._extract_metadata(path, content),
        }

    def _parse_pdf(self, file_path: str) -> str:
        """解析 PDF 文件"""
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    def _parse_docx(self, file_path: str) -> str:
        """解析 Word 文件"""
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    def _parse_txt(self, file_path: str) -> str:
        """解析純文字檔案"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_md(self, file_path: str) -> str:
        """解析 Markdown 檔案"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_html(self, file_path: str) -> str:
        """解析 HTML 檔案"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)

    def _parse_csv(self, file_path: str) -> str:
        """解析 CSV 檔案（轉為文字描述）"""
        import pandas as pd
        df = pd.read_csv(file_path)
        # 轉為結構化文字描述
        lines = [f"表格包含 {len(df)} 列 {len(df.columns)} 欄"]
        lines.append(f"欄位: {', '.join(df.columns)}")
        lines.append(f"前 5 列資料:\n{df.head().to_string()}")
        return "\n".join(lines)

    def _extract_metadata(self, path: Path, content: str) -> Dict[str, Any]:
        """提取文件中繼資料"""
        return {
            "char_count": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n') + 1,
        }
```

### 26-3-2 智慧分塊策略

建立 `app/pipelines/ingestion.py`：

```python
"""文件匯入管線：解析 → 分塊 → 嵌入 → 儲存"""

from typing import List, Dict, Any
import re
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


class SmartChunker:
    """智慧分塊器：根據文件類型選擇最佳分塊策略"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 通用分塊器
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
            length_function=len,
        )

        # Markdown 分塊器
        self.md_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """將文件分塊"""
        content = document["content"]
        file_type = document["file_type"]

        # 根據類型選擇分塊器
        if file_type == '.md':
            splitter = self.md_splitter
        else:
            splitter = self.general_splitter

        # 先按標題分割（保持語意完整性）
        sections = self._split_by_headings(content)

        chunks = []
        chunk_id = 0

        for section in sections:
            # 如果 section 太大，再細分
            if len(section) > self.chunk_size * 1.5:
                sub_chunks = splitter.split_text(section)
            else:
                sub_chunks = [section]

            for sub_chunk in sub_chunks:
                chunks.append({
                    "chunk_id": f"{document['file_name']}_{chunk_id}",
                    "file_name": document["file_name"],
                    "file_path": document["file_path"],
                    "content": sub_chunk.strip(),
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": chunk_id,
                        "total_chunks": len(sub_chunks),
                    }
                })
                chunk_id += 1

        return chunks

    def _split_by_headings(self, text: str) -> List[str]:
        """按標題分割文字（保持語意段落完整）"""
        # 支援 Markdown 和一般文字的標題
        heading_pattern = r'^(#{1,6}\s+.+)$|^(第[一二三四五六七八九十\d]+[章節篇].+)$'
        lines = text.split('\n')
        sections = []
        current_section = []

        for line in lines:
            if re.match(heading_pattern, line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        return sections if sections else [text]
```

---

## 26-4 混合檢索系統

### 26-4-1 向量檢索

建立 `app/services/embedding.py`：

```python
"""嵌入服務：將文字轉為向量"""

import requests
from typing import List
from config import settings


class EmbeddingService:
    """使用 Ollama 的嵌入模型"""

    def __init__(self):
        self.base_url = settings.ollama.base_url
        self.model = settings.ollama.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """將文字轉為向量"""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                }
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """單一查詢嵌入"""
        return self.embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文件嵌入"""
        # 批次處理，避免一次送太多
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embed(batch))
        return all_embeddings
```

### 26-4-2 向量檢索服務

建立 `app/services/retriever.py`：

```python
"""檢索服務：混合檢索（向量 + 關鍵字 + 圖譜）"""

import chromadb
import requests
from typing import List, Dict, Any
from config import settings


class HybridRetriever:
    """混合檢索器：結合向量、關鍵字和知識圖譜"""

    def __init__(self):
        # 向量資料庫
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chromadb.path
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chromadb.collection,
            metadata={"hnsw:space": "cosine"}
        )

        # 全文檢索
        self.meili_url = settings.meilisearch.url
        self.meili_key = settings.meilisearch.api_key
        self.meili_index = settings.meilisearch.index

        # 檢索設定
        self.top_k = settings.retrieval.top_k
        self.vector_weight = settings.retrieval.vector_weight
        self.keyword_weight = settings.retrieval.keyword_weight
        self.graph_weight = settings.retrieval.graph_weight

    def search(self, query: str) -> List[Dict[str, Any]]:
        """執行混合檢索"""
        # 1. 向量檢索
        vector_results = self._vector_search(query)

        # 2. 關鍵字檢索
        keyword_results = self._keyword_search(query)

        # 3. 知識圖譜檢索（可選）
        graph_results = self._graph_search(query)

        # 4. 融合排名
        fused_results = self._fuse_results(
            vector_results, keyword_results, graph_results
        )

        return fused_results[:self.top_k]

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """向量相似度檢索"""
        from app.services.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k * 2,  # 多取一些，後續融合時篩選
            include=["documents", "metadatas", "distances"]
        )

        vector_results = []
        for i in range(len(results["ids"][0])):
            vector_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_score": 1 - results["distances"][0][i],  # 轉為相似度
            })

        return vector_results

    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """關鍵字全文檢索"""
        headers = {"Authorization": f"Bearer {self.meili_key}"}
        response = requests.post(
            f"{self.meili_url}/indexes/{self.meili_index}/search",
            headers=headers,
            json={
                "q": query,
                "limit": self.top_k * 2,
            }
        )

        keyword_results = []
        for hit in response.json().get("hits", []):
            keyword_results.append({
                "chunk_id": hit.get("chunk_id"),
                "content": hit.get("content"),
                "metadata": hit.get("metadata", {}),
                "keyword_score": hit.get("_rankingScore", 0),
            })

        return keyword_results

    def _graph_search(self, query: str) -> List[Dict[str, Any]]:
        """知識圖譜檢索"""
        # 簡化版：從图谱中找相關實體
        # 實際應用中會用 NER 提取實體，然後查詢图谱
        return []

    def _fuse_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        graph_results: List[Dict],
    ) -> List[Dict[str, Any]]:
        """融合多種檢索結果（RRF 算法）"""
        # 建立 chunk_id 到分數的映射
        scores = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "total_score": 0,
                    "sources": {},
                }
            scores[chunk_id]["total_score"] += (
                result["vector_score"] * self.vector_weight
            )
            scores[chunk_id]["sources"]["vector"] = result["vector_score"]

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "total_score": 0,
                    "sources": {},
                }
            scores[chunk_id]["total_score"] += (
                result["keyword_score"] * self.keyword_weight
            )
            scores[chunk_id]["sources"]["keyword"] = result["keyword_score"]

        # 按總分排序
        fused = sorted(
            scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )

        return fused

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """新增文件到檢索系統"""
        from app.services.embedding import EmbeddingService
        embedding_service = EmbeddingService()

        # 嵌入所有 chunk
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_service.embed_documents(texts)

        # 存入 ChromaDB
        self.collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in chunks],
        )

        # 存入 Meilisearch
        headers = {"Authorization": f"Bearer {self.meili_key}"}
        meili_docs = []
        for chunk in chunks:
            meili_docs.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "file_name": chunk["file_name"],
                "metadata": chunk["metadata"],
            })

        requests.post(
            f"{self.meili_url}/indexes/{self.meili_index}/documents",
            headers=headers,
            json=meili_docs,
        )
```

---

## 26-5 生成服務

### 26-5-1 RAG 生成器

建立 `app/services/generator.py`：

```python
"""生成服務：基於檢索結果生成回答"""

import requests
import yaml
from typing import List, Dict, Any
from config import settings


class RAGGenerator:
    """RAG 生成器：結合上下文生成回答"""

    def __init__(self):
        self.base_url = settings.ollama.base_url
        self.model = settings.ollama.llm_model
        self.temperature = settings.generation.temperature
        self.max_tokens = settings.generation.max_tokens
        self.system_prompt = settings.generation.system_prompt

        # 載入提示詞模板
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """生成回答"""
        # 組裝上下文
        context_text = self._format_contexts(contexts)

        # 選擇提示詞模板
        prompt_template = self.prompts["rag_qa"]["template"]

        # 組裝完整提示
        system_prompt = self.system_prompt
        user_prompt = prompt_template.format(
            context=context_text,
            question=query,
        )

        # 呼叫 Ollama
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
        )
        response.raise_for_status()

        result = response.json()

        return {
            "answer": result["message"]["content"],
            "model": self.model,
            "contexts": contexts,
            "context_count": len(contexts),
        }

    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """格式化上下文為文字"""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx["metadata"].get("file_name", "未知來源")
            parts.append(
                f"[來源 {i}] {source}\n{ctx['content']}"
            )
        return "\n\n".join(parts)
```

### 26-5-2 提示詞模板

建立 `config/prompts.yaml`：

```yaml
# 提示詞模板

rag_qa:
  template: |
    請根據以下提供的上下文資訊回答問題。

    【上下文】
    {context}

    【問題】
    {question}

    【回答要求】
    1. 只根據提供的上下文回答
    2. 如果上下文中沒有相關資訊，請回答「根據目前的知識庫，我無法回答這個問題」
    3. 回答時請註明資訊來源
    4. 使用繁體中文

rag_summary:
  template: |
    請總結以下文件的要點。

    【文件內容】
    {context}

    【總結要求】
    1. 列出 3-5 個主要要點
    2. 每個要點不超過 50 字
    3. 使用繁體中文

rag_compare:
  template: |
    請比較以下兩個概念的差異。

    【上下文】
    {context}

    【比較對象】
    {question}

    【比較要求】
    1. 使用表格比較
    2. 列出相同點和不同點
    3. 使用繁體中文
```

---

## 26-6 FastAPI 主程式

### 26-6-1 API 路由

建立 `app/main.py`：

```python
"""FastAPI 主程式：RAG 系統 API 入口"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from app.services.document import DocumentParser
from app.pipelines.ingestion import SmartChunker
from app.services.retriever import HybridRetriever
from app.services.generator import RAGGenerator

app = FastAPI(
    title="企業級 RAG 知識庫系統",
    description="從零構建的企業級 RAG 系統，支援混合檢索和多 Agent 協作",
    version="1.0.0",
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服務
parser = DocumentParser()
chunker = SmartChunker(chunk_size=500, chunk_overlap=50)
retriever = HybridRetriever()
generator = RAGGenerator()


# ===== 請求/回應模型 =====

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.3


class QueryResponse(BaseModel):
    answer: str
    model: str
    context_count: int
    sources: List[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    status: str


# ===== API 端點 =====

@app.get("/")
async def root():
    """健康檢查"""
    return {
        "status": "ok",
        "service": "RAG Knowledge Base System",
        "version": "1.0.0",
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG 問答端點"""
    try:
        # 1. 混合檢索
        results = retriever.search(request.question)

        if not results:
            return QueryResponse(
                answer="根據目前的知識庫，我找不到相關資訊。",
                model=generator.model,
                context_count=0,
                sources=[],
            )

        # 2. 生成回答
        response = generator.generate(request.question, results)

        # 3. 提取來源
        sources = list(set([
            ctx["metadata"].get("file_name", "未知")
            for ctx in results
        ]))

        return QueryResponse(
            answer=response["answer"],
            model=response["model"],
            context_count=response["context_count"],
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=List[IngestResponse])
async def ingest_files(files: List[UploadFile] = File(...)):
    """匯入文件到知識庫"""
    results = []

    for file in files:
        try:
            # 1. 儲存上傳檔案
            file_path = f"data/raw/{file.filename}"
            os.makedirs("data/raw", exist_ok=True)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 2. 解析文件
            document = parser.parse(file_path)

            # 3. 分塊
            chunks = chunker.chunk(document)

            # 4. 存入檢索系統
            retriever.add_documents(chunks)

            results.append(IngestResponse(
                file_name=file.filename,
                chunks_count=len(chunks),
                status="success",
            ))

        except Exception as e:
            results.append(IngestResponse(
                file_name=file.filename,
                chunks_count=0,
                status=f"error: {str(e)}",
            ))

    return results


@app.post("/api/ingest-directory")
async def ingest_directory(directory: str = "data/raw"):
    """批次匯入目錄中的所有文件"""
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail=f"目錄不存在: {directory}")

    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                document = parser.parse(file_path)
                chunks = chunker.chunk(document)
                retriever.add_documents(chunks)
                results.append({
                    "file_name": filename,
                    "chunks_count": len(chunks),
                    "status": "success",
                })
            except Exception as e:
                results.append({
                    "file_name": filename,
                    "chunks_count": 0,
                    "status": f"error: {str(e)}",
                })

    return {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "errors": sum(1 for r in results if r["status"] != "success"),
        "details": results,
    }


@app.get("/api/stats")
async def stats():
    """系統統計資訊"""
    return {
        "vector_db": {
            "collection": retriever.collection.name,
            "count": retriever.collection.count(),
        },
        "models": {
            "embedding": "bge-m3",
            "llm": "qwen3-8b",
        },
        "settings": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 5,
        }
    }
```

### 26-6-2 啟動服務

```bash
cd ~/rag-system
source .venv/bin/activate

# 啟動 FastAPI 服務
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

打開 `http://DGX_Spark_IP:8000/docs` 可以看到自動生成的 API 文件。

---

## 26-7 批次匯入腳本

### 26-7-1 建立匯入腳本

建立 `scripts/ingest.py`：

```python
#!/usr/bin/env python3
"""批次匯入腳本：將目錄中的所有文件匯入知識庫"""

import os
import sys
import argparse
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document import DocumentParser
from app.pipelines.ingestion import SmartChunker
from app.services.retriever import HybridRetriever


def ingest_directory(directory: str, dry_run: bool = False):
    """批次匯入目錄"""
    parser = DocumentParser()
    chunker = SmartChunker()
    retriever = HybridRetriever()

    path = Path(directory)
    if not path.exists():
        print(f"❌ 目錄不存在: {directory}")
        return

    files = list(path.glob("*"))
    print(f"📁 找到 {len(files)} 個檔案")

    total_chunks = 0
    success = 0
    errors = 0

    for file_path in files:
        if file_path.is_file():
            try:
                print(f"\n📄 處理: {file_path.name}")

                # 解析
                document = parser.parse(str(file_path))
                print(f"   內容長度: {document['metadata']['char_count']} 字元")

                # 分塊
                chunks = chunker.chunk(document)
                print(f"   分塊數: {len(chunks)}")

                if not dry_run:
                    # 存入
                    retriever.add_documents(chunks)
                    print(f"   ✅ 已存入知識庫")

                total_chunks += len(chunks)
                success += 1

            except Exception as e:
                print(f"   ❌ 錯誤: {e}")
                errors += 1

    print(f"\n{'='*50}")
    print(f"📊 匯入統計")
    print(f"   成功: {success} 個檔案")
    print(f"   失敗: {errors} 個檔案")
    print(f"   總分塊數: {total_chunks}")
    if not dry_run:
        print(f"   向量資料庫總數: {retriever.collection.count()}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="批次匯入文件到知識庫")
    arg_parser.add_argument("directory", help="要匯入的目錄")
    arg_parser.add_argument("--dry-run", action="store_true", help="預覽模式，不實際匯入")
    args = arg_parser.parse_args()

    ingest_directory(args.directory, args.dry_run)
```

執行：

```bash
# 預覽模式
python scripts/ingest.py data/raw --dry-run

# 實際匯入
python scripts/ingest.py data/raw
```

---

## 26-8 整合 Open WebUI

### 26-8-1 添加自訂 RAG 端點

Open WebUI 可以連接任何 OpenAI 相容的 API。我們的 RAG 系統已經提供了 `/api/query` 端點，但需要包裝成 OpenAI 格式。

在 `app/main.py` 中加入：

```python
@app.post("/v1/chat/completions")
async def openai_compat_chat(request: dict):
    """OpenAI 相容端點（供 Open WebUI 使用）"""
    messages = request.get("messages", [])
    question = messages[-1]["content"] if messages else ""

    # RAG 檢索 + 生成
    results = retriever.search(question)
    response = generator.generate(question, results)

    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "created": 0,
        "model": "rag-system",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response["answer"],
            },
            "finish_reason": "stop",
        }],
    }


@app.get("/v1/models")
async def openai_compat_models():
    """列出模型（OpenAI 相容）"""
    return {
        "data": [{
            "id": "rag-system",
            "object": "model",
            "created": 0,
            "owned_by": "rag-system",
        }]
    }
```

### 26-8-2 在 Open WebUI 中連接

1. 打開 Open WebUI
2. **Admin Panel → Settings → Connections**
3. 添加新的 OpenAI 端點：
   - URL：`http://DGX_Spark_IP:8000`
   - API Key：任意填寫
4. 儲存後，在模型選擇器中選擇 `rag-system`

現在你的 Open WebUI 對話會自動使用 RAG 知識庫！

---

## 26-9 效能評估

### 26-9-1 建立評估腳本

建立 `scripts/evaluate.py`：

```python
#!/usr/bin/env python3
"""RAG 系統評估腳本"""

import json
import time
from typing import List, Dict


class RAGEvaluator:
    """RAG 系統評估器"""

    def __init__(self):
        self.results = []

    def evaluate(
        self,
        test_cases: List[Dict],
        retriever,
        generator,
    ) -> Dict:
        """執行評估"""
        for case in test_cases:
            start_time = time.time()

            # 檢索
            contexts = retriever.search(case["question"])

            # 生成
            response = generator.generate(case["question"], contexts)

            elapsed = time.time() - start_time

            self.results.append({
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "actual_answer": response["answer"],
                "context_count": response["context_count"],
                "sources": [ctx["metadata"].get("file_name") for ctx in contexts],
                "latency": elapsed,
                "retrieval_relevant": self._check_relevance(
                    contexts, case["expected_sources"]
                ),
            })

        return self._compute_metrics()

    def _check_relevance(self, contexts, expected_sources):
        """檢查檢索結果是否包含預期來源"""
        actual_sources = set([
            ctx["metadata"].get("file_name") for ctx in contexts
        ])
        expected = set(expected_sources)
        return len(actual_sources & expected) / len(expected) if expected else 0

    def _compute_metrics(self) -> Dict:
        """計算評估指標"""
        if not self.results:
            return {}

        avg_latency = sum(r["latency"] for r in self.results) / len(self.results)
        avg_context_count = sum(r["context_count"] for r in self.results) / len(self.results)
        avg_retrieval_relevant = sum(r["retrieval_relevant"] for r in self.results) / len(self.results)

        return {
            "total_cases": len(self.results),
            "avg_latency": f"{avg_latency:.2f}s",
            "avg_context_count": f"{avg_context_count:.1f}",
            "retrieval_precision": f"{avg_retrieval_relevant:.1%}",
            "details": self.results,
        }


# 測試案例範例
TEST_CASES = [
    {
        "question": "DGX Spark 的記憶體容量是多少？",
        "expected_answer": "128GB LPDDR5x",
        "expected_sources": ["dgx-spark-specs.pdf"],
    },
    {
        "question": "如何安裝 Ollama？",
        "expected_answer": "使用 curl 指令下載安裝腳本",
        "expected_sources": ["ollama-setup-guide.md"],
    },
]

if __name__ == "__main__":
    from app.services.retriever import HybridRetriever
    from app.services.generator import RAGGenerator

    retriever = HybridRetriever()
    generator = RAGGenerator()
    evaluator = RAGEvaluator()

    metrics = evaluator.evaluate(TEST_CASES, retriever, generator)

    print("\n📊 RAG 系統評估報告")
    print(f"   測試案例: {metrics['total_cases']}")
    print(f"   平均延遲: {metrics['avg_latency']}")
    print(f"   平均上下文數: {metrics['avg_context_count']}")
    print(f"   檢索精確率: {metrics['retrieval_precision']}")
```

執行評估：

```bash
python scripts/evaluate.py
```

### 26-9-2 評估指標說明

| 指標 | 說明 | 目標值 |
|------|------|--------|
| **檢索精確率** | 檢索到的文件與預期來源的重合度 | > 80% |
| **回答準確率** | 生成的回答與預期答案的相似度 | > 70% |
| **平均延遲** | 從提問到回答完成的時間 | < 10s |
| **上下文數量** | 每次檢索返回的片段數 | 3-5 |

---

## 26-10 監控與維運

### 26-10-1 請求日誌

在 `app/main.py` 中加入中介軟體：

```python
import logging
import time
from fastapi import Request

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("data/rag_system.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """記錄所有請求"""
    start_time = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} | "
        f"狀態: {response.status_code} | "
        f"耗時: {elapsed:.2f}s"
    )

    return response
```

### 26-10-2 效能監控端點

```python
@app.get("/api/health")
async def health_check():
    """系統健康檢查"""
    import psutil

    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "vector_db_size": retriever.collection.count(),
    }
```

### 26-10-3 用 Docker Compose 部署

建立 `docker-compose.yml`：

```yaml
version: "3.8"

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    restart: unless-stopped

  meilisearch:
    image: getmeili/meilisearch:latest
    ports:
      - "7700:7700"
    volumes:
      - ./data/meilisearch:/meili_data
    environment:
      - MEILI_MASTER_KEY=masterKey
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
    restart: unless-stopped
```

---

## 26-11 完整工作流程示範

### 26-11-1 步驟一：準備文件

```bash
# 建立測試文件目錄
mkdir -p ~/rag-system/data/raw

# 放入你的文件（PDF、MD、TXT 等）
cp ~/documents/*.pdf ~/rag-system/data/raw/
cp ~/documents/*.md ~/rag-system/data/raw/
```

### 26-11-2 步驟二：匯入知識庫

```bash
cd ~/rag-system
source .venv/bin/activate

# 批次匯入
python scripts/ingest.py data/raw
```

### 26-11-3 步驟三：啟動服務

```bash
# 啟動 RAG API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 另一個終端機：啟動 Ollama（如果還沒啟動）
systemctl start ollama
```

### 26-11-4 步驟四：測試問答

```bash
# 用 curl 測試
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "DGX Spark 的 GPU 架構是什麼？",
    "top_k": 5
  }'
```

預期回應：

```json
{
  "answer": "DGX Spark 搭載的是 NVIDIA GB10 Grace Blackwell 超級晶片，\n其中 GPU 部分採用最新的 Blackwell 架構...",
  "model": "qwen3-8b",
  "context_count": 5,
  "sources": ["dgx-spark-specs.pdf", "hardware-overview.md"]
}
```

### 26-11-5 步驟五：連接 Open WebUI

在 Open WebUI 中添加 RAG 端點（如 26-8 節所述），然後就可以用漂亮的網頁介面跟你的知識庫對話了！

---

## 26-12 進階優化

### 26-12-1 查詢重寫（Query Rewriting）

使用者提問可能不夠精確，可以先重寫查詢再檢索：

```python
def rewrite_query(self, original_query: str) -> str:
    """用 LLM 重寫查詢，提升檢索準確率"""
    response = requests.post(
        f"{self.base_url}/api/chat",
        json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一個查詢重寫助手。請將使用者的查詢重寫為更適合檢索的形式。只輸出重寫後的查詢，不要輸出其他內容。"},
                {"role": "user", "content": f"重寫以下查詢：{original_query}"},
            ],
            "stream": False,
        }
    )
    return response.json()["message"]["content"]
```

### 26-12-2 自我反思（Self-RAG）

讓模型評估自己的回答品質：

```python
def self_reflect(self, question: str, answer: str, contexts: List[Dict]) -> Dict:
    """自我反思：評估回答品質"""
    response = requests.post(
        f"{self.base_url}/api/chat",
        json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": "請評估以下回答的品質。評分標準：1-5 分，5 分為最佳。評估維度：準確性、完整性、相關性。"},
                {"role": "user", "content": f"問題：{question}\n\n回答：{answer}\n\n請評分並說明原因。"},
            ],
            "stream": False,
        }
    )
    return response.json()["message"]["content"]
```

### 26-12-3 快取機制

對於常見問題，快取回答可以提升速度：

```python
import hashlib
import json
from pathlib import Path

class ResponseCache:
    """回答快取"""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        """生成快取鍵"""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict]:
        """獲取快取"""
        key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def set(self, query: str, response: Dict):
        """設定快取"""
        key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
```

---

## 26-13 常見問題與疑難排解

### 26-13-1 檢索結果不相關

**問題**：回答與問題無關。

**解決方案**：
1. 檢查嵌入模型是否正確載入
2. 增加 Top K（5 → 10）
3. 調整分塊大小（500 → 300）
4. 加入查詢重寫

### 26-13-2 匯入文件失敗

**問題**：`ingest.py` 執行時報錯。

**解決方案**：
```bash
# 確認 Ollama 正在執行
systemctl status ollama

# 確認嵌入模型已下載
ollama list | grep bge-m3

# 如果沒有，下載
ollama pull bge-m3
```

### 26-13-3 回應速度太慢

**問題**：從提問到回答超過 15 秒。

**解決方案**：
1. 減少 Top K（5 → 3）
2. 減少上下文長度
3. 換用更快的 LLM（qwen3-8b → qwen3-8b-coder）
4. 啟用回應快取

### 26-13-4 Meilisearch 中文分詞不準確

**問題**：關鍵字檢索結果不理想。

**解決方案**：
```bash
# Meilisearch 預設不支援中文分詞
# 需要使用支援中文的替代品，如：
# 1. Tantivy + jieba
# 2. Elasticsearch + IK 分詞器

# 或者在 Meilisearch 中設定同義詞
curl -X PUT 'http://localhost:7700/indexes/documents/settings/synonyms' \
  -H 'Authorization: Bearer masterKey' \
  -H 'Content-Type: application/json' \
  -d '{
    "人工智慧": ["AI", "機器學習"],
    "GPU": ["顯示卡", "圖形處理器"]
  }'
```

---

## 26-14 本章小結

::: success ✅ 你現在知道了
- 企業級 RAG 系統的完整架構：文件解析 → 分塊 → 嵌入 → 混合檢索 → 生成
- 混合檢索結合向量、關鍵字和知識圖譜，大幅提升準確率
- FastAPI 提供 RESTful API，可以輕鬆整合到現有系統
- Open WebUI 可以無縫連接 RAG 系統
- 評估指標幫助持續優化系統品質
- 查詢重寫、自我反思、快取等進階技巧進一步提升體驗
:::

::: tip 🎉 恭喜！
你已經完成了整本書的所有章節！從硬體認識到企業級 RAG 系統部署，你現在是 DGX Spark 的真正專家了！

👉 [回到首頁](/) | [查看附錄](/guide/appendix-a/)
:::

::: info 📝 上一章
← [回到第 25 章：多機互連與分散式運算](/guide/chapter25/)
:::
