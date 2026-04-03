# 第 21 章：RAG 與知識圖譜

::: tip 🎯 本章你將學到什麼
- RAG 檢索增強生成
- Multi-Agent Chatbot 多代理協作
- Text to Knowledge Graph 文字轉知識圖譜
:::

---

## 21-1 RAG 檢索增強生成

### 21-1-1 建立知識庫

RAG（Retrieval-Augmented Generation）讓 AI 能讀你的文件來回答問題。

在 Open WebUI 中：
1. **Admin Panel → Documents**
2. 上傳你的文件（PDF、TXT、MD）
3. 系統自動建立向量索引

### 21-1-2 選擇模型與 RAG 問答

選擇一個模型，然後在對話中提問。AI 會自動檢索相關文件並回答。

### 21-1-3 RAG 設定調整

| 設定 | 說明 | 建議 |
|------|------|------|
| Top K | 檢索片段數 | 3-5 |
| Chunk Size | 片段大小 | 500-1000 |
| Embedding Model | 嵌入模型 | 第 19 章自訂的領域 Embedding |

---

## 21-2 Multi-Agent Chatbot — 多代理協作系統

### 21-2-1 下載模型

```bash
ollama pull qwen3-8b
ollama pull nemotron-nano
```

### 21-2-2 啟動服務

```bash
docker run -d \
  --name multi-agent \
  --network host \
  -e OLLAMA_HOST=http://localhost:11434 \
  ghcr.io/community/multi-agent-chatbot:latest
```

### 21-2-3 多代理實測

設定不同 Agent 的角色：
- **研究員**：負責搜尋和分析
- **寫手**：負責撰寫
- **審稿人**：負責審查品質

---

## 21-3 Text to Knowledge Graph — 文字轉知識圖譜

### 21-3-1 部署 Text2KG

```bash
docker run -d \
  --name text2kg \
  --gpus all \
  --network host \
  -v ~/kg-data:/data \
  ghcr.io/community/text2kg:latest
```

### 21-3-2 上傳文件與三元組抽取

打開 `http://DGX_Spark_IP:7860`，上傳文件，系統自動抽取（主詞, 關係, 受詞）三元組。

### 21-3-3 知識圖譜編輯與視覺化

在 WebUI 中可以：
- 編輯三元組
- 視覺化圖譜
- 匯出為 GraphML

### 21-3-4 圖譜查詢與 ArangoDB

```bash
# 部署 ArangoDB
docker run -d \
  --name arangodb \
  -p 8529:8529 \
  -e ARANGO_ROOT_PASSWORD=secret \
  arangodb:latest
```

---

## 21-4 本章小結

::: success ✅ 你現在知道了
- RAG 讓 AI 能讀你的文件
- Multi-Agent 讓多個 AI 協作完成複雜任務
- 知識圖譜把非結構化文字轉為結構化知識
:::

::: tip 🚀 下一章預告
AI 不只是聊天和生成，還能「行動」！下一章來看看 AI Agent 和安全沙箱！

👉 [前往第 22 章：AI Agent 與安全沙箱 →](/guide/chapter22/)
:::

::: info 📝 上一章
← [回到第 20 章：多模態推論](/guide/chapter20/)
:::
