# 第 21 章：RAG 與知識圖譜

::: tip 🎯 本章你將學到什麼
- RAG 檢索增強生成的完整流程
- 建立知識庫與調整檢索參數
- Multi-Agent Chatbot 多代理協作系統
- Text to Knowledge Graph 文字轉知識圖譜
- 知識圖譜的視覺化與查詢
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 21-1 RAG 檢索增強生成

### 21-1-1 什麼是 RAG？為什麼需要它？

::: info 🤔 RAG 是什麼？
RAG（Retrieval-Augmented Generation，檢索增強生成）解決了一個核心問題：

**AI 模型的知識只到訓練截止日期，而且不知道你公司的內部資料。**

RAG 的做法是：
1. 把你的文件切成小片段（chunk）
2. 把每個片段轉成向量（embedding）
3. 存入向量資料庫
4. 使用者提問時，先檢索最相關的片段
5. 把片段和問題一起送給 AI 生成回答
:::

```
使用者提問
  → 檢索系統找到相關文件片段
    → AI 根據文件片段回答
      → 回答有依據、可追溯
```

### 21-1-2 用 Open WebUI 建立知識庫

**方法一：在對話中上傳文件（最簡單）**

1. 打開 Open WebUI
2. 在對話視窗中，點擊輸入框旁的 📎 圖示
3. 選擇檔案（支援 PDF、TXT、MD、DOCX、CSV）
4. 上傳後，在對話中提問，AI 會自動參考文件內容

**方法二：建立可重複使用的知識庫**

1. 進入 **Admin Panel → Documents**
2. 點擊 **Upload** 上傳多個文件
3. 設定知識庫名稱和描述
4. 在對話中選擇要使用的知識庫

### 21-1-3 用 Docker 部署專屬 RAG 服務

如果你需要更強大的 RAG 功能：

```bash
# 部署 RAGFlow（開源 RAG 平台）
docker run -d \
  --name ragflow \
  --gpus all \
  --network host \
  -v ~/ragflow-data:/ragflow/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  infiniflow/ragflow:latest
```

打開 `http://DGX_Spark_IP:9380`，設定：
1. 建立知識庫
2. 上傳文件（支援批量上傳）
3. 設定解析規則（自動解析 PDF 表格、圖片等）
4. 開始對話

### 21-1-4 RAG 設定調校

**關鍵參數說明**：

| 參數 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| **Top K** | 檢索最相關的 K 個片段 | 3-5 | 太少可能遺漏資訊，太多會干擾 AI |
| **Chunk Size** | 每個片段的大小（字元數） | 500-1000 | 太小缺乏上下文，太大不夠精確 |
| **Chunk Overlap** | 片段之間的重疊比例 | 10-20% | 避免重要資訊被切斷 |
| **Similarity Threshold** | 相似度閾值 | 0.3-0.5 | 太低會檢索到不相關的內容 |

**Embedding 模型選擇**：

| 模型 | 語言 | 記憶體 | 品質 | 推薦場景 |
|------|------|--------|------|---------|
| BGE-M3 | 中文+英文 | ~2 GB | 最高 | 中文文件 |
| text-embedding-3-small | 多語言 | ~0.5 GB | 高 | 一般用途 |
| 第 19 章自訂 Embedding | 領域專用 | ~1 GB | **領域最佳** | 專業領域 |

### 21-1-5 RAG 問答實測

**測試文件**：DGX Spark 使用者手冊（PDF，50 頁）

```
問題：DGX Spark 的記憶體頻寬是多少？

沒有 RAG 的回答：
「DGX Spark 搭載 128GB 記憶體...」（模糊、可能不準確）

有 RAG 的回答：
「根據 DGX Spark 技術規格書第 12 頁，記憶體頻寬為 400 GB/s，
採用 LPDDR5x 規格，透過 NVLink-C2C 與 CPU/GPU 連接。」
```

::: tip 💡 RAG 的優勢
- 回答有依據，可以追溯到原始文件
- 不會「幻覺」（hallucination），因為有文件支撐
- 文件更新後，知識庫同步更新即可
:::

### 21-1-6 進階：混合檢索（Hybrid Search）

混合檢索結合了向量檢索和關鍵字檢索的優點：

```bash
# 部署支援混合檢索的 RAG 系統
docker run -d \
  --name hybrid-rag \
  --network host \
  -v ~/hybrid-rag-data:/data \
  ghcr.io/community/hybrid-rag:latest
```

混合檢索的效果：

| 檢索方式 | 準確率 | 召回率 | 適合場景 |
|---------|--------|--------|---------|
| 純向量 | 高 | 中 | 語意搜尋 |
| 純關鍵字 | 中 | 高 | 精確匹配 |
| **混合** | **最高** | **最高** | **所有場景** |

---

## 21-2 Multi-Agent Chatbot — 多代理協作系統

### 21-2-1 什麼是 Multi-Agent？

::: info 🤔 為什麼需要多個 Agent？
單一 AI 模型什麼都要做：理解問題、搜尋資料、分析、撰寫、審查。

Multi-Agent 的做法是分工：
- **研究員**：負責搜尋和整理資料
- **分析師**：負責分析和歸納
- **寫手**：負責撰寫報告
- **審稿人**：負責審查品質和準確性

就像一個團隊，每個人負責自己擅長的部分。
:::

### 21-2-2 下載模型

```bash
# 研究員用（需要強的理解能力）
ollama pull qwen3-8b

# 寫手用（需要好的文字表達能力）
ollama pull nemotron-nano
```

### 21-2-3 部署 Multi-Agent 服務

```bash
# 建立工作目錄
mkdir -p ~/multi-agent && cd ~/multi-agent

# 部署服務
docker run -d \
  --name multi-agent \
  --network host \
  -v ~/multi-agent-data:/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  -e AGENT_COUNT=3 \
  ghcr.io/community/multi-agent-chatbot:latest
```

### 21-2-4 設定 Agent 角色

在設定頁面中定義每個 Agent 的角色：

```yaml
# agent_config.yaml
agents:
  - name: "研究員"
    model: "qwen3-8b"
    role: "你是一個專業的研究員，負責搜尋和整理相關資料。
          請列出所有找到的資訊，並標註來源。"
    tools: ["search", "read_file"]

  - name: "分析師"
    model: "qwen3-8b"
    role: "你是一個數據分析師，負責分析研究員提供的資料。
          請歸納重點、找出趨勢、提出洞察。"
    tools: ["analyze"]

  - name: "寫手"
    model: "nemotron-nano"
    role: "你是一個專業寫手，負責把分析結果寫成易讀的報告。
          請使用清晰的結構和流暢的文字。"
    tools: ["write"]
```

### 21-2-5 多代理實測

**任務**：「幫我寫一份 DGX Spark 的市場分析報告」

```
使用者 → 研究員：
  「搜尋 DGX Spark 的市場資訊，包括競爭產品、價格、目標客群」

研究員 → 分析師：
  「找到以下資訊：
   1. DGX Spark 售價 $3,999
   2. 競爭產品：Mac Studio M3 Ultra、AMD Ryzen AI Max+ 395
   3. 目標客群：AI 開發者、研究人員、創作者」

分析師 → 寫手：
  「分析結果：
   1. DGX Spark 在 AI 推論領域有記憶體優勢
   2. 價格比 Mac Studio 便宜，但軟體生態系更強
   3. 市場定位明確：個人 AI 工作站」

寫手 → 使用者：
  「# DGX Spark 市場分析報告

   ## 產品概述
   DGX Spark 是 NVIDIA 推出的個人 AI 工作站...

   ## 競爭分析
   相較於競爭對手，DGX Spark 的優勢在於...

   ## 市場前景
   隨著 AI 應用普及，個人 AI 工作站市場預計...」
```

### 21-2-6 記憶體用量

| Agent 數量 | 模型大小 | 記憶體用量 | 回應時間 |
|-----------|---------|-----------|---------|
| 1 | 8B | ~16 GB | 快 |
| 3 | 8B | ~48 GB | 中等 |
| 5 | 8B | ~80 GB | 慢 |

::: tip 💡 建議
在 DGX Spark 上，建議同時運行 3-5 個 Agent，使用 8B 等級的模型。
如果需要更高品質，可以用 70B 模型但減少 Agent 數量。
:::

---

## 21-3 Text to Knowledge Graph — 文字轉知識圖譜

### 21-3-1 什麼是知識圖譜？

::: info 🤔 知識圖譜是什麼？
知識圖譜（Knowledge Graph）把非結構化的文字轉為結構化的「三元組」（Triples）：

```
（主詞, 關係, 受詞）
```

例如：
```
（DGX Spark, 搭載, GB10 超級晶片）
（GB10, 包含, 20 核 CPU）
（GB10, 包含, Blackwell GPU）
（DGX Spark, 記憶體, 128 GB）
```

這些三元組可以組成一張圖，讓你看到概念之間的關係。
:::

### 21-3-2 部署 Text2KG 服務

```bash
# 部署 Text2KG
docker run -d \
  --name text2kg \
  --gpus all \
  --network host \
  -v ~/kg-data:/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  ghcr.io/community/text2kg:latest
```

打開 `http://DGX_Spark_IP:7860`。

### 21-3-3 上傳文件與三元組抽取

1. 在 WebUI 中上傳文件（TXT、MD、PDF）
2. 選擇抽取模型（建議用 Qwen3-8B）
3. 點擊 **Extract** 開始抽取三元組

**抽取結果範例**（上傳 DGX Spark 規格書）：

```
(DGX Spark, 製造商, NVIDIA)
(DGX Spark, 搭載, GB10 超級晶片)
(GB10, CPU 核心數, 20)
(GB10, GPU 架構, Blackwell)
(DGX Spark, 記憶體容量, 128 GB)
(DGX Spark, 記憶體類型, LPDDR5x)
(DGX Spark, 記憶體頻寬, 273 GB/s)
(DGX Spark, 運算能力, 1 PFLOP FP4)
(DGX Spark, 售價, $3,999)
```

### 21-3-4 知識圖譜編輯與視覺化

在 WebUI 中：

1. **視覺化**：點擊 **Graph View**，用圖形方式查看知識圖譜
2. **編輯**：點擊任何節點或邊，可以修改或刪除
3. **新增**：手動新增三元組
4. **匯出**：匯出為 GraphML、JSON、CSV 格式

### 21-3-5 圖譜查詢與 ArangoDB

對於大型的知識圖譜，建議使用專業的圖資料庫：

```bash
# 部署 ArangoDB
docker run -d \
  --name arangodb \
  -p 8529:8529 \
  -e ARANGO_ROOT_PASSWORD=dgxspark \
  -v ~/arangodb-data:/var/lib/arangodb3 \
  arangodb:latest
```

打開 `http://DGX_Spark_IP:8529`，用 root / dgxspark 登入。

**AQL 查詢範例**：

```sql
-- 找出所有和 DGX Spark 直接相關的節點
FOR v, e IN 1..1 OUTBOUND 'products/dgx-spark' edges
  RETURN { node: v, relation: e.type }

-- 找出記憶體相關的所有資訊
FOR v, e IN 1..2 OUTBOUND 'products/dgx-spark' edges
  FILTER e.type == '記憶體' OR e.type == '搭載'
  RETURN v
```

### 21-3-6 知識圖譜 + RAG 的強大組合

把知識圖譜和 RAG 結合，可以實現更精準的問答：

```
使用者：DGX Spark 的 CPU 是什麼？

傳統 RAG：
  → 搜尋文件片段
    → AI 根據片段回答

知識圖譜 RAG：
  → 查詢圖譜：(DGX Spark, 搭載, ?) → (GB10)
  → 查詢圖譜：(GB10, CPU, ?) → (20 核 ARM Neoverse V2)
    → AI 根據圖譜結構化資料回答（更精準）
```

---

## 21-4 常見問題與疑難排解

### 21-4-1 RAG 回答品質不佳

**問題**：AI 的回答和文件內容不符。

**解決方案**：
1. 增加 Top K（3 → 5）
2. 增加 Chunk Size（500 → 1000）
3. 換用更好的 Embedding 模型（BGE-M3）
4. 檢查文件品質（掃描的 PDF 可能 OCR 不準確）

### 21-4-2 知識圖譜抽取不準確

**問題**：抽取的三元組有錯誤或遺漏。

**解決方案**：
1. 換用更大的模型（8B → 70B）
2. 調整抽取提示詞
3. 手動校對和修正
4. 分段抽取（長文件切成小段）

### 21-4-3 Multi-Agent 回應太慢

**問題**：多個 Agent 依序執行，總回應時間太長。

**解決方案**：
1. 減少 Agent 數量
2. 使用更小的模型
3. 讓部分 Agent 並行執行（研究員和分析師可以同時工作）

---

## 21-5 本章小結

::: success ✅ 你現在知道了
- RAG 讓 AI 能讀你的文件，回答有依據、可追溯
- 混合檢索結合向量和關鍵字，效果最佳
- Multi-Agent 讓多個 AI 分工，完成複雜任務
- 知識圖譜把非結構化文字轉為結構化的三元組
- 知識圖譜 + RAG 是最強大的問答組合
:::

::: tip 🚀 下一章預告
AI 不只是聊天和生成，還能「行動」！下一章來看看 AI Agent 和安全沙箱！

👉 [前往第 22 章：AI Agent 與安全沙箱 →](/guide/chapter22/)
:::

::: info 📝 上一章
← [回到第 20 章：多模態推論與即時視覺 AI](/guide/chapter20/)
:::
