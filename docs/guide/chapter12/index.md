# 第 12 章：NIM 推論微服務與引擎總比較

::: tip 🎯 本章你將學到什麼
- 申請 NGC 帳號和 API Key 的完整流程
- NIM 的特色、架構和部署方式
- 七大推論引擎完整比較表
- 選擇決策樹和場景推薦
- 常見問題與疑難排解
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 12-0 申請 NGC 帳號

### 12-0-1 為什麼需要 NGC 帳號？

NVIDIA GPU Cloud（NGC）是 NVIDIA 的雲端服務平台，提供：

| 服務 | 說明 |
|------|------|
| **容器映像** | 最佳化的 AI/ML Docker 映像（如 TRT-LLM、NIM） |
| **預訓練模型** | NVIDIA 訓練好的模型，可直接使用 |
| **Helm Charts** | Kubernetes 部署模板 |
| **API** | 雲端推論 API（無需自建伺服器） |
| **NIM** | 推論微服務（本章重點） |

::: info 🤔 NGC 是免費的嗎？
是的！NGC 帳號是免費的。大部分容器和模型都可以免費使用。部分企業級功能（如 NIM 的某些模型）可能需要額外的授權。
:::

### 12-0-2 申請帳號的完整步驟

**步驟 1：前往 NGC 網站**

打開瀏覽器，前往 [https://ngc.nvidia.com](https://ngc.nvidia.com)

**步驟 2：註冊或登入**

1. 點擊右上角的 **Sign In**
2. 如果你已有 NVIDIA 帳號（例如買過 GeForce 顯示卡時註冊的），直接登入
3. 如果沒有，點擊 **Create Account** 註冊

**步驟 3：完成註冊**

| 欄位 | 說明 |
|------|------|
| 姓名 | 你的真實姓名 |
| 電子郵件 | 有效的 Email 地址 |
| 密碼 | 至少 8 個字元 |
| 組織 | 個人使用者可填「Personal」 |
| 用途 | 選擇「Research / Learning」 |

**步驟 4：驗證 Email**

NGC 會發送一封驗證信到你的 Email，點擊信中的連結完成驗證。

### 12-0-3 申請 API Key

API Key 是你存取 NGC 資源的憑證，用於 Docker 登入和 API 呼叫。

**步驟 1：進入 API Key 頁面**

1. 登入 NGC 後，點擊右上角的帳號名稱
2. 選擇 **Setup → API Key**

**步驟 2：生成 API Key**

1. 點擊 **Generate API Key**
2. 系統會產生一組新的 API Key
3. **立刻複製並儲存**（只會顯示一次！）

::: warning ⚠️ 重要提醒
API Key 只會顯示一次！請務必：
1. 複製到安全的地方（如密碼管理器）
2. 不要分享給他人
3. 不要提交到 Git 倉庫
4. 如果遺失，只能重新生成
:::

**步驟 3：測試 API Key**

```bash
# 用 API Key 登入 Docker Registry
export NGC_API_KEY="你的API_KEY"
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 成功會顯示：Login Succeeded
```

### 12-0-4 管理 API Key

| 操作 | 步驟 |
|------|------|
| 查看現有 Key | Setup → API Key |
| 生成新 Key | 點擊 Generate API Key |
| 撤銷 Key | 點擊 Key 旁邊的撤銷按鈕 |
| 忘記 Key | 只能生成新的，無法找回舊的 |

---

## 12-1 NIM 概觀

### 12-1-1 什麼是 NIM？

NIM（NVIDIA Inference Microservice）是 NVIDIA 推出的推論微服務。它把最佳化的 LLM 打包成一個 Docker 容器，讓你可以一鍵部署。

::: info 🤔 NIM 的核心理念
想像你去麥當勞點餐：
- **自行部署（vLLM 等）** = 自己買食材、自己煮
- **NIM** = 直接點套餐，開箱即食

NIM 的好處是：NVIDIA 已經幫你做了所有最佳化工作，你只需要一個 Docker 指令就能啟動。
:::

**NIM 的技術架構：**

```
┌─────────────────────────────────────┐
│         用戶端                        │
│   Open WebUI / curl / Python SDK     │
├─────────────────────────────────────┤
│         OpenAI 相容 API              │
│   /v1/chat/completions               │
│   /v1/completions                    │
│   /v1/embeddings                     │
├─────────────────────────────────────┤
│         NIM 容器                      │
│   • 內建 TensorRT-LLM 引擎           │
│   • 針對特定 GPU 最佳化               │
│   • 自動下載和快取模型                │
│   • 健康檢查和監控                    │
├─────────────────────────────────────┤
│         NVIDIA GPU                   │
└─────────────────────────────────────┘
```

### 12-1-2 NIM 的特色

| 特色 | 說明 | 好處 |
|------|------|------|
| **一鍵部署** | 一個 Docker 指令就能啟動 | 零設定成本 |
| **NVIDIA 最佳化** | 針對每種 NVIDIA GPU 做了最佳化 | 開箱即得最佳效能 |
| **企業級** | 支援認證、監控、自動擴展 | 適合生產環境 |
| **OpenAI 相容** | 標準的 API 介面 | 無縫對接現有應用 |
| **持續更新** | NVIDIA 定期發布新版本 | 自動獲得效能改進 |
| **模型快取** | 自動下載並快取模型 | 重啟不需要重新下載 |
| **健康檢查** | 內建健康檢查端點 | 方便監控和自動擴展 |

### 12-1-3 NIM 支援的模型

截至 2025 年，NIM 支援以下模型家族：

| 模型家族 | 代表模型 | 用途 |
|---------|---------|------|
| **Llama** | Llama 3.1 8B/70B/405B | 通用對話 |
| **Llama** | Llama 3.3 70B | 通用對話 |
| **Mistral** | Mistral 7B / Mixtral 8x7B | 輕量高效 |
| **Mistral** | Mistral Large | 高品質 |
| **Qwen** | Qwen2.5-72B | 中文最佳 |
| **Gemma** | Gemma 2 9B/27B | Google 開源 |
| **Nemotron** | Nemotron-3-Super | NVIDIA 自家模型 |
| **DeepSeek** | DeepSeek-R1 | 推理能力強 |
| **Embedding** | NV-EmbedQA-4 | 向量嵌入 |

::: info 🤔 NIM 的命名規則
NIM 容器名稱遵循以下規則：

```
nvcr.io/nim/<組織>/<模型名稱>:<版本>

例如：
nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
         ^^^  ^^^^  ^^^^^^^^^^^^^^^^^^^^  ^^^^^^
          Registry  組織    模型名稱       版本
```
:::

### 12-1-4 NIM vs. 自行部署

| | NIM | 自行部署（vLLM 等） |
|--|-----|-------------------|
| **部署難度** | **最簡單**（一行指令） | 中等（需要設定參數） |
| **效能** | 最佳化（NVIDIA 預先調校） | 需自行調校 |
| **靈活性** | 有限（固定配置） | 高（自由調整） |
| **模型選擇** | 有限（NIM 支援的模型） | 廣泛（任何 Hugging Face 模型） |
| **成本** | 免費（部分模型需授權） | 完全免費 |
| **更新** | NVIDIA 負責 | 自行負責 |
| **監控** | 內建健康檢查 | 需自行設定 |
| **適合場景** | 快速部署、企業使用 | 研究、客製化需求 |

---

## 12-2 部署 NIM

### 12-2-1 確認環境

```bash
# 檢查 Docker 版本
docker --version
# 建議 24.0+

# 檢查 GPU 狀態
nvidia-smi
# 確認 GPU 可用，驅動版本正確

# 測試 Docker GPU 支援
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# 應該顯示與主機相同的 GPU 資訊
```

### 12-2-2 設定 NGC 認證

```bash
# 設定環境變數
export NGC_API_KEY="你的API_KEY"

# 登入 NGC Docker Registry
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 預期輸出：Login Succeeded
```

::: tip 💡 永久儲存 NGC 認證
如果你不想每次都重新登入，可以把認證儲存到 Docker config：

```bash
# 登入後，認證會自動儲存到 ~/.docker/config.json
# 下次執行 docker pull 時會自動使用

# 查看已儲存的認證
cat ~/.docker/config.json | grep nvcr.io
```
:::

### 12-2-3 啟動 NIM 容器

**以 Llama 3.1 8B 為例：**

```bash
# 建立模型快取目錄
mkdir -p ~/.cache/nim

# 啟動 NIM
docker run -d \
  --name nim-llm \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

**參數解釋：**

| 參數 | 說明 | 為什麼需要 |
|------|------|-----------|
| `-d` | 背景執行 | 讓終端機可以繼續使用 |
| `--name nim-llm` | 容器名稱 | 方便後續管理 |
| `--gpus all` | 使用所有 GPU | DGX Spark 只有一顆 |
| `--network host` | 使用主機網路 | 避免 port 映射 |
| `--shm-size=16g` | 共享記憶體 | 多行程通訊需要 |
| `-v ~/.cache/nim:/opt/nim/.cache` | 模型快取 | 重啟不需要重新下載 |
| `-e NGC_API_KEY` | 傳遞 API Key | NIM 需要驗證授權 |

### 12-2-4 等待模型下載

首次啟動 NIM 時，它會自動下載模型。這需要一些時間：

```bash
# 查看下載進度
docker logs -f nim-llm

# 預期輸出：
# Downloading model...
# Download progress: 25%
# Download progress: 50%
# Download progress: 75%
# Download progress: 100%
# Model loaded successfully.
# Server is ready.
```

**不同模型的下載時間：**

| 模型 | 大小 | 下載時間（100 Mbps） | 載入時間 |
|------|------|-------------------|---------|
| Llama 3.1 8B | ~16 GB | ~15 分鐘 | ~2 分鐘 |
| Llama 3.1 70B | ~140 GB | ~2 小時 | ~5 分鐘 |
| Qwen2.5-72B | ~140 GB | ~2 小時 | ~5 分鐘 |
| Mistral 7B | ~14 GB | ~12 分鐘 | ~1 分鐘 |

::: tip 💡 預先下載模型
如果你不想在啟動時等待，可以先下載模型：

```bash
# 使用 NIM 的預下載功能
docker run --rm \
  --gpus all \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
  --download-only
```
:::

### 12-2-5 測試 API

```bash
# 步驟 1：健康檢查
curl http://localhost:8000/v1/health/ready
# 預期回應：{"status": "ready"}

# 步驟 2：列出可用模型
curl http://localhost:8000/v1/models
# 預期回應：{"data": [{"id": "meta/llama-3.1-8b-instruct", ...}]}

# 步驟 3：測試對話
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "system", "content": "你是一個有幫助的中文助手。"},
      {"role": "user", "content": "你好！請用三句話介紹你自己。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
  }'
```

**預期回應：**

```json
{
  "id": "chatcmpl-nim-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "meta/llama-3.1-8b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！我是..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 30,
    "completion_tokens": 50,
    "total_tokens": 80
  }
}
```

### 12-2-6 串流模式

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "請寫一首關於春天的詩。"}
    ],
    "stream": true,
    "max_tokens": 300
  }'
```

**預期輸出（Server-Sent Events）：**

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"春"},"index":0}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"風"},"index":0}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"拂"},"index":0}]}

...

data: [DONE]
```

### 12-2-7 用 Python 測試

```python
from openai import OpenAI

# 建立客戶端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # NIM 本地部署不需要
)

# 基本對話
response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {"role": "system", "content": "你是一個有幫助的助手。"},
        {"role": "user", "content": "解釋什麼是量子計算？"}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"\nToken 用量: {response.usage}")

# 串流模式
stream = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "寫一個 Python 的快速排序。"}
    ],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 12-2-8 部署其他 NIM 模型

**部署 Qwen2.5-72B：**

```bash
docker run -d \
  --name nim-qwen \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/qwen/qwen2.5-72b-instruct:latest
```

**部署 Mistral 7B：**

```bash
docker run -d \
  --name nim-mistral \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/mistralai/mistral-7b-instruct-v0.3:latest
```

---

## 12-3 七大推論引擎總比較

### 12-3-1 七大引擎簡介

在本書中，我們介紹了以下七種推論引擎：

| # | 引擎 | 開發者 | 核心特色 |
|---|------|--------|---------|
| 1 | **Ollama** | Ollama 團隊 | 最簡單的一行指令部署 |
| 2 | **LM Studio** | LM Studio 團隊 | 漂亮的 GUI，適合初學者 |
| 3 | **llama.cpp** | Georgi Gerganov | 最大彈性，支援最多硬體 |
| 4 | **vLLM** | UC Berkeley | PagedAttention + Continuous Batching |
| 5 | **TRT-LLM** | NVIDIA | 編譯期最佳化，極致效能 |
| 6 | **SGLang** | UC Berkeley | RadixAttention + 結構化生成 |
| 7 | **NIM** | NVIDIA | 一鍵部署，企業級微服務 |

### 12-3-2 功能比較表

| 特性 | Ollama | LM Studio | llama.cpp | vLLM | TRT-LLM | SGLang | NIM |
|------|--------|-----------|-----------|------|---------|--------|-----|
| **部署難度** | ⭐ 最易 | ⭐⭐ 簡單 | ⭐⭐⭐ 中等 | ⭐⭐ 中等 | ⭐⭐⭐ 較難 | ⭐⭐ 中等 | ⭐ 最易 |
| **GUI** | ❌ | ✅ 內建 | ❌（需 WebUI） | ❌（需 WebUI） | ❌ | ❌ | ❌ |
| **OpenAI API** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **PagedAttention** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅（Radix） | ✅ |
| **推測性解碼** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **多模態** | ✅ | ✅ | ✅ | 部分 | ✅ | 部分 | 部分 |
| **結構化生成** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ 原生 | ✅ |
| **LoRA Serving** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **ARM64 支援** | ✅ | ✅ | ✅ | ✅ | 部分 | ✅ | 部分 |
| **模型選擇** | 廣泛 | 廣泛 | 最廣泛 | 廣泛 | 有限 | 廣泛 | 有限 |
| **高併發** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **監控指標** | ❌ | ❌ | ❌ | ✅ Prometheus | ✅ Prometheus | ✅ 內建 | ✅ 內建 |

### 12-3-3 速度比較

以下是在 DGX Spark 上的近似數據（8B 模型，BF16）：

| 引擎 | 輸出速度 (t/s) | 首次回應 (ms) | 記憶體用量 | 100 併發吞吐量 |
|------|---------------|--------------|-----------|---------------|
| Ollama | ~50 | ~200 | ~16 GB | ~50 t/s（無法併發） |
| LM Studio | ~45 | ~250 | ~16 GB | ~45 t/s（無法併發） |
| llama.cpp | ~55 | ~180 | ~16 GB | ~55 t/s（無法併發） |
| **vLLM** | ~60 | ~150 | ~16 GB | ~500 t/s |
| **TRT-LLM** | ~70 | ~120 | ~15 GB | ~600 t/s |
| **SGLang** | ~58 | ~160 | ~16 GB | ~480 t/s |
| **NIM** | ~65 | ~130 | ~15 GB | ~550 t/s |

::: info 🤔 為什麼單使用者時差異不大？
對於 8B 這種小模型，瓶頸不在引擎，而在**記憶體頻寬**。

DGX Spark 的記憶體頻寬約 400 GB/s，8B BF16 模型大小約 16 GB。每次生成 token 需要讀取整個模型，所以理論上限約為：

```
400 GB/s ÷ 16 GB ≈ 25 次/秒
```

但實際上有各種最佳化（如 KV cache 不需要每次都讀取），所以可以達到 50-70 t/s。

引擎之間的差異在以下場景才會明顯：
1. **大模型**（120B+）：最佳化差異被放大
2. **高併發**（多使用者）：Continuous Batching 的優勢
3. **長上下文**：PagedAttention 的記憶體管理優勢
:::

### 12-3-4 大模型速度比較（120B NVFP4）

| 引擎 | 輸出速度 (t/s) | 記憶體用量 | 備註 |
|------|---------------|-----------|------|
| vLLM | ~12 | ~65 GB | 穩定，社群支援好 |
| TRT-LLM | ~14 | ~62 GB | 最快，但需要編譯 engine |
| SGLang | ~11 | ~65 GB | RadixAttention 加速多輪對話 |
| NIM | ~13 | ~63 GB | 一鍵部署，效能接近 TRT-LLM |
| Ollama | ~10 | ~70 GB | 記憶體用量較高 |
| llama.cpp | ~11 | ~68 GB | 需要 GGUF 格式 |

### 12-3-5 易用性評分

| 引擎 | 安裝 | 部署 | 調校 | 維護 | 總分 |
|------|------|------|------|------|------|
| Ollama | 10/10 | 10/10 | 7/10 | 9/10 | **9.0** |
| NIM | 8/10 | 10/10 | 6/10 | 9/10 | **8.3** |
| LM Studio | 9/10 | 9/10 | 8/10 | 8/10 | **8.5** |
| llama.cpp | 6/10 | 7/10 | 8/10 | 7/10 | **7.0** |
| vLLM | 7/10 | 8/10 | 9/10 | 8/10 | **8.0** |
| SGLang | 7/10 | 8/10 | 9/10 | 7/10 | **7.8** |
| TRT-LLM | 5/10 | 6/10 | 7/10 | 6/10 | **6.0** |

### 12-3-6 選擇決策樹

```
你想做什麼？
│
├─ 快速體驗、個人使用
│   │
│   ├─ 想要最簡單的方式
│   │   └─ → Ollama（一行指令搞定）
│   │
│   └─ 想要有圖形介面
│       └─ → LM Studio（漂亮的 GUI）
│
├─ 需要高效的 API 服務
│   │
│   ├─ 追求最高吞吐量、多人服務
│   │   └─ → vLLM（社群支援好、彈性大）
│   │
│   ├─ 追求極致效能
│   │   └─ → TRT-LLM（編譯期最佳化）
│   │
│   └─ 想要一鍵部署
│       └─ → NIM（NVIDIA 官方最佳化）
│
├─ 需要特殊功能
│   │
│   ├─ 結構化生成（JSON、regex）
│   │   └─ → SGLang（原生支援）
│   │
│   ├─ RadixAttention 加速多輪對話
│   │   └─ → SGLang（獨家技術）
│   │
│   └─ 最大彈性、離線推論
│       └─ → llama.cpp（支援最多硬體和格式）
│
└─ 企業級需求
    │
    ├─ 需要監控、自動擴展
    │   └─ → NIM 或 TRT-LLM
    │
    └─ 需要 LoRA 多任務
        └─ → vLLM
```

### 12-3-7 場景推薦

| 場景 | 推薦引擎 | 原因 |
|------|---------|------|
| **個人日常使用** | Ollama | 簡單、夠用、社群活躍 |
| **開發者測試** | vLLM | 彈性大、API 完善 |
| **RAG 系統** | SGLang | RadixAttention 加速相同前綴 |
| **結構化資料提取** | SGLang | 原生 regex/JSON 支援 |
| **高併發 API 服務** | vLLM | Continuous Batching + 監控 |
| **極致效能** | TRT-LLM | 編譯期最佳化 |
| **快速部署** | NIM | 一鍵啟動 |
| **離線推論** | llama.cpp | 不依賴 Docker |
| **多模型同時服務** | vLLM | LoRA Serving |
| **企業生產環境** | NIM / TRT-LLM | 企業級支援 |

---

## 12-4 整合建議

### 12-4-1 推薦的 DGX Spark 配置

對於 DGX Spark 的個人使用場景，我們推薦以下配置：

**日常使用配置：**

```
Ollama（日常對話） + Open WebUI（介面）
│
└─ 簡單、快速、夠用
```

**高效能配置：**

```
vLLM（主力推論） + Open WebUI（介面）
│
├─ 部署 Qwen3.5-122B NVFP4（高品質）
├─ 部署 Qwen3.5-35B-A3B NVFP4（高速度）
└─ 啟用 Prefix Caching + Chunked Prefill
```

**極致效能配置：**

```
NIM（一鍵部署） + Open WebUI（介面）
│
└─ NVIDIA 官方最佳化，效能接近 TRT-LLM
```

### 12-4-2 多引擎共存

你可以在同一台 DGX Spark 上同時運行多個引擎：

```bash
# Ollama（port 11434）
ollama serve &

# vLLM（port 8000）
docker run -d --name vllm --gpus all --network host ... --port 8000

# SGLang（port 30000）
docker run -d --name sglang --gpus all --network host ... --port 30000

# NIM（port 8001）
docker run -d --name nim --gpus all --network host ... --port 8001
```

::: warning ⚠️ 注意記憶體
同時運行多個引擎會佔用更多記憶體。確保：
1. 每個引擎的 `gpu-memory-utilization` 或 `mem-fraction-static` 總和不超過 1.0
2. 系統記憶體（RAM）足夠支援所有容器
3. 用 `nvidia-smi` 和 `docker stats` 監控資源用量
:::

---

## 12-5 疑難排解 FAQ

### Q1：NIM 容器啟動後一直無法就緒？

```bash
# 步驟 1：查看日誌
docker logs -f nim-llm

# 步驟 2：檢查常見問題
# 問題 A：模型下載中
# 解決：等待下載完成（查看日誌中的進度）

# 問題 B：NGC API Key 不正確
# 解決：重新登入
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 問題 C：記憶體不足
# 解決：檢查 GPU 記憶體
nvidia-smi
```

### Q2：哪個引擎最適合 DGX Spark？

對於 DGX Spark 的個人使用場景：

| 需求 | 推薦 | 原因 |
|------|------|------|
| **日常使用** | Ollama | 簡單、夠用、社群活躍 |
| **需要高效能** | vLLM | 社群支援好、彈性大、高併發 |
| **想體驗 NVIDIA 最佳化** | NIM | 一鍵部署、效能接近 TRT-LLM |
| **需要結構化生成** | SGLang | 原生 JSON/regex 支援 |
| **追求極致效能** | TRT-LLM | 編譯期最佳化 |

### Q3：如何選擇量化格式？

| 你的情況 | 推薦格式 | 原因 |
|---------|---------|------|
| DGX Spark（Blackwell） | NVFP4 | 原生最佳化，速度最快 |
| 需要跨平台 | MXFP4 | 相容性最佳 |
| 需要最高品質 | FP8 | 精度損失最小 |
| 通用場景 | INT4（GGUF） | 最廣泛支援 |

### Q4：如何監控引擎效能？

```bash
# vLLM
curl http://localhost:8000/metrics

# SGLang
curl http://localhost:30000/metrics

# NIM / TRT-LLM
curl http://localhost:8002/metrics

# GPU 用量
watch -n 1 nvidia-smi

# 容器資源用量
docker stats
```

### Q5：引擎效能不如預期？

**檢查清單：**

| 檢查項目 | 指令 | 正常值 |
|---------|------|--------|
| GPU 利用率 | `nvidia-smi` | > 80% |
| GPU 記憶體 | `nvidia-smi` | 接近設定值 |
| 溫度 | `nvidia-smi` | < 85°C |
| 網路 | `ping ngc.nvidia.com` | < 100ms |
| 磁碟空間 | `df -h` | > 50GB 可用 |

**常見效能瓶頸：**

| 瓶頸 | 症狀 | 解決方案 |
|------|------|---------|
| CPU 瓶頸 | GPU 利用率低 | 減少 batch size |
| 記憶體瓶頸 | OOM 錯誤 | 降低 gpu-memory-utilization |
| 散熱瓶頸 | GPU 降頻 | 改善散熱 |
| 磁碟瓶頸 | 模型載入慢 | 使用 SSD |

---

## 12-6 本章小結

::: success ✅ 你現在知道了
- NIM 是 NVIDIA 官方的一鍵部署推論微服務，結合了 TRT-LLM 的效能和易用的部署方式
- NIM 需要 NGC 帳號和 API Key，帳號免費申請
- NIM 自動下載和快取模型，重啟不需要重新下載
- 七大引擎各有優劣，沒有絕對的「最好」：
  - 個人使用選 Ollama（簡單）
  - 高效能選 vLLM（彈性大、社群好）
  - 極致效能選 TRT-LLM（編譯期最佳化）
  - 一鍵部署選 NIM（NVIDIA 官方）
  - 結構化生成選 SGLang（獨家功能）
- 選擇時要考慮：部署難度、功能需求、效能需求、維護成本
- 多個引擎可以在同一台 DGX Spark 上共存，但要注意記憶體分配
- 對於小模型（8B），引擎之間的差異不大；對於大模型（120B+）和高併發場景，差異會非常明顯
:::

::: tip 🚀 第三篇完結！
恭喜！你已經完成了「LLM 推論進階」篇，掌握了七大推論引擎的部署、調校和比較。

現在你已經具備了在 DGX Spark 上運行任何大型語言模型的能力。不管是 8B 的小模型還是 120B 的超大模型，你都知道如何選擇最適合的引擎並進行最佳化。

接下來我們要進入更有趣的部分 — 用 AI 生成圖片、影片、音樂和語音！

👉 [前往第 13 章：圖片與影片生成 →](/guide/chapter13/)
:::

::: info 📝 上一章
← [回到第 11 章：SGLang](/guide/chapter11/)
:::
