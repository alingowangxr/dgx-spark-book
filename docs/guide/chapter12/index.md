# 第 12 章：NIM 推論微服務與引擎總比較

::: tip 🎯 本章你將學到什麼
- 申請 NGC 帳號和 API Key
- NIM 的特色和部署方式
- 七大推論引擎完整比較
- 選擇決策樹
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。
:::

---

## 12-0 申請 NGC 帳號

### 12-0-1 申請帳號

1. 前往 [NGC 網站](https://ngc.nvidia.com)
2. 點擊右上角 **Sign In**
3. 用 NVIDIA 帳號登入（沒有的話先註冊）
4. 完成註冊

### 12-0-2 申請 API Key

1. 登入 NGC 後，點擊右上角的帳號名稱
2. 選擇 **Setup → API Key**
3. 點擊 **Generate API Key**
4. 複製並妥善儲存（只會顯示一次！）

---

## 12-1 NIM 概觀

### 12-1-1 NIM 的特色

NIM（NVIDIA Inference Microservice）是 NVIDIA 推出的推論微服務。

它的特色：

| 特色 | 說明 |
|------|------|
| **一鍵部署** | 一個 Docker 指令就能啟動 |
| **NVIDIA 最佳化** | 針對每種 NVIDIA GPU 做了最佳化 |
| **企業級** | 支援認證、監控、自動擴展 |
| **OpenAI 相容** | 標準的 API 介面 |
| **持續更新** | NVIDIA 定期發布新版本 |

### 12-1-2 NIM vs. 自行部署

| | NIM | 自行部署（vLLM 等） |
|--|-----|-------------------|
| 部署難度 | **最簡單** | 中等 |
| 效能 | 最佳化 | 需自行調校 |
| 靈活性 | 有限 | 高 |
| 成本 | 免費（部分模型需授權） | 完全免費 |
| 更新 | NVIDIA 負責 | 自行負責 |
| 適合場景 | 快速部署、企業使用 | 研究、客製化 |

---

## 12-2 部署 NIM

### 12-2-1 確認環境

```bash
docker --version
nvidia-smi
```

### 12-2-2 設定 NGC 認證

```bash
# 登入 NGC Docker Registry
export NGC_API_KEY="你的API_KEY"
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin
```

### 12-2-3 啟動 NIM 容器

```bash
# 建立模型快取目錄
mkdir -p ~/.cache/nim

# 啟動 NIM（以 Llama 3.1 8B 為例）
docker run -d \
  --name nim-llm \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### 12-2-4 測試 API

```bash
# 等待模型載入完成（首次需要下載，約 5-10 分鐘）
curl http://localhost:8000/v1/health/ready

# 測試對話
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "stream": false
  }'
```

### 12-2-5 清理

```bash
docker stop nim-llm
docker rm nim-llm
docker rmi nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
rm -rf ~/.cache/nim
```

---

## 12-3 七大推論引擎總比較

### 12-3-1 功能比較表

| 特性 | Ollama | LM Studio | llama.cpp | vLLM | TRT-LLM | SGLang | NIM |
|------|--------|-----------|-----------|------|---------|--------|-----|
| **部署難度** | ⭐ 最易 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ 最易 |
| **GUI** | ❌ | ✅ | WebUI | ❌ | ❌ | ❌ | ❌ |
| **OpenAI API** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **PagedAttention** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **推測性解碼** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **多模態** | ✅ | ✅ | ✅ | 部分 | ✅ | 部分 | 部分 |
| **結構化生成** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **LoRA Serving** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **ARM64 支援** | ✅ | ✅ | ✅ | ✅ | 部分 | ✅ | 部分 |

### 12-3-2 速度比較

以下是在 DGX Spark 上的近似數據（8B 模型，BF16）：

| 引擎 | 輸出速度 (t/s) | 首次回應 (ms) | 記憶體用量 |
|------|---------------|--------------|-----------|
| Ollama | ~50 | ~200 | ~16 GB |
| LM Studio | ~45 | ~250 | ~16 GB |
| llama.cpp | ~55 | ~180 | ~16 GB |
| **vLLM** | ~60 | ~150 | ~16 GB |
| **TRT-LLM** | ~70 | ~120 | ~15 GB |
| **SGLang** | ~58 | ~160 | ~16 GB |
| **NIM** | ~65 | ~130 | ~15 GB |

::: info 🤔 為什麼差異不大？
對於 8B 這種小模型，瓶頸不在引擎，而在記憶體頻寬。引擎之間的差異在**大模型**和**高併發**場景才會明顯。

當同時服務 100 個使用者時，vLLM 和 TRT-LLM 的優勢會非常明顯。
:::

### 12-3-3 選擇決策樹

```
你想做什麼？
│
├─ 快速體驗、個人使用
│   └─ → Ollama（最簡單）
│
├─ 需要漂亮的網頁介面
│   └─ → Open WebUI + Ollama
│
├─ 追求最高吞吐量、多人服務
│   └─ → vLLM
│
├─ 追求極致效能、NVIDIA 生態系
│   └─ → TRT-LLM 或 NIM
│
├─ 需要結構化生成、RadixAttention
│   └─ → SGLang
│
├─ 最大彈性、離線推論
│   └─ → llama.cpp
│
└─ 需要 GUI + 進階參數調整
    └─ → LM Studio
```

---

## 12-4 疑難排解

### Q：NIM 容器啟動後一直無法就緒？

```bash
# 查看日誌
docker logs nim-llm

# 常見原因：
# 1. 模型下載中（首次需要時間）
# 2. NGC API Key 不正確
# 3. 記憶體不足
```

### Q：哪個引擎最適合 DGX Spark？

對於 DGX Spark 的個人使用場景：
- **日常使用**：Ollama（簡單、夠用）
- **需要高效能**：vLLM（社群支援好、彈性大）
- **想體驗 NVIDIA 最佳化**：NIM（一鍵部署）

---

## 12-5 本章小結

::: success ✅ 你現在知道了
- NIM 是 NVIDIA 官方的一鍵部署推論微服務
- 七大引擎各有優劣，沒有絕對的「最好」
- 個人使用選 Ollama，高效能選 vLLM，極致效能選 TRT-LLM
- 選擇時要考慮部署難度、功能需求、效能需求
:::

::: tip 🚀 第三篇完結！
恭喜！你已經完成了「LLM 推論進階」篇，掌握了七大推論引擎。

接下來我們要進入更有趣的部分 — 用 AI 生成圖片、影片、音樂和語音！

👉 [前往第 13 章：圖片與影片生成 →](/guide/chapter13/)
:::

::: info 📝 上一章
← [回到第 11 章：SGLang](/guide/chapter11/)
:::
