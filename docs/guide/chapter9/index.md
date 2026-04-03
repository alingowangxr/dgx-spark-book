# 第 9 章：vLLM — 高吞吐量推論伺服器

::: tip 🎯 本章你將學到什麼
- vLLM 的核心技術：PagedAttention 和 Continuous Batching
- 部署 NVFP4、MXFP4、FP8 量化模型
- 用 Claude Code 部署 vLLM 容器
- 效能調校：GPU 記憶體利用率、Chunked Prefill
- vLLM vs. Ollama vs. llama.cpp 比較
- 網路傳言 91t/s 的真相
:::

::: warning ⏱️ 預計閱讀時間
約 30 分鐘。
:::

---

## 9-1 vLLM 架構與特色

### 9-1-1 什麼是 vLLM？

vLLM 是由加州大學柏克萊分校（UC Berkeley）研究團隊開發的開源 LLM 推論伺服器。它的目標很簡單：**讓大型語言模型的推論速度更快、吞吐量更高**。

::: info 🤔 為什麼需要專門的推論伺服器？
你可能想問：「我用 Python 直接跑模型不就好了嗎？」

答案是：**可以，但效率很差**。

想像一家餐廳：
- **直接跑模型** = 一個廚師一次只做一道菜，做完才做下一道
- **vLLM** = 一個專業廚房，多位廚師同時處理多道菜，食材管理井然有序

當你需要同時服務多個使用者時，vLLM 的優勢會非常明顯。它可以同時處理數十甚至數百個請求，而直接跑模型一次只能處理一個。
:::

vLLM 的核心創新有兩大技術：

| 技術 | 解決的問題 | 效果 |
|------|-----------|------|
| **PagedAttention** | KV cache 記憶體碎片化 | 記憶體浪費從 60-80% 降到 4% 以下 |
| **Continuous Batching** | 傳統 batching 的等待問題 | 吞吐量提升 2-4 倍 |

### 9-1-2 PagedAttention 深入解析

::: info 🤔 什麼是 PagedAttention？
想像你在讀一本很長的書。傳統的做法是把整本書攤在桌子上（預先分配所有記憶體），但這樣很浪費空間。

PagedAttention 的做法是：只攤開你正在讀的那幾頁，其他頁放在書架上，需要時再翻。這樣可以大幅減少記憶體浪費。

技術上來說，PagedAttention 把 KV cache 分成固定大小的「頁面」（page），像作業系統的虛擬記憶體一樣動態管理。
:::

讓我們用更具體的方式理解：

**傳統 KV Cache 管理方式的問題：**

```
請求 A 需要 1000 token 的上下文 → 預先分配 2000 token 空間（浪費 50%）
請求 B 需要 500 token 的上下文  → 預先分配 2000 token 空間（浪費 75%）
請求 C 需要 1500 token 的上下文 → 預先分配 2000 token 空間（浪費 25%）

總浪費：(1000+1500+500) / 6000 = 50% 記憶體被浪費！
```

**PagedAttention 的做法：**

```
頁面大小 = 16 tokens

請求 A: [Page1][Page2][Page3]...[Page63]  ← 用多少分配多少
請求 B: [Page1][Page2]...[Page32]
請求 C: [Page1][Page2]...[Page94]

記憶體浪費 < 4%（只有最後一頁可能沒填滿）
```

**PagedAttention 的關鍵優勢：**

| 優勢 | 說明 | 實際影響 |
|------|------|---------|
| 零外部碎片 | 頁面可以分散在記憶體各處 | 不會因為找不到連續空間而失敗 |
| 零內部碎片 | 只有最後一頁可能未滿 | 浪費 < 4% |
| 動態擴充 | 需要時才分配新頁面 | 支援更長的上下文 |
| 頁面共享 | 多個請求可以共享相同頁面 | 適合 system prompt 共享 |

### 9-1-3 Continuous Batching 深入解析

傳統 batching 是等所有請求都處理完才輸出結果。Continuous Batching 則是：

```
傳統 Batching：
請求 A: [===========] ← 等 A 完成才處理 B
請求 B:             [===========]
請求 C:                           [===========]
總時間：3 × 單次時間

Continuous Batching：
請求 A: [=======] ← A 完成就輸出
請求 B:   [===========] ← B 可以插隊
請求 C:     [===] ← C 也可以插隊
總時間：≈ 最長請求的時間
```

::: info 🤔 為什麼 Continuous Batching 更快？
LLM 推論有兩個階段：
1. **Prefill（預填充）**：處理輸入 prompt，計算量大但可以做 parallel
2. **Decoding（解碼）**：逐 token 生成輸出，計算量小但必須 sequential

在傳統 batching 中，如果請求 A 需要生成 100 個 token，請求 B 需要生成 10 個 token，B 必須等 A 完成。但在 Continuous Batching 中，B 完成後立刻輸出，空出的位置可以讓新請求 C 進來。

這就像高速公路的車輛：傳統 batching 是所有車必須同時到達終點才能下交流道；Continuous Batching 是每輛車到達自己的出口就可以離開。
:::

**效果對比：**

| 指標 | 傳統 Batching | Continuous Batching | 提升倍數 |
|------|--------------|---------------------|---------|
| 吞吐量 | 基準 | 2-4x | 2-4 倍 |
| 平均延遲 | 高 | 低 | 降低 50-70% |
| GPU 利用率 | 40-60% | 80-95% | 提升 50%+ |
| 同時服務使用者數 | 有限 | 大幅增加 | 3-5 倍 |

### 9-1-4 為什麼 vLLM 適合 DGX Spark

| 特性 | 為什麼適合 DGX Spark |
|------|---------------------|
| 高吞吐量 | 128GB 記憶體可以批次處理更多請求 |
| PagedAttention | 減少記憶體浪費，裝更大的模型 |
| Continuous Batching | 充分利用 GPU 算力 |
| 多模型支援 | 支援 NVFP4、FP8 等 Blackwell 原生格式 |
| ARM64 相容 | DGX Spark 使用 ARM64 架構，vLLM 完整支援 |
| OpenAI 相容 API | 無縫對接各種前端應用 |

### 9-1-5 vLLM 的進階功能

除了核心的 PagedAttention 和 Continuous Batching，vLLM 還提供以下進階功能：

| 功能 | 說明 | 適合場景 |
|------|------|---------|
| **Speculative Decoding** | 用小模型預測大模型的輸出，加速推論 | 需要高速輸出的場景 |
| **Prefix Caching** | 快取常用前綴（如 system prompt），避免重複計算 | RAG、多輪對話 |
| **LoRA Serving** | 同時服務多個 LoRA adapter，無需切換模型 | 多任務、多領域應用 |
| **Logits Processor** | 自訂輸出行為（如禁止特定詞、強制格式） | 結構化生成、安全過濾 |
| **Metrics** | Prometheus 監控指標，可整合 Grafana | 生產環境監控 |
| **Chunked Prefill** | 把長輸入分塊處理，避免 OOM | 長文處理 |
| **Tensor Parallelism** | 多 GPU 分割模型 | 超大模型部署 |
| **Pipeline Parallelism** | 多 GPU 流水線處理 | 多節點部署 |

::: tip 💡 DGX Spark 只有一顆 GPU
DGX Spark 是單 GPU 設備，所以 Tensor Parallelism 和 Pipeline Parallelism 在這裡用不到。但其他功能如 Prefix Caching、LoRA Serving、Speculative Decoding 都非常實用！
:::

---

## 9-2 支援的模型

### 9-2-1 量化格式全解析

在深入模型之前，我們先了解 vLLM 支援的量化格式。量化是把模型的權重從高精度轉換為低精度，以減少記憶體用量和加速推論。

| 格式 | 位元數 | 記憶體用量（相對於 BF16） | 精度損失 | 硬體支援 |
|------|--------|------------------------|---------|---------|
| **BF16** | 16-bit | 1x（基準） | 無 | 所有 GPU |
| **FP8** | 8-bit | 約 1/2 | 極小 | Hopper+ / Blackwell |
| **NVFP4** | 4-bit | 約 1/4 | 小 | Blackwell 專屬 |
| **MXFP4** | 4-bit | 約 1/4 | 小 | 跨平台 |
| **INT4** | 4-bit | 約 1/4 | 中等 | 所有 GPU |
| **GPTQ INT4** | 4-bit | 約 1/4 | 小 | 所有 GPU |

::: info 🤔 什麼是 NVFP4？
NVFP4（NVIDIA FP4）是 NVIDIA Blackwell 架構專屬的 4-bit 量化格式。它與一般的 INT4 不同，保留了浮點數的特性，在保持極小記憶體用量的同時，精度損失比傳統 INT4 更小。

簡單來說：
- **INT4** = 把數字四捨五入到 16 個整數值
- **NVFP4** = 把數字四捨五入到 16 個浮點值（分佈更聰明）

在 DGX Spark（Blackwell 架構）上，NVFP4 是最佳選擇。
:::

### 9-2-2 NVFP4 量化模型

NVFP4 是 NVIDIA Blackwell 架構原生的 4-bit 格式。vLLM 在 DGX Spark 上原生支援 NVFP4。

```bash
# vLLM 支援的 NVFP4 模型（截至 2025 年）
- Qwen3.5-122B-A14B-NVFP4      # 中文最強，MoE 架構
- GPT-OSS-120B-NVFP4           # OpenAI 開源模型
- Nemotron-3-Super-120B-NVFP4  # NVIDIA 自家模型
```

**各模型詳細資訊：**

| 模型 | 總參數 | 激活參數 | NVFP4 大小 | 特色 |
|------|--------|---------|-----------|------|
| Qwen3.5-122B-A14B | 122B | 14B | ~35 GB | 中文能力最強，MoE 架構 |
| GPT-OSS-120B | 120B | 120B | ~35 GB | 英文能力強，Dense 架構 |
| Nemotron-3-Super-120B | 120B | 120B | ~35 GB | 指令遵循最佳，Dense 架構 |

::: info 🤔 MoE 架構是什麼？
MoE（Mixture of Experts）是一種模型架構。想像一個公司有 122 個專家，但每次遇到問題只需要 14 個專家來處理。

- **Dense 模型**：每次推論都要用全部 120B 參數
- **MoE 模型**：每次推論只用 14B 參數（但從 122B 中挑選最合適的）

結果：MoE 模型擁有大模型的知識量，但推論速度接近小模型！
:::

### 9-2-3 MXFP4 和 FP8 模型

| 格式 | 特點 | 適合場景 | 優點 | 缺點 |
|------|------|---------|------|------|
| **NVFP4** | Blackwell 原生、速度最快 | 推論 | 速度最快、記憶體最小 | 僅限 Blackwell |
| **MXFP4** | 跨平台相容 | 推論 + 訓練 | 跨平台、精度好 | 速度略慢於 NVFP4 |
| **FP8** | 精度較高、體積較大 | 高品質推論 | 精度接近 BF16 | 記憶體用量較大 |

**選擇建議：**

| 你的需求 | 推薦格式 | 原因 |
|---------|---------|------|
| 在 DGX Spark 上追求最快速度 | NVFP4 | Blackwell 原生最佳化 |
| 需要在多種 GPU 上運行 | MXFP4 | 跨平台相容性最佳 |
| 需要最高輸出品質 | FP8 | 精度損失最小 |
| 記憶體有限 | NVFP4 | 記憶體用量最小 |

### 9-2-4 模型下載與快取管理

vLLM 會自動從 Hugging Face Hub 下載模型，並快取到本地：

```bash
# 預設快取位置
~/.cache/huggingface/hub/

# 查看快取大小
du -sh ~/.cache/huggingface/hub/

# 手動預下載模型（避免執行時等待）
export HF_HOME=~/.cache/huggingface
huggingface-cli download Qwen/Qwen3.5-122B-A14B-NVFP4
```

::: tip 💡 預下載模型的好處
在部署 vLLM 容器之前先下載模型，可以：
1. 避免容器啟動時的長時間等待
2. 確認模型檔案完整無損
3. 透過掛載 volume 讓多個容器共享模型快取
:::

---

## 9-3 用 Claude Code 部署 vLLM

### 9-3-1 確認 Docker 環境

在開始之前，確認你的環境就緒：

```bash
# 檢查 Docker 版本（建議 24.0+）
docker --version
# 預期輸出：Docker version 24.0.x 或更高

# 檢查 NVIDIA GPU 狀態
nvidia-smi
# 預期輸出：顯示 GPU 型號、驅動版本、記憶體用量

# 確認 Docker 可以使用 GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# 如果成功，會顯示與主機相同的 nvidia-smi 輸出
```

::: warning ⚠️ 常見問題
如果 `docker run --gpus all` 失敗，表示 NVIDIA Container Toolkit 沒有正確安裝。請執行：

```bash
# 安裝 NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
:::

### 9-3-2 拉取 vLLM 容器

::: info 🤔 為什麼用 `vllm/vllm-openai` 而不是 `vllm/vllm`？
vLLM 提供兩個官方映像：
- `vllm/vllm`：基礎映像，需要自己設定啟動指令
- `vllm/vllm-openai`：預設啟動 OpenAI 相容 API 伺服器，開箱即用

對於大多數使用者，`vllm-openai` 是更好的選擇。
:::

```bash
# 拉取最新穩定版
docker pull vllm/vllm-openai:latest

# 或指定特定版本（推薦用於生產環境）
docker pull vllm/vllm-openai:v0.7.3
```

### 9-3-3 用 Claude Code 一鍵部署

告訴 Claude Code：

> 「用 Docker 部署 vLLM，啟用 GPU 加速，部署 Qwen3.5-122B NVFP4 模型。」

Claude Code 會執行：

```bash
docker run -d \
  --name vllm \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

**參數詳細解釋：**

| 參數 | 說明 | 為什麼需要 |
|------|------|-----------|
| `-d` | 背景執行（detached mode） | 讓終端機可以繼續使用 |
| `--name vllm` | 容器名稱 | 方便後續管理（stop/logs/rm） |
| `--gpus all` | 使用所有可用 GPU | DGX Spark 只有一顆 GPU，所以就是它 |
| `--network host` | 使用主機網路 | 避免 port 映射的複雜性 |
| `--shm-size=16g` | 增加共享記憶體 | vLLM 的多行程通訊需要大量共享記憶體，預設 64MB 不夠 |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | 掛載 Hugging Face 快取 | 避免重複下載模型，多個容器可共享 |
| `--model` | 模型名稱（Hugging Face repo） | 指定要載入的模型 |
| `--tensor-parallel-size 1` | 使用 1 顆 GPU | DGX Spark 只有一顆 GPU |
| `--max-model-len 8192` | 最大上下文長度 | 包含輸入 + 輸出的總 token 數 |
| `--gpu-memory-utilization 0.90` | GPU 記憶體使用上限 | 留 10% 給系統和其他行程 |
| `--enforce-eager` | 使用 eager mode | ARM64 架構下避免 Triton 編譯問題 |

::: warning ⚠️ 為什麼需要 `--enforce-eager`？
在 ARM64 架構（DGX Spark 使用 Grace-Blackwell）上，Triton 編譯器可能無法正常工作。`--enforce-eager` 會跳過 Triton 的圖最佳化，改用 eager mode 執行。

這會帶來約 5-10% 的效能損失，但換取穩定性。如果你的環境 Triton 可以正常編譯，可以去掉這個參數。
:::

### 9-3-4 測試 API

vLLM 啟動後會提供 OpenAI 相容的 REST API：

```bash
# 基本對話測試
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-122B-A14B-NVFP4",
    "messages": [
      {"role": "system", "content": "你是一個有幫助的助手。"},
      {"role": "user", "content": "你好！請用三句話介紹你自己。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
  }'
```

**預期回應格式：**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3.5-122B-A14B-NVFP4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！我是..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

**串流（Streaming）模式：**

```bash
# 串流模式：逐 token 接收回應
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-122B-A14B-NVFP4",
    "messages": [
      {"role": "user", "content": "請寫一首關於春天的詩。"}
    ],
    "stream": true,
    "max_tokens": 300
  }'
```

::: tip 💡 用 Python 測試 API
如果你偏好 Python，可以用 `openai` 套件：

```python
from openai import OpenAI

# 建立客戶端，指向本地的 vLLM 伺服器
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # 本地部署不需要 API key
)

# 發送對話請求
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-122B-A14B-NVFP4",
    messages=[
        {"role": "user", "content": "解釋什麼是量子計算？"}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"\nToken 用量: {response.usage}")
```
:::

### 9-3-5 查看伺服器狀態

```bash
# 查看容器日誌
docker logs vllm

# 查看即時日誌（持續追蹤）
docker logs -f vllm

# 查看容器資源用量
docker stats vllm

# 健康檢查
curl http://localhost:8000/health
# 回應 {"status": "healthy"} 表示正常
```

---

## 9-4 部署超大模型：Qwen3.5-122B 和 GPT-OSS-120B

### 9-4-1 部署 Qwen3.5-122B NVFP4

Qwen3.5-122B 是阿里巴巴通義千問系列的最新版本，採用 MoE（Mixture of Experts）架構：

```bash
docker run -d \
  --name vllm-qwen \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-prefix-caching
```

::: tip 💡 加上 `--enable-prefix-caching`
Prefix Caching 對於多輪對話非常有用。如果多個請求有相同的開頭（例如相同的 system prompt），vLLM 會快取計算結果，避免重複計算。

在 RAG 場景中，system prompt 通常很長且固定，Prefix Caching 可以帶來 2-3 倍的速度提升。
:::

### 9-4-2 部署 GPT-OSS-120B NVFP4

GPT-OSS-120B 是 OpenAI 的開源模型：

```bash
docker run -d \
  --name vllm-gpt \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model openai/GPT-OSS-120B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

### 9-4-3 部署 Nemotron-3-Super-120B NVFP4

Nemotron 是 NVIDIA 自家的模型，在指令遵循方面表現優異：

```bash
docker run -d \
  --name vllm-nemotron \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model nvidia/Nemotron-3-Super-120B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

### 9-4-4 社群實測速度數據

以下數據來自社群使用者在 DGX Spark 上的實測結果：

| 模型 | 量化 | 上下文長度 | 輸出速度 (t/s) | 記憶體用量 | 首次載入時間 |
|------|------|-----------|---------------|-----------|------------|
| Qwen3.5-122B | NVFP4 | 8192 | ~12 | ~65 GB | ~3 分鐘 |
| GPT-OSS-120B | NVFP4 | 8192 | ~11 | ~62 GB | ~3 分鐘 |
| Nemotron-3-Super | NVFP4 | 8192 | ~10 | ~63 GB | ~3 分鐘 |
| Qwen3.5-35B-A3B | NVFP4 | 8192 | ~45 | ~15 GB | ~1 分鐘 |
| Qwen3-8B | BF16 | 8192 | ~55 | ~16 GB | ~30 秒 |
| Llama-3.1-8B | BF16 | 8192 | ~52 | ~16 GB | ~30 秒 |

::: info 🤔 t/s 是什麼？
t/s = tokens per second，每秒生成的 token 數。

換算參考：
- 一個英文單字 ≈ 1.3 tokens
- 一個中文字 ≈ 1-2 tokens
- 所以 12 t/s ≈ 每秒 6-12 個中文字

對於 120B 級別的模型，12 t/s 是合理的速度。如果你需要更快的速度，可以考慮：
1. 使用更小的模型（如 35B MoE）
2. 啟用 Speculative Decoding
3. 降低 max-model-len
:::

### 9-4-5 如何選擇模型

| 需求 | 推薦模型 | 原因 |
|------|---------|------|
| 最佳中文能力 | Qwen3.5-122B | 訓練資料包含大量中文 |
| 最佳英文能力 | GPT-OSS-120B | OpenAI 的英文能力最強 |
| 最佳指令遵循 | Nemotron-3-Super | NVIDIA 針對指令遵循最佳化 |
| 最快速度 | Qwen3.5-35B-A3B（MoE） | 每次只激活 3B 參數 |
| 平衡速度與品質 | Qwen3-8B BF16 | 8B 模型速度快且品質不錯 |
| 多語言支援 | Qwen3.5-122B | 支援 100+ 語言 |

---

## 9-5 效能調校

### 9-5-1 GPU 記憶體利用率

```bash
--gpu-memory-utilization 0.90
```

這個參數決定 vLLM 可以使用多少比例的 GPU 記憶體。

| 值 | 說明 | 適合場景 | 風險 |
|----|------|---------|------|
| 0.80 | 保守，留 20% 給系統 | 穩定性優先、同時運行其他服務 | KV cache 較小，批次量較低 |
| 0.90 | ✅ 推薦，平衡 | 一般用途 | 低風險 |
| 0.95 | 激進，最大化吞吐量 | 獨佔 GPU、追求極致效能 | 可能 OOM（記憶體不足） |
| 0.98 | 極限 | 基準測試 | 高風險，不建議生產使用 |

::: warning ⚠️ 不要設太高
如果設為 1.0（100%），vLLM 會用光所有 GPU 記憶體，可能導致 CUDA OOM 錯誤。建議至少留 5-10% 給系統和其他行程。
:::

**如何監控記憶體用量：**

```bash
# 即時監控 GPU 記憶體
watch -n 1 nvidia-smi

# 查看 vLLM 容器的詳細用量
docker stats vllm
```

### 9-5-2 最大模型長度

```bash
--max-model-len 8192
```

這個參數限制「輸入 + 輸出」的總 token 數。

| 長度 | 記憶體影響 | 適合場景 | 注意事項 |
|------|-----------|---------|---------|
| 2048 | 最小 | 簡短問答、翻譯 | 可能不夠用 |
| 4096 | 較小 | 一般對話、摘要 | 大部分場景足夠 |
| 8192 | ✅ 推薦 | 一般用途、文件分析 | 平衡點 |
| 16384 | 較大 | 長文分析、程式碼 | 需要更多記憶體 |
| 32768 | 最大 | 整本書分析 | 可能 OOM，需降低 gpu-memory-utilization |

::: info 🤔 長度 vs. 記憶體的關係
KV cache 的記憶體用量與上下文長度成正比。在 DGX Spark 上：
- 8192 長度 ≈ 額外用 5-10 GB
- 32768 長度 ≈ 額外用 20-30 GB

如果你需要長上下文，建議降低 `--gpu-memory-utilization` 到 0.85 或 0.80。
:::

### 9-5-3 批次處理參數

```bash
--max-num-batched-tokens 4096
--max-num-seqs 128
```

| 參數 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| `max-num-batched-tokens` | 每批次最大 token 數 | 4096 | 越大 = 吞吐量越高，但記憶體用量越大 |
| `max-num-seqs` | 最大同時處理的序列數 | 128 | 越大 = 同時服務越多使用者 |

**調校建議：**

| 場景 | max-num-batched-tokens | max-num-seqs |
|------|----------------------|-------------|
| 低延遲優先（少量使用者） | 2048 | 32 |
| 平衡 | 4096 | 128 |
| 高吞吐量優先（大量使用者） | 8192 | 256 |

### 9-5-4 Chunked Prefill

Chunked Prefill 把長輸入分成多塊處理，避免一次佔用太多記憶體。

```bash
--enable-chunked-prefill \
--max-num-batched-tokens 4096
```

::: info 🤔 為什麼需要 Chunked Prefill？
想像你要把一篇 10,000 字的文章餵給模型。傳統做法是一次把整篇文章送進去，這會瞬間佔用大量記憶體。

Chunked Prefill 的做法是把文章切成小塊（例如每塊 512 tokens），逐塊處理。這樣：
1. 不會瞬間佔用太多記憶體
2. 可以在處理長輸入的同時，穿插處理其他短請求
3. 降低延遲：使用者不需要等到整篇文章處理完才看到回應
:::

**何時啟用 Chunked Prefill：**

| 場景 | 建議 |
|------|------|
| 主要處理短對話（< 2048 tokens） | 不需要 |
| 需要處理長文（> 4096 tokens） | ✅ 啟用 |
| 高併發場景 | ✅ 啟用 |
| 記憶體緊張 | ✅ 啟用 |

### 9-5-5 Prefix Caching（前綴快取）

```bash
--enable-prefix-caching \
--prefix-caching-max-cache-size 4096
```

Prefix Caching 對於以下場景特別有用：

| 場景 | 加速效果 | 說明 |
|------|---------|------|
| RAG 系統 | 2-3x | System prompt + 檢索內容通常很長且重複 |
| 多輪對話 | 1.5-2x | 對話歷史是前綴，逐輪增長 |
| 批量相似任務 | 3-5x | 相同的 system prompt，只有輸入不同 |

### 9-5-6 最速配置：把所有參數組合起來

**高吞吐量配置：**

```bash
docker run -d \
  --name vllm-fast \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --disable-log-requests
```

**低延遲配置（適合互動式對話）：**

```bash
docker run -d \
  --name vllm-low-latency \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --enable-prefix-caching
```

---

## 9-6 vLLM vs. Ollama vs. llama.cpp

### 9-6-1 功能比較

| 特性 | vLLM | Ollama | llama.cpp |
|------|------|--------|-----------|
| **開發者** | UC Berkeley | Ollama 團隊 | Georgi Gerganov |
| **語言** | Python + CUDA | Go + C++ | C/C++ |
| **安裝難度** | 中等（Docker） | 超簡單（一行指令） | 需編譯（或下載 binary） |
| **吞吐量** | **最高** | 中等 | 中等 |
| **延遲** | 低 | 低 | 低 |
| **PagedAttention** | ✅ | ❌ | ❌ |
| **Continuous Batching** | ✅ | ❌ | ❌ |
| **多模型同時服務** | ✅ | ❌ | ❌ |
| **LoRA Serving** | ✅ | ❌ | ✅（有限） |
| **OpenAI API** | ✅ | ✅ | ✅（透過 server） |
| **GUI** | ❌（需搭配 WebUI） | ❌（需搭配 WebUI） | ❌（需搭配 WebUI） |
| **適合場景** | 生產環境、多人服務 | 個人使用、快速部署 | 個人使用、離線推論 |
| **ARM64 支援** | ✅ | ✅ | ✅ |

### 9-6-2 效能比較（DGX Spark 實測）

| 引擎 | 8B BF16 (t/s) | 120B NVFP4 (t/s) | 記憶體用量 | 同時服務使用者數 |
|------|--------------|-----------------|-----------|----------------|
| vLLM | ~60 | ~12 | 最佳化 | 100+ |
| Ollama | ~50 | ~10 | 較高 | 1-5 |
| llama.cpp | ~55 | ~11 | 較高 | 1-5 |

### 9-6-3 選擇建議

| 你的需求 | 推薦 | 原因 |
|---------|------|------|
| 個人使用、快速部署 | Ollama | 一行指令搞定 |
| 追求極致效能、多人服務 | vLLM | PagedAttention + Continuous Batching |
| 最大彈性、離線推論 | llama.cpp | 支援最多硬體平台 |
| 需要 LoRA 多任務 | vLLM | 原生支援多 LoRA 同時服務 |
| 生產環境 API 服務 | vLLM | 完善的監控和擴展能力 |

---

## 9-7 清理與移除

### 9-7-1 移除容器和映像檔

```bash
# 停止容器
docker stop vllm

# 移除容器
docker rm vllm

# 移除映像檔（如果想節省磁碟空間）
docker rmi vllm/vllm-openai:latest

# 一次清理所有 vLLM 相關容器
docker rm -f $(docker ps -a --filter "name=vllm" -q)
```

### 9-7-2 清理模型快取

```bash
# 查看快取大小
du -sh ~/.cache/huggingface/hub/

# 清理特定模型
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A14B-NVFP4

# 清理所有 Hugging Face 快取（會刪除所有下載的模型）
rm -rf ~/.cache/huggingface
```

::: warning ⚠️ 清理前確認
清理模型快取後，下次部署需要重新下載。如果磁碟空間足夠，建議保留快取。
:::

### 9-7-3 一鍵清理腳本

```bash
#!/bin/bash
# 清理所有 vLLM 相關資源
echo "停止 vLLM 容器..."
docker stop $(docker ps -a --filter "name=vllm" -q) 2>/dev/null

echo "移除 vLLM 容器..."
docker rm $(docker ps -a --filter "name=vllm" -q) 2>/dev/null

echo "移除 vLLM 映像檔..."
docker rmi vllm/vllm-openai:latest 2>/dev/null

echo "清理完成！"
```

---

## 9-8 網路流言 91t/s 的模型真的假的？

### 9-8-1 實測結果分析

社群中有人宣稱在 DGX Spark 上跑出 91 t/s 的速度。我們來分析：

| 說法 | 真相 | 說明 |
|------|------|------|
| 91 t/s 用 120B 模型 | ❌ 不可能 | 120B NVFP4 約 12 t/s，差了 7 倍 |
| 91 t/s 用 8B 模型 | ✅ 可能 | 8B BF16 約 50-80 t/s，最佳化後可達 90+ |
| 91 t/s 用特定小模型 | ✅ 可能 | 3B 模型可以超過 100 t/s |
| 91 t/s 用 MoE（35B-A3B） | ✅ 可能 | MoE 只激活 3B，速度接近 3B 模型 |

::: info 🤔 為什麼 91 t/s 對 120B 不可能？
簡單的物理限制：
1. DGX Spark 的記憶體頻寬約 400 GB/s
2. 120B NVFP4 模型大小約 35 GB
3. 每次生成一個 token 需要讀取整個模型
4. 理論上限 = 400 GB/s ÷ 35 GB ≈ 11.4 次/秒

所以 12 t/s 已經非常接近硬體極限了。宣稱 91 t/s 跑 120B 模型，就像說你的小轎車跑出了 F1 的速度 — 物理上不可能的。
:::

### 9-8-2 社群方案：spark-vllm-docker

社群有人建立了專門為 DGX Spark 最佳化的 vLLM Docker 映像：

```bash
docker run -d \
  --name spark-vllm \
  --gpus all \
  --network host \
  --shm-size=16g \
  ghcr.io/community/spark-vllm-docker:latest \
  --model Qwen3-8B \
  --max-model-len 4096
```

這個映像針對 DGX Spark 做了額外最佳化：
- 針對 Grace-Blackwell 架構編譯
- 預設最佳化的記憶體參數
- 移除不必要的依賴

**速度提升：** 比官方映像快 10-15%。

### 9-8-3 要 90+ t/s？換 Qwen3.5-35B-A3B

如果你真的需要超高速度，使用 MoE（Mixture of Experts）架構的模型：

```bash
docker run -d \
  --name vllm-moe \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-35B-A3B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Qwen3.5-35B-A3B 是一個 MoE 模型：
- 總參數：35B
- 每次激活：3B
- 速度：~45-60 t/s（接近 3B 模型的速度）
- 品質：接近 35B 模型的品質

::: tip 💡 MoE 的甜蜜點
MoE 模型是「用大模型的知識量，小模型的速度」的最佳折衷。如果你需要高速度但又不想犧牲太多品質，MoE 是你的最佳選擇。
:::

### 9-8-4 速度基準測試

你可以用以下腳本測試實際速度：

```bash
# 安裝基準測試工具
pip install vllm-benchmarks

# 執行基準測試
python -m vllm_benchmarks \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --base-url http://localhost:8000/v1 \
  --num-prompts 100 \
  --max-tokens 256
```

**預期輸出：**

```
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  45.2
Total input tokens:                      2500
Total generated tokens:                  25600
Request throughput (req/s):              2.21
Output token throughput (tok/s):         566.37
Time to First Token (ms):                150.5
Mean TTFT (ms):                          155.2
Median TTFT (ms):                        152.1
P99 TTFT (ms):                           210.3
Mean TPOT (ms):                          83.3
Median TPOT (ms):                        82.1
P99 TPOT (ms):                           95.7
==================================================
```

::: info 🤔 關鍵指標解釋
- **TTFT（Time To First Token）**：從發送請求到收到第一個 token 的時間。越低越好，影響使用者的「即時感」。
- **TPOT（Time Per Output Token）**：每個輸出 token 的平均時間。越低 = 速度越快。
- **Throughput（吞吐量）**：每秒處理的 token 數。越高越好。
- **P99**：99% 的請求都在這個時間內完成。反映最壞情況。
:::

---

## 9-9 疑難排解 FAQ

### Q1：容器啟動後立刻退出？

```bash
# 查看日誌找出原因
docker logs vllm

# 常見原因：
# 1. GPU 記憶體不足 → 降低 --gpu-memory-utilization
# 2. 模型下載失敗 → 檢查網路連線
# 3. --enforce-eager 沒加 → ARM64 需要此參數
```

### Q2：CUDA Out of Memory 錯誤？

```bash
# 解決方案 1：降低 GPU 記憶體利用率
--gpu-memory-utilization 0.80

# 解決方案 2：減少上下文長度
--max-model-len 4096

# 解決方案 3：啟用 Chunked Prefill
--enable-chunked-prefill --max-num-batched-tokens 2048
```

### Q3：API 回應很慢？

```bash
# 檢查是否有多個請求在排隊
curl http://localhost:8000/metrics | grep vllm:num_requests_running

# 檢查 GPU 利用率
nvidia-smi

# 如果 GPU 利用率低，可能是 CPU 瓶頸
# 嘗試增加 --max-num-batched-tokens
```

### Q4：如何同時運行多個模型？

```bash
# 在不同 port 上啟動多個 vLLM 容器
docker run -d \
  --name vllm-qwen \
  --gpus all \
  --network host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --port 8000

docker run -d \
  --name vllm-llama \
  --gpus all \
  --network host \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B \
  --port 8001
```

::: warning ⚠️ 注意記憶體
同時運行多個模型會佔用更多 GPU 記憶體。確保總用量不超過 128GB。
:::

### Q5：如何更新 vLLM 到最新版本？

```bash
# 停止並移除舊容器
docker stop vllm && docker rm vllm

# 拉取最新映像
docker pull vllm/vllm-openai:latest

# 重新啟動（用相同的指令）
docker run -d \
  --name vllm \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

---

## 9-10 本章小結

::: success ✅ 你現在知道了
- vLLM 的 PagedAttention 和 Continuous Batching 是高效能的關鍵
- PagedAttention 把 KV cache 分頁管理，記憶體浪費從 60-80% 降到 4% 以下
- Continuous Batching 讓完成的請求立即輸出，新請求立即插入
- vLLM 支援 NVFP4、MXFP4、FP8 等量化格式，NVFP4 在 Blackwell 上最快
- 正確的參數調校（gpu-memory-utilization、max-model-len、chunked prefill）可以大幅提升吞吐量
- 91 t/s 的說法需要看是用什麼模型 — 120B 模型不可能達到，但 8B 或 MoE 模型可以
- MoE 架構的模型可以在保持品質的同時大幅提升速度
- vLLM 適合生產環境、多人服務、高吞吐量場景
:::

::: tip 🚀 下一章預告
vLLM 是開源社群的寵兒，那 NVIDIA 官方有什麼對應方案呢？TensorRT-LLM 就是答案 — 它是 NVIDIA 自己開發的推論加速引擎，追求極致效能！

👉 [前往第 10 章：TensorRT-LLM — NVIDIA 原生加速引擎 →](/guide/chapter10/)
:::

::: info 📝 上一章
← [回到第 8 章：llama.cpp](/guide/chapter8/)
:::
