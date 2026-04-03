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
約 20 分鐘。
:::

---

## 9-1 vLLM 架構與特色

### 9-1-1 PagedAttention

::: info 🤔 什麼是 PagedAttention？
想像你在讀一本很長的書。傳統的做法是把整本書攤在桌子上（預先分配所有記憶體），但這樣很浪費空間。

PagedAttention 的做法是：只攤開你正在讀的那幾頁，其他頁放在書架上，需要時再翻。這樣可以大幅減少記憶體浪費。

技術上來說，PagedAttention 把 KV cache 分成固定大小的「頁面」（page），像作業系統的虛擬記憶體一樣動態管理。
:::

**效果**：
- 記憶體浪費從 60-80% 降到 4% 以下
- 可以同時服務更多使用者
- 更長的上下文長度

### 9-1-2 Continuous Batching

傳統 batching 是等所有請求都處理完才輸出結果。Continuous Batching 則是：

```
傳統 Batching：
請求 A: [===========] ← 等 A 完成才處理 B
請求 B:             [===========]

Continuous Batching：
請求 A: [=======] ← A 完成就輸出
請求 B:   [===========] ← B 可以插隊
請求 C:     [===] ← C 也可以插隊
```

**效果**：
- 吞吐量提升 2-4 倍
- 延遲降低
- GPU 利用率更高

### 9-1-3 為什麼 vLLM 適合 DGX Spark

| 特性 | 為什麼適合 DGX Spark |
|------|---------------------|
| 高吞吐量 | 128GB 記憶體可以批次處理更多請求 |
| PagedAttention | 減少記憶體浪費，裝更大的模型 |
| Continuous Batching | 充分利用 GPU 算力 |
| 多模型支援 | 支援 NVFP4、FP8 等 Blackwell 原生格式 |

### 9-1-4 vLLM 的其他進階功能

- **Speculative Decoding**：推測性解碼加速
- **Prefix Caching**：快取常用前綴，加速重複查詢
- **LoRA Serving**：同時服務多個 LoRA adapter
- **Logits Processor**：自訂輸出行為
- **Metrics**：Prometheus 監控指標

---

## 9-2 支援的模型

### 9-2-1 NVFP4 量化模型

NVFP4 是 NVIDIA Blackwell 架構原生的 4-bit 格式。vLLM 在 DGX Spark 上原生支援 NVFP4。

```bash
# vLLM 支援的 NVFP4 模型
- Qwen3.5-122B-NVFP4
- GPT-OSS-120B-NVFP4
- Nemotron-3-Super-120B-NVFP4
```

### 9-2-2 MXFP4 和 FP8 模型

| 格式 | 特點 | 適合場景 |
|------|------|---------|
| **NVFP4** | Blackwell 原生、速度最快 | 推論 |
| **MXFP4** | 跨平台相容 | 推論 + 訓練 |
| **FP8** | 精度較高、體積較大 | 高品質推論 |

---

## 9-3 用 Claude Code 部署 vLLM

### 9-3-1 確認 Docker 環境

```bash
docker --version
nvidia-smi
```

### 9-3-2 拉取 vLLM 容器

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

**參數解釋**：

| 參數 | 說明 |
|------|------|
| `--shm-size=16g` | 增加共享記憶體（vLLM 需要） |
| `--tensor-parallel-size 1` | 單 GPU |
| `--max-model-len 8192` | 最大上下文長度 |
| `--gpu-memory-utilization 0.90` | 使用 90% GPU 記憶體 |
| `--enforce-eager` | 避免 Triton 編譯問題（ARM64 需要） |

### 9-3-3 測試 API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-122B-A14B-NVFP4",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "stream": false
  }'
```

---

## 9-4 部署超大模型：Qwen3.5-122B 和 GPT-OSS-120B

### 9-4-1 部署 Qwen3.5-122B NVFP4

```bash
docker run -d \
  --name vllm-qwen \
  --gpus all \
  --network host \
  --shm-size=16g \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

### 9-4-2 社群實測速度數據

| 模型 | 量化 | 上下文長度 | 輸出速度 | 記憶體用量 |
|------|------|-----------|---------|-----------|
| Qwen3.5-122B | NVFP4 | 8192 | ~12 t/s | ~65 GB |
| GPT-OSS-120B | NVFP4 | 8192 | ~11 t/s | ~62 GB |
| Nemotron-3-Super | NVFP4 | 8192 | ~10 t/s | ~63 GB |

::: info 🤔 t/s 是什麼？
t/s = tokens per second，每秒生成的 token 數。一個中文字約 1-2 個 token。

12 t/s 約等於每秒 6-12 個中文字，對於 120B 模型來說是合理的速度。
:::

### 9-4-3 如何選擇模型

| 需求 | 推薦模型 |
|------|---------|
| 最佳中文能力 | Qwen3.5-122B |
| 最佳英文能力 | GPT-OSS-120B |
| 最佳指令遵循 | Nemotron-3-Super |
| 最快速度 | Qwen3.5-35B-A3B（MoE 架構） |

---

## 9-5 效能調校

### 9-5-1 GPU 記憶體利用率

```bash
--gpu-memory-utilization 0.90
```

這個參數決定 vLLM 可以使用多少比例的 GPU 記憶體。

| 值 | 說明 |
|----|------|
| 0.80 | 保守，留 20% 給系統 |
| 0.90 | ✅ 推薦，平衡 |
| 0.95 | 激進，可能不穩定 |

### 9-5-2 最大模型長度

```bash
--max-model-len 8192
```

| 長度 | 記憶體影響 | 適合場景 |
|------|-----------|---------|
| 4096 | 最小 | 簡短對話 |
| 8192 | ✅ 推薦 | 一般用途 |
| 16384 | 較大 | 長文分析 |
| 32768 | 最大 | 文件處理 |

### 9-5-3 批次處理參數

```bash
--max-num-batched-tokens 4096
--max-num-seqs 128
```

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `max-num-batched-tokens` | 每批次最大 token 數 | 4096 |
| `max-num-seqs` | 最大同時處理的序列數 | 128 |

### 9-5-4 Chunked Prefill

Chunked Prefill 把長輸入分成多塊處理，避免一次佔用太多記憶體。

```bash
--enable-chunked-prefill \
--max-num-batched-tokens 4096
```

### 9-5-5 最速配置：把所有參數組合起來

```bash
docker run -d \
  --name vllm-fast \
  --gpus all \
  --network host \
  --shm-size=16g \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --disable-log-requests
```

---

## 9-6 vLLM vs. Ollama vs. llama.cpp

### 9-6-1 功能比較

| 特性 | vLLM | Ollama | llama.cpp |
|------|------|--------|-----------|
| 安裝難度 | 中等 | 超簡單 | 需編譯 |
| 吞吐量 | **最高** | 中等 | 中等 |
| 延遲 | 低 | 低 | 低 |
| PagedAttention | ✅ | ❌ | ❌ |
| Continuous Batching | ✅ | ❌ | ❌ |
| 多模型同時服務 | ✅ | ❌ | ❌ |
| LoRA Serving | ✅ | ❌ | ❌ |
| 適合場景 | 生產環境 | 個人使用 | 個人使用 |

### 9-6-2 選擇建議

- **個人使用、快速部署**：Ollama
- **追求極致效能、多人服務**：vLLM
- **最大彈性、離線推論**：llama.cpp

---

## 9-7 清理與移除

### 9-7-1 移除容器和映像檔

```bash
docker stop vllm
docker rm vllm
docker rmi vllm/vllm-openai:latest
```

### 9-7-2 清理模型快取

```bash
# 清理 Hugging Face 快取
rm -rf ~/.cache/huggingface
```

---

## 9-8 網路流言 91t/s 的模型真的假的？

### 9-8-1 實測結果

社群中有人宣稱在 DGX Spark 上跑出 91 t/s 的速度。我們來分析：

| 說法 | 真相 |
|------|------|
| 91 t/s 用 120B 模型 | ❌ 不可能。120B NVFP4 約 12 t/s |
| 91 t/s 用 8B 模型 | ✅ 可能。8B BF16 約 50-80 t/s |
| 91 t/s 用特定小模型 | ✅ 可能。3B 模型可以超過 100 t/s |

### 9-8-2 社群方案：spark-vllm-docker

社群有人建立了專門為 DGX Spark 最佳化的 vLLM Docker 映像：

```bash
docker run -d \
  --name spark-vllm \
  --gpus all \
  --network host \
  ghcr.io/community/spark-vllm-docker:latest \
  --model Qwen3-8B \
  --max-model-len 4096
```

這個映像針對 DGX Spark 做了額外最佳化，速度比官方映像快 10-15%。

### 9-8-3 要 90+ t/s？換 Qwen3.5-35B-A3B

如果你真的需要超高速度，使用 MoE（Mixture of Experts）架構的模型：

```bash
docker run -d \
  --name vllm-moe \
  --gpus all \
  --network host \
  --shm-size=16g \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-35B-A3B-NVFP4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Qwen3.5-35B-A3B 是一個 MoE 模型，雖然總參數是 35B，但每次推論只激活 3B 參數，所以速度非常快。

---

## 9-9 本章小結

::: success ✅ 你現在知道了
- vLLM 的 PagedAttention 和 Continuous Batching 是高效能的關鍵
- vLLM 支援 NVFP4、MXFP4、FP8 等量化格式
- 正確的參數調校可以大幅提升吞吐量
- 91 t/s 的說法需要看是用什麼模型
- MoE 架構的模型可以在保持品質的同時大幅提升速度
:::

::: tip 🚀 下一章預告
vLLM 是開源社群的寵兒，那 NVIDIA 官方有什麼對應方案呢？TensorRT-LLM 就是答案 — 它是 NVIDIA 自己開發的推論加速引擎！

👉 [前往第 10 章：TensorRT-LLM — NVIDIA 原生加速引擎 →](/guide/chapter10/)
:::

::: info 📝 上一章
← [回到第 8 章：llama.cpp](/guide/chapter8/)
:::
