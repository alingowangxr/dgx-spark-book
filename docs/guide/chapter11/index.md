# 第 11 章：SGLang 與推測性解碼

::: tip 🎯 本章你將學到什麼
- SGLang 的核心技術：RadixAttention 和結構化生成
- SGLang 的程式設計模型和 DSL
- 部署 SGLang 容器
- 推測性解碼：EAGLE-3 vs. Draft-Target
- 完整應用範例
- 疑難排解
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 11-1 SGLang 推論框架

### 11-1-1 什麼是 SGLang？

SGLang（Structured Generation Language）是由 UC Berkeley 和社群共同開發的 LLM 推論框架。它的核心理念是：**讓 LLM 的程式設計像寫普通程式一樣簡單**。

::: info 🤔 SGLang 名字的由來
SGLang = **S**tructured **G**eneration **Lang**uage

它有兩個層面的含義：
1. **結構化生成**：讓模型輸出特定格式的內容（如 JSON、XML）
2. **生成語言**：提供一套 DSL（領域特定語言）來編寫 LLM 程式

簡單來說，SGLang 不只是推論伺服器，更是一個「LLM 的程式語言」。
:::

**SGLang 的兩大組成：**

| 組成 | 說明 | 類似於 |
|------|------|--------|
| **SGLang Runtime** | 高效能推論伺服器 | vLLM、TGI |
| **SGLang DSL** | 結構化生成語言 | Python 的 prompt 模板引擎 |

### 11-1-2 RadixAttention 深入解析

::: info 🤔 什麼是 RadixAttention？
RadixAttention 是 SGLang 的獨家技術。它把所有對話歷史組織成一個「前綴樹」（Radix Tree / Trie），讓相同前綴的請求可以共享計算結果。

舉例來說，如果 10 個使用者都問了類似開頭的問題，RadixAttention 只需要計算一次前綴，大幅節省資源。
:::

**視覺化理解：**

```
一般做法（每個請求獨立計算）：
請求 1: [System Prompt][User A 的問題] → 全部重新計算
請求 2: [System Prompt][User B 的問題] → 全部重新計算
請求 3: [System Prompt][User C 的問題] → 全部重新計算
總計算量 = 3 × (System Prompt + 問題)

RadixAttention（共享前綴）：
                    ┌─ [User A 的問題]
[System Prompt] ────┼─ [User B 的問題]
                    └─ [User C 的問題]
總計算量 = System Prompt + User A + User B + User C
節省 = 2 × System Prompt 的計算量
```

**效果：**

| 場景 | 加速效果 | 說明 |
|------|---------|------|
| RAG 系統 | 3-5x | 相同的 system prompt + 檢索模板 |
| 多輪對話 | 2-3x | 對話歷史逐輪增長，前綴重複 |
| 批量相似任務 | 5-10x | 相同的 prompt 模板，只有變數不同 |
| Function Calling | 2-4x | 相同的 function 定義 |

::: tip 💡 RadixAttention vs. vLLM 的 Prefix Caching
兩者概念相似，但實現不同：
- **vLLM Prefix Caching**：基於 KV cache 的頁面共享
- **SGLang RadixAttention**：基於前綴樹的全局管理

RadixAttention 更靈活，可以跨請求共享任意長度的前綴，而不僅限於固定的頁面大小。
:::

### 11-1-3 結構化生成

SGLang 最強大的功能之一是結構化生成。它可以強制模型輸出特定格式的內容。

**為什麼需要結構化生成？**

| 問題 | 傳統做法 | SGLang 做法 |
|------|---------|------------|
| 需要 JSON 輸出 | 用 prompt 要求，但可能失敗 | 用 regex 或 JSON schema 強制 |
| 需要特定格式 | 後處理修正，可能出錯 | 生成時就保證正確 |
| 需要列舉選項 | 可能輸出不在選項中的內容 | 強制只能輸出指定選項 |

**範例 1：JSON 提取**

```python
import sglang as sgl

# 定義一個結構化生成函數
@sgl.function
def json_extraction(s, text):
    s += "請從以下文字中提取資訊，輸出 JSON 格式：\n"
    s += text + "\n"
    # 使用 regex 強制輸出 JSON 格式
    s += sgl.gen("json_output", max_tokens=256, regex=r'\{.*\}')

# 執行
result = json_extraction.run(
    text="張三，35歲，住在台北，是一位軟體工程師。"
)
print(result["json_output"])
# 保證輸出是合法的 JSON，例如：
# {"name": "張三", "age": 35, "city": "台北", "occupation": "軟體工程師"}
```

**範例 2：選擇題**

```python
@sgl.function
def multiple_choice(s, question, choices):
    s += f"問題：{question}\n"
    for i, choice in enumerate(choices):
        s += f"{chr(ord('A') + i)}. {choice}\n"
    # 強制只能輸出 A、B、C、D
    s += sgl.gen("answer", max_tokens=1, choices=["A", "B", "C", "D"])

result = multiple_choice.run(
    question="台灣的首都是哪裡？",
    choices=["台北", "高雄", "台中", "台南"]
)
print(result["answer"])  # 保證是 A、B、C 或 D
```

**範例 3：複雜的多步驟推理**

```python
@sgl.function
def chain_of_thought(s, question):
    s += f"問題：{question}\n"
    s += "讓我們一步步思考：\n"
    # 第一步：分析問題
    s += "步驟 1 - 分析：" + sgl.gen("analysis", stop="\n") + "\n"
    # 第二步：推理
    s += "步驟 2 - 推理：" + sgl.gen("reasoning", stop="\n") + "\n"
    # 第三步：結論
    s += "步驟 3 - 結論：" + sgl.gen("conclusion", stop="\n")

result = chain_of_thought.run(
    question="如果所有貓都會飛，而小明養了一隻貓，小明能飛嗎？"
)
print(result["analysis"])
print(result["reasoning"])
print(result["conclusion"])
```

### 11-1-4 SGLang 的程式設計模型

SGLang 提供了一種直覺的程式設計方式來與 LLM 互動：

```python
import sglang as sgl

# 設定後端（指向運行中的 SGLang 伺服器）
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

# 方式 1：簡單的問答
@sgl.function
def simple_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A: " + sgl.gen("answer", max_tokens=256)

# 方式 2：多輪對話
@sgl.function
def multi_turn(s, topic):
    s += f"讓我們討論{topic}。\n"
    s += "你認為最重要的三個面向是什麼？\n"
    s += sgl.gen("point1", max_tokens=100, stop="\n") + "\n"
    s += sgl.gen("point2", max_tokens=100, stop="\n") + "\n"
    s += sgl.gen("point3", max_tokens=100, stop="\n")

# 方式 3：平行生成（同時生成多個回答）
@sgl.function
def parallel_gen(s, prompt):
    # 同時生成三個不同風格的回答
    forks = s.fork(3)
    forks[0] += f"用正式的語氣回答：{prompt}"
    forks[1] += f"用輕鬆的語氣回答：{prompt}"
    forks[2] += f"用專業的語氣回答：{prompt}"
    results = forks.join()
    s += "正式：" + results[0]["answer"]
    s += "輕鬆：" + results[1]["answer"]
    s += "專業：" + results[2]["answer"]
```

### 11-1-5 SGLang vs. vLLM vs. TRT-LLM

| 特性 | SGLang | vLLM | TRT-LLM |
|------|--------|------|---------|
| **開發者** | UC Berkeley + 社群 | UC Berkeley + 社群 | NVIDIA |
| **核心技術** | RadixAttention | PagedAttention | TensorRT Engine |
| **RadixAttention** | ✅ 獨家 | ❌ | ❌ |
| **結構化生成** | ✅ 原生支援（DSL） | ✅（Outlines 整合） | 有限 |
| **推測性解碼** | ✅ EAGLE-3 | ✅ | ✅ |
| **效能** | 高 | 高 | 最高 |
| **靈活性** | **最高** | 高 | 低 |
| **程式設計模型** | DSL（類 Python） | REST API | REST/gRPC API |
| **適合場景** | 研究、結構化生成、複雜流程 | 一般生產環境 | 極致效能 |
| **學習曲線** | 中等 | 低 | 高 |

---

## 11-2 部署 SGLang

### 11-2-1 驗證環境

```bash
# 檢查 Docker 版本
docker --version
# 預期：Docker version 24.0+

# 檢查 GPU 狀態
nvidia-smi
# 確認 GPU 可用

# 測試 Docker GPU 支援
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 11-2-2 拉取 SGLang 容器

```bash
# 拉取最新穩定版
docker pull lmsysorg/sglang:latest

# 或指定特定版本（推薦用於生產環境）
docker pull lmsysorg/sglang:v0.4.0
```

::: info 🤔 為什麼是 lmsysorg？
lmsysorg（Large Model System Organization）是 UC Berkeley 領導的開源組織，也是 SGLang 的維護者。他們也維護了 LMSYS Chatbot Arena（知名的 LLM 評比平台）。
:::

### 11-2-3 啟動推論伺服器

**基本部署（8B 模型）：**

```bash
docker run -d \
  --name sglang \
  --gpus all \
  --network host \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85
```

**參數詳細解釋：**

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--model-path` | 模型路徑（Hugging Face repo 或本地路徑） | 模型名稱 |
| `--host` | 監聽位址 | 0.0.0.0（允許外部連線） |
| `--port` | 監聽埠號 | 30000 |
| `--mem-fraction-static` | 用於 KV cache 的 GPU 記憶體比例 | 0.8-0.9 |
| `--context-length` | 最大上下文長度 | 8192 |
| `--tp-size` | Tensor Parallelism 大小 | 1（單 GPU） |

### 11-2-4 測試 API

SGLang 提供兩種 API 格式：

**格式 1：原生 API**

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好！請用三句話介紹 DGX Spark。",
    "sampling_params": {
      "max_new_tokens": 200,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }'
```

**格式 2：OpenAI 相容 API**

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 11-2-5 部署大型模型（122B NVFP4）

```bash
docker run -d \
  --name sglang-large \
  --gpus all \
  --network host \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --context-length 8192
```

### 11-2-6 啟用 RadixAttention

RadixAttention 在 SGLang 中預設啟用，但可以透過參數調整：

```bash
docker run -d \
  --name sglang-radix \
  --gpus all \
  --network host \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096
```

::: tip 💡 測試 RadixAttention 的效果
你可以用以下 Python 腳本測試 RadixAttention 的加速效果：

```python
import sglang as sgl
import time

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def qa_with_prefix(s, question):
    # 所有請求都有相同的 system prompt
    s += "你是一個專業的中文助手。請用簡潔的方式回答問題。\n\n"
    s += f"問題：{question}\n"
    s += "回答：" + sgl.gen("answer", max_tokens=256)

# 第一次請求（需要計算 system prompt）
start = time.time()
ret1 = qa_with_prefix.run(question="什麼是人工智慧？")
first_time = time.time() - start
print(f"第一次：{first_time:.2f}s")

# 第二次請求（RadixAttention 會快取 system prompt）
start = time.time()
ret2 = qa_with_prefix.run(question="什麼是機器學習？")
second_time = time.time() - start
print(f"第二次：{second_time:.2f}s")

print(f"加速比：{first_time/second_time:.2f}x")
```
:::

### 11-2-7 離線推論

SGLang 也支援離線批次推論，不需要啟動伺服器：

```python
import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint

# 方式 1：連接遠端伺服器
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A: " + gen("answer", max_tokens=256)

ret = text_qa.run(question="什麼是 DGX Spark？")
print(ret["answer"])

# 方式 2：本地離線推論（不需要伺服器）
from sglang import Runtime

# 建立本地 runtime
runtime = Runtime(model_path="Qwen/Qwen3-8B")
set_default_backend(runtime)

@function
def summarize(s, text):
    s += "請總結以下文字：\n" + text + "\n"
    s += "總結：" + gen("summary", max_tokens=200)

ret = summarize.run(text="DGX Spark 是 NVIDIA 推出的個人 AI 超級電腦...")
print(ret["summary"])

# 關閉 runtime
runtime.shutdown()
```

### 11-2-8 支援的模型

SGLang 支援大部分主流的開源模型：

| 模型家族 | 支援情況 | 備註 |
|---------|---------|------|
| **Llama 3.x** | ✅ 完整支援 | 最佳化最完整 |
| **Qwen 2.5/3/3.5** | ✅ 完整支援 | 中文最佳 |
| **Mistral/Mixtral** | ✅ 完整支援 | MoE 支援 |
| **Gemma 2** | ✅ 完整支援 | |
| **DeepSeek** | ✅ 支援 | 包括 V3 MoE |
| **Phi-3/4** | ✅ 支援 | 微軟小型模型 |
| **Yi** | ✅ 支援 | 零一萬物 |

---

## 11-3 推測性解碼

### 11-3-1 什麼是推測性解碼？

::: info 🤔 推測性解碼的原理
推測性解碼（Speculative Decoding）的核心想法是：**用一個小模型快速「猜測」大模型的輸出，然後用大模型驗證**。

想像你在考試：
- **傳統做法**：每道題都仔細思考很久才寫答案
- **推測性解碼**：先用直覺快速寫出答案，再仔細檢查修正

如果猜對了，就省下了大模型的計算時間。如果猜錯了，大模型會修正。

關鍵是：小模型的計算成本遠低於大模型，所以即使有猜錯的情況，整體仍然更快。
:::

**視覺化流程：**

```
傳統解碼：
大模型: [token1] → [token2] → [token3] → [token4] → [token5]
         (慢)      (慢)       (慢)       (慢)       (慢)
總時間 = 5 × 大模型時間

推測性解碼：
小模型: [猜token1, 猜token2, 猜token3, 猜token4, 猜token5]  ← 快速
大模型: [✓驗證1, ✓驗證2, ✗修正3, ✓驗證4, ✓驗證5]          ← 一次驗證多個
總時間 ≈ 1 × 小模型時間 + 1 × 大模型時間
```

### 11-3-2 EAGLE-3

EAGLE-3 是 SGLang 支援的一種推測性解碼方法。它訓練一個小型的「草稿模型」來預測大模型的輸出。

```bash
docker run -d \
  --name sglang-eagle \
  --gpus all \
  --network host \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path Qwen/Qwen3-8B-EAGLE-3 \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 64
```

**EAGLE-3 參數解釋：**

| 參數 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| `--speculative-algorithm` | 推測演算法 | EAGLE | 固定 |
| `--speculative-draft-model-path` | 草稿模型路徑 | 對應的 EAGLE 模型 | 必須與目標模型匹配 |
| `--speculative-num-steps` | 每次推測的步數 | 5 | 越大 = 猜越多，但驗證成本越高 |
| `--speculative-eagle-topk` | 每個步驟的候选數 | 8 | 越大 = 命中率越高，但記憶體用量越大 |
| `--speculative-num-draft-tokens` | 草稿 token 總數 | 64 | 控制整體推測規模 |

::: tip 💡 EAGLE-3 的效果
在 DGX Spark 上，EAGLE-3 可以帶來：
- 推論速度提升：1.5-2.5 倍
- 額外記憶體用量：~5-10 GB（草稿模型）
- 最佳場景：程式碼生成、技術文件（有明確模式的內容）
- 較差場景：創意寫作、詩詞（難以預測）
:::

**EAGLE-3 的運作方式：**

```
EAGLE 不是簡單地用一個小模型來猜測，而是：

1. 用目標模型的前幾層特徵來預測下一個 token
2. 用一個輕量級的「草稿頭」（draft head）來預測後續多個 token
3. 用目標模型一次驗證所有預測

好處：
- 不需要額外的草稿模型（節省記憶體）
- 草稿頭與目標模型高度匹配（命中率高）
- 可以動態調整推測步數
```

### 11-3-3 Draft-Target

Draft-Target 是另一種推測性解碼方法，使用完全不同的模型作為草稿模型。

```bash
docker run -d \
  --name sglang-draft \
  --gpus all \
  --network host \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --speculative-algorithm DRAFT_TARGET \
  --speculative-draft-model-path Qwen/Qwen3-8B \
  --speculative-num-steps 3
```

這裡用 Qwen3-8B 作為 Qwen3.5-122B 的草稿模型。

::: info 🤔 Draft-Target 的草稿模型選擇原則
1. **相同架構**：草稿模型和目標模型應該使用相同的架構（如都是 Qwen）
2. **較小參數**：草稿模型應該比目標模型小很多（至少 4 倍以上）
3. **相同詞表**：最好使用相同的 tokenizer
4. **相同訓練資料**：訓練資料越接近，命中率越高

好的組合：
- Qwen3-8B → Qwen3.5-122B ✅
- Llama-3.1-8B → Llama-3.1-70B ✅
- Qwen3-8B → Llama-3.1-70B ❌（不同架構）
:::

### 11-3-4 EAGLE-3 vs. Draft-Target

| 特性 | EAGLE-3 | Draft-Target |
|------|---------|-------------|
| **草稿模型** | 專門訓練的 EAGLE 頭 | 任意小模型 |
| **加速比** | 1.8-2.5x | 1.3-2.0x |
| **記憶體用量** | 中等（+5-10 GB） | 較高（需要載入完整小模型） |
| **命中率** | 高（與目標模型匹配度高） | 中等（取決於模型選擇） |
| **設定難度** | 高（需要特定 EAGLE 模型） | 低（任意小模型即可） |
| **模型支援** | 有限（僅部分模型有 EAGLE 版本） | 廣泛（任意小模型） |
| **推薦場景** | 追求極致速度、有對應 EAGLE 模型 | 快速設定、沒有對應 EAGLE 模型 |

### 11-3-5 推測性解碼的效能基準

以下是在 DGX Spark 上的近似數據：

| 配置 | 輸出速度 (t/s) | 加速比 | 記憶體用量 |
|------|---------------|--------|-----------|
| Qwen3-8B（無推測） | ~55 | 1.0x | ~16 GB |
| Qwen3-8B + EAGLE-3 | ~110 | 2.0x | ~22 GB |
| Qwen3.5-122B（無推測） | ~12 | 1.0x | ~65 GB |
| Qwen3.5-122B + Draft-Target（8B） | ~20 | 1.7x | ~80 GB |

::: warning ⚠️ 記憶體注意事項
推測性解碼需要額外的記憶體來載入草稿模型或 EAGLE 頭。在 DGX Spark 上：
- EAGLE-3：額外 5-10 GB
- Draft-Target：額外 16 GB（完整小模型）

確保總記憶體用量不超過 GPU 的 128GB。
:::

---

## 11-4 完整應用範例

### 11-4-1 RAG 系統（檢索增強生成）

```python
import sglang as sgl

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def rag_qa(s, question, context):
    # System prompt 會被 RadixAttention 快取
    s += "你是一個專業的問答助手。請根據提供的上下文回答問題。\n\n"
    s += f"上下文：\n{context}\n\n"
    s += f"問題：{question}\n\n"
    s += "回答：" + sgl.gen("answer", max_tokens=512)

# 模擬檢索到的上下文
context = """
DGX Spark 是 NVIDIA 於 2025 年推出的個人 AI 超級電腦。
它搭載 Grace-Blackwell 架構，擁有 128GB 統一記憶體。
支援運行 120B 參數的大型語言模型。
"""

# 第一個問題（system prompt 會被計算並快取）
ret1 = rag_qa.run(
    question="DGX Spark 的記憶體有多大？",
    context=context
)
print(ret1["answer"])

# 第二個問題（system prompt + 上下文會被 RadixAttention 快取）
ret2 = rag_qa.run(
    question="DGX Spark 支援多大的模型？",
    context=context
)
print(ret2["answer"])
# 第二個問題會比第一個快，因為前綴被快取了！
```

### 11-4-2 結構化資料提取

```python
@sgl.function
def extract_info(s, text):
    s += "請從以下文字中提取結構化資訊，輸出 JSON 格式：\n\n"
    s += f"文字：{text}\n\n"
    s += "請輸出以下格式的 JSON：\n"
    s += '{"name": "...", "age": ..., "city": "...", "skills": [...]}\n\n'
    s += "JSON：" + sgl.gen(
        "json_output",
        max_tokens=256,
        # 強制輸出 JSON 格式
        regex=r'\{[^}]*\}'
    )

result = extract_info.run(
    text="李明是一位 28 歲的資料科學家，住在上海。他精通 Python、TensorFlow 和 PyTorch。"
)
print(result["json_output"])
# 輸出：{"name": "李明", "age": 28, "city": "上海", "skills": ["Python", "TensorFlow", "PyTorch"]}
```

### 11-4-3 多步驟工作流程

```python
@sgl.function
def code_review(s, code):
    # 步驟 1：理解程式碼
    s += "請審查以下程式碼：\n\n"
    s += f"```python\n{code}\n```\n\n"
    
    s += "步驟 1 - 理解：這段程式碼在做什麼？\n"
    s += sgl.gen("understanding", stop="\n\n") + "\n\n"
    
    # 步驟 2：找出問題
    s += "步驟 2 - 問題：有什麼潛在問題或 bug？\n"
    s += sgl.gen("issues", stop="\n\n") + "\n\n"
    
    # 步驟 3：建議改進
    s += "步驟 3 - 建議：如何改進？\n"
    s += sgl.gen("suggestions", stop="\n\n") + "\n\n"
    
    # 步驟 4：評分
    s += "步驟 4 - 評分（1-10 分）："
    s += sgl.gen("score", max_tokens=5, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

result = code_review.run(
    code="""
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
"""
)
print(result["understanding"])
print(result["issues"])
print(result["suggestions"])
print(result["score"])
```

---

## 11-5 清理

```bash
# 停止基本伺服器
docker stop sglang
docker rm sglang

# 停止推測性解碼伺服器
docker stop sglang-eagle sglang-draft
docker rm sglang-eagle sglang-draft

# 停止大型模型伺服器
docker stop sglang-large
docker rm sglang-large

# 移除映像檔
docker rmi lmsysorg/sglang:latest
```

---

## 11-6 疑難排解 FAQ

### Q1：伺服器啟動後無法連線？

```bash
# 確認 port 正確（SGLang 預設使用 30000）
curl http://localhost:30000/health
# 預期回應：{"status": "healthy"}

# 查看日誌
docker logs sglang

# 常見原因：
# 1. 模型下載中（首次需要時間）
# 2. Port 被其他行程佔用
# 3. GPU 記憶體不足
```

### Q2：記憶體不足？

```bash
# 解決方案 1：降低 mem-fraction-static
--mem-fraction-static 0.75

# 解決方案 2：減少上下文長度
--context-length 4096

# 解決方案 3：使用更小的模型
--model-path Qwen/Qwen3-8B

# 監控記憶體
watch -n 1 nvidia-smi
```

### Q3：EAGLE-3 模型下載失敗？

```bash
# 確認模型名稱正確
# EAGLE-3 模型通常與目標模型在同一個 repo 或獨立的 repo

# 手動預下載
huggingface-cli download Qwen/Qwen3-8B-EAGLE-3

# 確認有足夠的磁碟空間
df -h ~/.cache/huggingface
```

### Q4：加速效果不明顯？

推測性解碼的效果取決於以下因素：

| 因素 | 影響 | 改善方法 |
|------|------|---------|
| 草稿模型匹配度 | 低匹配度 = 低命中率 | 選擇相同架構的小模型 |
| 輸出內容類型 | 程式碼 > 技術文 > 創意文字 | 對創意內容不建議使用 |
| 推測步數 | 太多 = 驗證成本高 | 調整 --speculative-num-steps |
| 溫度設定 | 高溫度 = 難以預測 | 降低 temperature |

**調校建議：**

```bash
# 如果命中率低，減少推測步數
--speculative-num-steps 3

# 如果命中率高，增加推測步數
--speculative-num-steps 7
```

### Q5：如何監控 SGLang 的效能？

```bash
# SGLang 提供內建的監控端點
curl http://localhost:30000/metrics

# 關鍵指標：
# prompt_tokens_total - 總輸入 token 數
# completion_tokens_total - 總輸出 token 數
# time_to_first_token_seconds - 首次回應時間
# inter_token_latency_seconds - token 間延遲
# cache_hit_rate - RadixAttention 快取命中率
```

### Q6：SGLang 和 vLLM 該選哪個？

| 你的需求 | 推薦 |
|---------|------|
| 需要結構化生成（JSON、regex） | SGLang |
| 需要 RadixAttention 加速多輪對話 | SGLang |
| 需要複雜的多步驟工作流程 | SGLang |
| 追求最高吞吐量 | vLLM |
| 需要完善的 OpenAI API 相容性 | vLLM |
| 需要 LoRA 多任務 | vLLM |
| 研究和實驗 | SGLang |

---

## 11-7 本章小結

::: success ✅ 你現在知道了
- SGLang 的 RadixAttention 可以快取重複前綴，大幅加速多輪對話和 RAG 場景
- 結構化生成讓模型輸出特定格式的內容（JSON、regex、choices），保證格式正確
- SGLang 提供 DSL（領域特定語言），讓 LLM 程式設計像寫 Python 一樣簡單
- EAGLE-3 和 Draft-Target 是兩種推測性解碼方法，可以帶來 1.5-2.5 倍的速度提升
- EAGLE-3 使用專門訓練的草稿頭，命中率更高但需要對應模型
- Draft-Target 使用任意小模型作為草稿模型，設定更簡單
- SGLang 在靈活性、結構化生成和研究用途上表現最佳
- RadixAttention 在 RAG 場景中可以帶來 3-5 倍的加速
:::

::: tip 🚀 下一章預告
我們已經介紹了六種推論引擎，到底該選哪個？下一章我們要做一次全面的比較，並介紹 NVIDIA 的 NIM 推論微服務，讓你一鍵部署最佳化的推論服務！

👉 [前往第 12 章：NIM 推論微服務與引擎總比較 →](/guide/chapter12/)
:::

::: info 📝 上一章
← [回到第 10 章：TensorRT-LLM](/guide/chapter10/)
:::
