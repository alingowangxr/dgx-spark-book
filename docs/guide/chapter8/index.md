# 第 8 章：llama.cpp 與 Nemotron — 輕量原生推論

::: tip 🎯 本章你將學到什麼
- 什麼是 GGUF 格式
- 從原始碼編譯 llama.cpp
- 用 llama-server 啟動推論服務
- 推測性解碼實戰
- llama.cpp 的最新發展：MXFP4、內建 WebUI、多模態
- 完整效能調校指南
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。編譯約需 5-10 分鐘。
:::

---

## 8-1 llama.cpp 基礎

### 8-1-1 什麼是 llama.cpp？

::: info 🤔 llama.cpp 是什麼？
llama.cpp 是一個用 C/C++ 寫的 LLM 推論框架，由 Georgi Gerganov 開發。它的核心價值：

1. **不需要 Python**：純 C/C++，依賴極少
2. **GGUF 格式**：專為推論設計的高效模型格式
3. **跨平台**：支援 x86、ARM、Apple Silicon
4. **GPU 加速**：支援 CUDA、Metal、Vulkan
5. **極致效能**：在相同硬體上通常比 Python 方案快 10-20%

在 DGX Spark 上，llama.cpp 是追求極致效能的首選。
:::

### 8-1-2 什麼是 GGUF？

GGUF（GGML Universal File）是 llama.cpp 專案開發的模型格式。

**為什麼需要 GGUF？**

傳統的模型格式（如 PyTorch 的 `.bin` 或 `.safetensors`）是為訓練設計的，推論時效率不高。

GGUF 是專門為**推論**設計的格式：
- 內建量化支援（Q4、Q5、Q8 等）
- 所有中繼資料都存在同一個檔案中
- 載入速度快、記憶體用量低
- 一個檔案包含所有資訊，不需要額外的 config 檔案

**GGUF 量化選項完整比較**：

| 量化 | 大小（相對於 FP16） | 品質損失 | 記憶體（8B 模型） | 推薦用途 |
|------|-------------------|---------|-----------------|---------|
| Q2_K | ~25% | 明顯 | ~2 GB | 記憶體極度受限 |
| Q3_K_S | ~30% | 中等 | ~2.5 GB | 低記憶體 |
| Q3_K_M | ~35% | 輕微 | ~3 GB | 平衡 |
| **Q4_K_M** | **~50%** | **極小** | **~4.5 GB** | **✅ 最推薦** |
| Q5_K_S | ~55% | 極小 | ~5 GB | 高品質 |
| Q5_K_M | ~60% | 幾乎沒有 | ~5.5 GB | 高品質需求 |
| Q6_K | ~65% | 無 | ~6 GB | 接近 FP16 |
| Q8_0 | ~75% | 無 | ~7 GB | 最佳品質 |
| FP16 | 100% | 無 | ~16 GB | 訓練/研究 |

### 8-1-3 SM_121：DGX Spark 的 GPU 架構代號

SM_121 是 NVIDIA 給 DGX Spark GPU 的 compute capability 代號。編譯 llama.cpp 時需要指定這個代號，才能啟用 GPU 加速。

---

## 8-2 用 Claude Code 編譯 llama.cpp

### 8-2-1 確認編譯環境

```bash
# 確認編譯工具已安裝
gcc --version
cmake --version
```

如果缺少，安裝它們：

```bash
sudo apt install -y build-essential cmake git
```

### 8-2-2 複製原始碼並編譯

告訴 Claude Code：

> 「幫我從原始碼編譯 llama.cpp，啟用 CUDA 支援，GPU 架構設為 SM_121。」

Claude Code 會執行：

```bash
# 複製原始碼
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 建立編譯目錄
mkdir build && cd build

# 設定編譯選項
cmake .. \
  -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DLLAMA_CURL=ON

# 編譯
cmake --build . --config Release -j $(nproc)
```

::: tip 💡 編譯時間
首次編譯約需 5-10 分鐘。編譯完成後，執行檔會在 `build/bin/` 目錄中。

主要執行檔：
- `llama-server`：API 伺服器（含內建 WebUI）
- `llama-cli`：命令列對話工具
- `llama-bench`：效能測試工具
- `llama-quantize`：模型量化工具
:::

### 8-2-3 驗證編譯結果

```bash
# 回到 llama.cpp 根目錄
cd ..

# 確認執行檔存在
ls build/bin/llama-server
ls build/bin/llama-cli
ls build/bin/llama-bench

# 測試版本
./build/bin/llama-server --version
```

---

## 8-3 下載 Nemotron-3-Nano 模型

### 8-3-1 安裝 Hugging Face CLI

```bash
# 用 uv 安裝
uv pip install huggingface_hub

# 登入（如果需要下載需要授權的模型）
huggingface-cli login
```

### 8-3-2 下載 GGUF 模型

```bash
# 建立模型目錄
mkdir -p ~/models

# 下載 Nemotron-3-Nano 的 GGUF 版本
huggingface-cli download nvidia/Nemotron-3-Nano-GGUF \
  --local-dir ~/models/nemotron-nano \
  --include "*.gguf"
```

下載完成後：

```bash
# 確認檔案
ls -lh ~/models/nemotron-nano/
# 輸出範例：nemotron-3-nano-q4_k_m.gguf  6.8G
```

---

## 8-4 啟動 llama-server

### 8-4-1 啟動伺服器

```bash
./build/bin/llama-server \
  --model ~/models/nemotron-nano/*.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --gpu-layers 999 \
  --flash-attn
```

**參數解釋**：

| 參數 | 說明 |
|------|------|
| `--model` | 模型路徑 |
| `--host 0.0.0.0` | 監聽所有網路介面 |
| `--port 8080` | 監聽 port |
| `--ctx-size 8192` | 上下文長度 |
| `--gpu-layers 999` | 所有層都放 GPU |
| `--flash-attn` | 啟用 Flash Attention |

### 8-4-2 內建 WebUI

llama-server 內建了一個簡單的 WebUI。啟動伺服器後，直接用瀏覽器打開：

```
http://DGX_Spark_IP:8080
```

這個 WebUI 功能包括：
- 基本對話
- 參數調整（temperature、top_p、max tokens）
- 系統提示詞設定

### 8-4-3 測試 API

```bash
# 測試 OpenAI 相容端點
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-nano",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "stream": false
  }'
```

### 8-4-4 用 llama-cli 測試

```bash
./build/bin/llama-cli \
  --model ~/models/nemotron-nano/*.gguf \
  --prompt "請解釋什麼是人工智慧" \
  --n-predict 256 \
  --temp 0.7 \
  --gpu-layers 999
```

---

## 8-5 效能調校

### 8-5-1 Flash Attention

```bash
./build/bin/llama-server \
  --model ~/models/nemotron-nano/*.gguf \
  --flash-attn
```

Flash Attention 在 llama.cpp 中已經相當成熟，建議一律開啟。

**效果**：
- 推論速度提升：15-25%
- 記憶體用量減少：10-15%
- 品質損失：無

### 8-5-2 KV Cache 量化

```bash
./build/bin/llama-server \
  --model ~/models/nemotron-nano/*.gguf \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

| 選項 | 記憶體 | 品質影響 | 推薦 |
|------|--------|---------|------|
| f16 | 100% | 無 | 預設 |
| **q8_0** | **~50%** | **極小** | ✅ **推薦** |
| q4_0 | ~25% | 輕微 | 記憶體不足 |

### 8-5-3 其他實用參數

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--threads` | CPU 線程數 | 20（DGX Spark 核心數） |
| `--batch-size` | 批次大小 | 512-2048 |
| `--ubatch-size` | micro-batch 大小 | 512 |
| `--ctx-size` | 上下文長度 | 8192 |
| `--parallel` | 平行處理數 | 4 |

### 8-5-4 DGX Spark 推薦參數組合

```bash
./build/bin/llama-server \
  --model ~/models/nemotron-nano/*.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --gpu-layers 999 \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --threads 20 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel 4
```

### 8-5-5 Benchmark 測試

```bash
./build/bin/llama-bench \
  --model ~/models/nemotron-nano/*.gguf \
  --gpu-layers 999 \
  --batch-size 512
```

輸出範例：

```
| Model                  | Test | T/s    |
|------------------------|------|--------|
| nemotron-nano Q4_K_M   | pp   | 1250.5 |
| nemotron-nano Q4_K_M   | tg   | 85.3   |
```

- **pp**（prompt processing）：處理輸入的速度（tokens/s）
- **tg**（token generation）：生成輸出的速度（tokens/s）

---

## 8-6 推測性解碼

### 8-6-1 什麼是推測性解碼？

::: info 🤔 推測性解碼原理
用一個小模型（draft model）來「猜測」大模型的輸出，然後用大模型驗證。如果猜對了就直接用，不需要大模型重新計算。

```
小模型猜測：「今天天氣很好，我想去公園散步」
大模型驗證：✅ 正確 → 直接輸出
           ❌ 錯誤 → 重新生成
```
:::

### 8-6-2 實戰設定

```bash
./build/bin/llama-server \
  --model ~/models/large-model.gguf \
  --model-draft ~/models/small-model.gguf \
  --gpu-layers-draft 999 \
  --gpu-layers 999 \
  --draft 8 \
  --flash-attn
```

| 參數 | 說明 |
|------|------|
| `--model-draft` | 草稿模型路徑 |
| `--gpu-layers-draft` | 草稿模型的 GPU 層數 |
| `--draft` | 每次猜測的 token 數 |

### 8-6-3 加速效果

| 模型組合 | 原始速度 | 推測性解碼 | 加速比 |
|---------|---------|-----------|--------|
| 70B + 8B | ~20 t/s | ~30 t/s | **1.5x** |
| 120B + 8B | ~12 t/s | ~18 t/s | **1.5x** |

---

## 8-7 多模態推論

llama.cpp 也支援多模態模型（如 LLaVA、Qwen2.5-VL）：

```bash
./build/bin/llama-server \
  --model ~/models/llava-gguf/*.gguf \
  --mmproj ~/models/llava-gguf/mmproj*.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --flash-attn
```

`--mmproj` 是多模態投影器（multimodal projector），多模態模型需要這個檔案才能處理影像。

**測試影像分析**：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "這張圖片中有什麼？"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
      }
    ],
    "stream": false
  }'
```

---

## 8-8 Nemotron-3-Nano vs. 其他工具的 Nemotron

### 8-8-1 三種方式比較

| 方式 | 優點 | 缺點 | 適合場景 |
|------|------|------|---------|
| **Ollama** | 最簡單、自動下載 | 參數調整有限 | 快速體驗 |
| **LM Studio** | GUI、進階參數 | 需要 AppImage | 模型測試 |
| **llama.cpp** | 最靈活、效能最佳 | 需要編譯 | 生產環境、最佳化 |

### 8-8-2 效能比較

| 工具 | 輸出速度（t/s） | 首次回應（ms） | 記憶體用量 |
|------|---------------|--------------|-----------|
| Ollama | ~75 | ~200 | ~6 GB |
| LM Studio | ~70 | ~250 | ~6 GB |
| **llama.cpp** | **~85** | **~180** | **~6 GB** |

---

## 8-9 llama.cpp 的最新發展

### 8-9-1 MXFP4 原生支援

最新版本的 llama.cpp 已經支援 MXFP4（Mixed eXponential Float Point 4-bit）格式，這是 NVIDIA Blackwell 架構原生的 4-bit 格式。

```bash
./build/bin/llama-server \
  --model model.mxfp4.gguf \
  --flash-attn
```

### 8-9-2 程式碼補全：VS Code 和 Vim 擴充

llama.cpp 可以作為程式碼補全後端：

**VS Code**：
- 安裝 **Continue** 擴充
- 設定後端為 `http://localhost:8080`

**Vim**：
- 使用 **nvim-cmp** + **llama.cpp** 外掛

### 8-9-3 Hugging Face GGUF 整合

Hugging Face 現在直接在模型頁面顯示 GGUF 格式的量化版本：

1. 在 Hugging Face 搜尋想要的模型
2. 找到「Files and versions」標籤
3. 找到 GGUF 格式的檔案
4. 用 `huggingface-cli download` 下載

---

## 8-10 清理與移除

```bash
# 停止伺服器（如果在背景執行）
pkill llama-server

# 移除編譯目錄
rm -rf llama.cpp

# 移除模型
rm -rf ~/models/nemotron-nano
```

---

## 8-11 本章小結

::: success ✅ 你現在知道了
- GGUF 是專為推論設計的高效模型格式
- llama.cpp 可以從原始碼編譯，完全控制所有參數
- llama-server 提供 OpenAI 相容 API 和內建 WebUI
- 推測性解碼可以用小模型加速大模型的推論
- llama.cpp 持續更新，支援 MXFP4、多模態、程式碼補全等功能
- 在相同硬體上，llama.cpp 通常比 Python 方案快 10-20%
:::

::: tip 🚀 第二篇完結！
恭喜！你已經完成了「LLM 推論入門」篇，學會了四種推論工具：
- **Ollama**：最簡單
- **Open WebUI**：最漂亮的介面
- **LM Studio**：GUI + CLI 雙模式
- **llama.cpp**：最靈活、效能最佳

接下來我們要進入更進階的推論框架，追求更高的吞吐量和更低的延遲！

👉 [前往第 9 章：vLLM — 高吞吐量推論伺服器 →](/guide/chapter9/)
:::

::: info 📝 上一章
← [回到第 7 章：LM Studio](/guide/chapter7/)
:::
