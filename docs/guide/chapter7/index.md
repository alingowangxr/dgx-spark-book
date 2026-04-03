# 第 7 章：LM Studio — Headless 模型服務

::: tip 🎯 本章你將學到什麼
- 在 DGX Spark 上安裝 LM Studio GUI（透過 VNC）
- 用 lms CLI 在命令列中管理模型
- 啟動 API 服務，從個人電腦呼叫
- LM Link 跨網路存取
- LM Studio vs. Ollama 的選擇建議
- 模型載入優化：Flash Attention、KV Cache 量化等
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 7-1 在 DGX Spark 上安裝 LM Studio GUI

### 7-1-1 用 Claude Code 下載 AppImage

LM Studio 提供 Linux AppImage 版本。告訴 Claude Code：

> 「幫我下載 LM Studio 的 Linux ARM64 AppImage，設定執行權限。」

Claude Code 會執行：

```bash
# 下載 LM Studio（以最新版本為準）
wget https://lmstudio.ai/download/linux-arm64/lmstudio-linux-arm64-latest.AppImage

# 設定執行權限
chmod +x lmstudio-linux-arm64-latest.AppImage
```

### 7-1-2 透過 VNC 執行 GUI

因為 LM Studio 是圖形化應用程式，需要透過 VNC 遠端桌面來操作（第 4 章已設定）。

```bash
# 執行 LM Studio
./lmstudio-linux-arm64-latest.AppImage
```

在 VNC 視窗中，你會看到 LM Studio 的介面。

### 7-1-3 在 GUI 中下載模型

LM Studio 內建模型搜尋和下載功能：

1. 點擊左側的 **Search** 圖示
2. 輸入模型名稱（例如 `Qwen3-8B`）
3. 選擇量化格式（建議 GGUF Q4_K_M）
4. 點擊 **Download**

### 7-1-4 載入模型與對話

1. 點擊左側的 **Chat** 圖示
2. 在上方選擇已下載的模型
3. 模型載入後，在下方輸入問題

---

## 7-2 用 Claude Code 安裝 lms CLI

### 7-2-1 確認 lms CLI

LM Studio 的 AppImage 內含 `lms` 命令列工具。

```bash
# 確認 lms 可用
./lmstudio-linux-arm64-latest.AppImage --appimage-extract
cd squashfs-root
./lms --version
```

或者建立一個 alias 方便使用：

```bash
echo 'alias lms="~/lmstudio-linux-arm64-latest.AppImage -- lms"' >> ~/.zshrc
source ~/.zshrc
```

### 7-2-2 GUI 和 lms 的關係

| 功能 | GUI | lms CLI |
|------|-----|---------|
| 搜尋模型 | ✅ | ✅ |
| 下載模型 | ✅ | ✅ |
| 對話 | ✅ | ✅ |
| 啟動 API 服務 | ✅ | ✅ |
| 進階參數調整 | 部分 | ✅ 完整 |
| 自動化腳本 | ❌ | ✅ |

**建議**：初次使用可以用 GUI 熟悉介面，之後用 lms CLI 進行日常操作。

---

## 7-3 模型管理

### 7-3-1 列出已下載的模型

```bash
lms ls
```

輸出範例：

```
NAME                          SIZE    QUANT    FAMILY
qwen3-8b.Q4_K_M.gguf          5.2 GB  Q4_K_M   Qwen3
nemotron-3-nano.Q5_K_M.gguf   6.8 GB  Q5_K_M   Nemotron
```

### 7-3-2 下載新模型

```bash
# 搜尋模型
lms search qwen3

# 下載模型
lms get qwen3-8b-Q4_K_M
```

### 7-3-3 載入模型並測試

```bash
# 載入模型並進入對話模式
lms load qwen3-8b-Q4_K_M

# 測試對話
lms chat
```

---

## 7-4 啟動 API 服務

### 7-4-1 啟動伺服器

```bash
# 啟動 OpenAI 相容 API 服務
lms server start
```

預設監聽 `localhost:1234`。

### 7-4-2 從個人電腦測試連線

```bash
# 從個人電腦測試（把 IP 換成 DGX Spark 的 IP）
curl http://DGX_Spark_IP:1234/v1/models
```

### 7-4-3 從個人電腦呼叫 API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://DGX_Spark_IP:1234/v1",
    api_key="lm-studio"  # LM Studio 不需要真正的 key
)

response = client.chat.completions.create(
    model="qwen3-8b-Q4_K_M",
    messages=[{"role": "user", "content": "你好！"}]
)

print(response.choices[0].message.content)
```

### 7-4-4 連接 Claude Code

```bash
# 設定環境變數
export ANTHROPIC_BASE_URL="http://localhost:1234/v1"
export ANTHROPIC_API_KEY="lm-studio"

claude
```

---

## 7-5 LM Link：跨網路存取

### 7-5-1 什麼是 LM Link

LM Link 是 LM Studio 內建的隧道服務，讓你可以把本地的 API 服務分享給其他人，不需要設定 Port Forwarding。

類似 ngrok 的概念，但內建在 LM Studio 中。

### 7-5-2 設定 LM Link

```bash
# 啟動 LM Link
lms link start
```

系統會產生一個公開網址，你可以把這個網址分享給其他人，他們就能透過網路呼叫你的模型 API。

::: warning ⚠️ 安全提醒
LM Link 會把你的模型 API 公開到網路上。建議：
- 只在需要時開啟
- 不要分享敏感模型
- 使用完畢後立即關閉
:::

```bash
# 關閉 LM Link
lms link stop
```

---

## 7-6 lms vs. Ollama：何時用哪個

### 7-6-1 功能比較

| 功能 | Ollama | LM Studio (lms) |
|------|--------|-----------------|
| 安裝難度 | 超簡單 | 中等 |
| GUI | 無（需搭配其他工具） | 內建 |
| 模型格式 | 自有格式 | GGUF |
| 模型來源 | Ollama 官方庫 | Hugging Face |
| API | OpenAI 相容 | OpenAI 相容 |
| 進階參數 | 有限 | 完整 |
| 跨網路分享 | 需自行設定 | LM Link 內建 |
| 社群大小 | 大 | 中等 |

### 7-6-2 使用建議

- **用 Ollama**：想要最簡單的體驗、快速下載模型
- **用 LM Studio**：需要 GUI、想從 Hugging Face 下載特定 GGUF 模型、需要進階參數調整
- **兩者都用**：Ollama 做日常推論，LM Studio 做模型測試和參數調校

---

## 7-7 進階操作

### 7-7-1 Speculative Decoding

推測性解碼用一個小模型來「猜測」大模型的輸出，如果猜對了就直接用，可以加速推論。

```bash
lms server start \
  --speculative-decoding \
  --draft-model small-model-Q4 \
  --target-model large-model-Q4
```

### 7-7-2 自訂推論參數

```bash
lms server start \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 4096 \
  --repeat-penalty 1.1
```

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `temperature` | 創造性（越高越有創意） | 0.3-0.7 |
| `top-p` | 候選詞比例 | 0.9 |
| `max-tokens` | 最大輸出長度 | 4096 |
| `repeat-penalty` | 重複懲罰 | 1.1 |

### 7-7-3 卸載與清理

```bash
# 停止伺服器
lms server stop

# 刪除模型
lms rm qwen3-8b-Q4_K_M

# 清理所有快取
lms cache clean
```

---

## 7-8 模型載入優化：榨出最大效能

### 7-8-1 Flash Attention：免費的效能提升

Flash Attention 是一種最佳化技術，可以加速注意力機制的計算，同時減少記憶體用量。

```bash
lms server start --flash-attention
```

::: tip 💡 效果
在 DGX Spark 上，Flash Attention 可以帶來：
- 推論速度提升：15-25%
- 記憶體用量減少：10-15%
- 完全免費，沒有品質損失
:::

### 7-8-2 KV Cache 量化：省記憶體的關鍵

KV Cache 量化把注意力機制的快取資料量化為較低精度，可以大幅減少記憶體用量。

```bash
lms server start --kv-cache-quantization q8_0
```

量化選項：

| 選項 | 記憶體用量 | 品質影響 | 建議 |
|------|-----------|---------|------|
| `f16` | 100% | 無 | 預設 |
| `q8_0` | ~50% | 極小 | ✅ 推薦 |
| `q4_0` | ~25% | 輕微 | 記憶體不足時 |

### 7-8-3 上下文長度：用多少開多少

上下文長度決定了模型能「記住」多少之前的對話。越長需要的記憶體越多。

```bash
lms server start --context-length 8192
```

| 上下文長度 | 額外記憶體（8B 模型） | 適合場景 |
|-----------|---------------------|---------|
| 2048 | ~500 MB | 簡短問答 |
| 4096 | ~1 GB | 一般對話 |
| 8192 | ~2 GB | 長文分析 |
| 32768 | ~8 GB | 文件摘要 |

::: tip 💡 建議
不要開太大，用多少開多少。128GB 雖然很多，但如果同時跑多個服務還是會不夠。
:::

### 7-8-4 Eval Batch Size：Prefill 加速

Eval Batch Size 影響模型處理輸入（prefill）階段的速度。

```bash
lms server start --eval-batch-size 4
```

較大的 batch size 可以加速長文本的處理，但會增加記憶體用量。

### 7-8-5 DGX Spark 推薦的載入參數組合

```bash
# 最佳化組合（適合 120B 模型）
lms server start \
  --flash-attention \
  --kv-cache-quantization q8_0 \
  --context-length 8192 \
  --eval-batch-size 4 \
  --gpu-layers 999 \
  --threads 20
```

| 參數 | 值 | 原因 |
|------|-----|------|
| `flash-attention` | 開啟 | 免費加速 |
| `kv-cache-quantization` | q8_0 | 省記憶體，品質幾乎不受影響 |
| `context-length` | 8192 | 夠用且不浪費 |
| `eval-batch-size` | 4 | 平衡速度和記憶體 |
| `gpu-layers` | 999 | 所有層都放 GPU |
| `threads` | 20 | 配合 20 核 CPU |

---

## 7-9 本章小結

::: success ✅ 你現在知道了
- LM Studio 提供 GUI 和 CLI 兩種操作方式
- lms CLI 可以管理模型、啟動 API 服務
- LM Link 讓你輕鬆跨網路分享模型
- Flash Attention 和 KV Cache 量化是兩個最重要的優化技術
- 推薦的參數組合可以在 DGX Spark 上榨出最大效能
:::

::: tip 🚀 下一章預告
接下來我們要看看更輕量、更原生的推論方案 — llama.cpp。它能讓你直接編譯並執行 GGUF 模型，不需要額外的執行環境！

👉 [前往第 8 章：llama.cpp 與 Nemotron →](/guide/chapter8/)
:::

::: info 📝 上一章
← [回到第 6 章：Open WebUI](/guide/chapter6/)
:::
