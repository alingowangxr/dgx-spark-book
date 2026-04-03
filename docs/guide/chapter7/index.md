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
約 25 分鐘。
:::

---

## 7-1 在 DGX Spark 上安裝 LM Studio GUI

### 7-1-1 什麼是 LM Studio？

::: info 🤔 LM Studio 是什麼？
LM Studio 是一個跨平台的本地 LLM 推論工具，特色是：

1. **GUI + CLI 雙模式**：有漂亮的圖形介面，也有命令列工具
2. **Hugging Face 整合**：直接從 Hugging Face 下載 GGUF 模型
3. **OpenAI API 相容**：啟動本地 API 伺服器
4. **LM Link**：內建隧道服務，跨網路分享模型
5. **進階參數調整**：完整的推論參數控制

跟 Ollama 相比，LM Studio 更適合需要精細控制推論參數的使用者。
:::

### 7-1-2 用 Claude Code 下載 AppImage

LM Studio 提供 Linux AppImage 版本。告訴 Claude Code：

> 「幫我下載 LM Studio 的 Linux ARM64 AppImage，設定執行權限。」

Claude Code 會執行：

```bash
# 下載 LM Studio
wget https://lmstudio.ai/download/linux-arm64/lmstudio-linux-arm64-latest.AppImage

# 設定執行權限
chmod +x lmstudio-linux-arm64-latest.AppImage
```

### 7-1-3 透過 VNC 執行 GUI

因為 LM Studio 是圖形化應用程式，需要透過 VNC 遠端桌面來操作（第 4 章已設定）。

```bash
# 執行 LM Studio
./lmstudio-linux-arm64-latest.AppImage
```

在 VNC 視窗中，你會看到 LM Studio 的介面，包含：
- **左側欄**：Search、Chat、My Models
- **中間**：主要工作區
- **右側**：模型參數調整

### 7-1-4 在 GUI 中下載模型

LM Studio 內建模型搜尋和下載功能：

1. 點擊左側的 **Search** 圖示
2. 輸入模型名稱（例如 `Qwen3-8B`）
3. 選擇量化格式（建議 GGUF Q4_K_M）
4. 點擊 **Download**

::: tip 💡 量化格式選擇指南
| 格式 | 大小 | 品質 | 推薦場景 |
|------|------|------|---------|
| Q2_K | 最小 | 低 | 記憶體極度受限 |
| Q3_K_M | 小 | 中 | 平衡 |
| **Q4_K_M** | **中** | **高** | **最推薦** |
| Q5_K_M | 大 | 很高 | 高品質需求 |
| Q8_0 | 最大 | 接近 FP16 | 最佳品質 |
:::

### 7-1-5 載入模型與對話

1. 點擊左側的 **Chat** 圖示
2. 在上方選擇已下載的模型
3. 模型載入後，在下方輸入問題
4. 右側可以調整 temperature、top_p 等參數

---

## 7-2 用 Claude Code 安裝 lms CLI

### 7-2-1 確認 lms CLI

LM Studio 的 AppImage 內含 `lms` 命令列工具。

```bash
# 解壓縮 AppImage
./lmstudio-linux-arm64-latest.AppImage --appimage-extract

# 確認 lms 可用
cd squashfs-root
./lms --version
```

建立 alias 方便使用：

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
| 批次操作 | ❌ | ✅ |

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

### 7-3-2 搜尋和下載新模型

```bash
# 搜尋模型
lms search qwen3

# 下載特定版本
lms get qwen3-8b-Q4_K_M

# 下載並指定路徑
lms get qwen3-8b-Q4_K_M --path /models/
```

### 7-3-3 載入模型並測試

```bash
# 載入模型並進入對話模式
lms load qwen3-8b-Q4_K_M

# 測試對話
lms chat

# 在對話模式中輸入問題
>>> 你好！
```

---

## 7-4 啟動 API 服務

### 7-4-1 啟動伺服器

```bash
# 啟動 OpenAI 相容 API 服務
lms server start
```

預設監聽 `localhost:1234`。

**開放給外部連線**：

```bash
lms server start --host 0.0.0.0 --port 1234
```

### 7-4-2 從個人電腦測試連線

```bash
# 列出可用模型
curl http://DGX_Spark_IP:1234/v1/models

# 測試對話
curl http://DGX_Spark_IP:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-Q4_K_M",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'
```

### 7-4-3 從個人電腦呼叫 API

**Python（OpenAI SDK）**：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://DGX_Spark_IP:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="qwen3-8b-Q4_K_M",
    messages=[{"role": "user", "content": "你好！"}]
)

print(response.choices[0].message.content)
```

### 7-4-4 連接 Claude Code

```bash
export ANTHROPIC_BASE_URL="http://localhost:1234/v1"
export ANTHROPIC_API_KEY="lm-studio"
claude
```

---

## 7-5 LM Link：跨網路存取

### 7-5-1 什麼是 LM Link

LM Link 是 LM Studio 內建的隧道服務，類似 ngrok。讓你可以把本地的 API 服務分享給其他人，不需要設定 Port Forwarding。

### 7-5-2 設定 LM Link

```bash
# 啟動 LM Link
lms link start
```

系統會產生一個公開網址，例如：
```
https://abc123.lm-link.app
```

你可以把這個網址分享給其他人，他們就能透過網路呼叫你的模型 API。

```bash
# 關閉 LM Link
lms link stop
```

::: warning ⚠️ 安全提醒
- 只在需要時開啟
- 使用完畢後立即關閉
- 不要在 LM Link 上暴露敏感資料
:::

---

## 7-6 lms vs. Ollama：何時用哪個

| 特性 | Ollama | LM Studio (lms) |
|------|--------|-----------------|
| 安裝難度 | ⭐ 最易 | ⭐⭐ |
| GUI | ❌ | ✅ 內建 |
| 模型格式 | 自有 | GGUF |
| 模型來源 | Ollama 官方 | Hugging Face |
| 進階參數 | 有限 | 完整 |
| 跨網路分享 | 自行設定 | LM Link 內建 |
| 適合對象 | 快速體驗 | 精細控制 |

**使用建議**：
- **日常推論**：Ollama（簡單快速）
- **參數調校**：LM Studio（完整控制）
- **兩者搭配**：Ollama 做日常，LM Studio 做測試

---

## 7-7 模型載入優化：榨出最大效能

### 7-7-1 Flash Attention：免費的效能提升

```bash
lms server start --flash-attention
```

| 效果 | 數值 |
|------|------|
| 推論速度提升 | 15-25% |
| 記憶體減少 | 10-15% |
| 品質損失 | 無 |

### 7-7-2 KV Cache 量化

```bash
lms server start --kv-cache-quantization q8_0
```

| 選項 | 記憶體 | 品質 | 推薦 |
|------|--------|------|------|
| f16 | 100% | 無損失 | 預設 |
| **q8_0** | **~50%** | **極小** | ✅ **推薦** |
| q4_0 | ~25% | 輕微 | 記憶體不足 |

### 7-7-3 上下文長度

```bash
lms server start --context-length 8192
```

| 長度 | 額外記憶體（8B） | 適合場景 |
|------|-----------------|---------|
| 2048 | ~500 MB | 簡短問答 |
| **4096** | **~1 GB** | **一般對話** |
| 8192 | ~2 GB | 長文分析 |
| 32768 | ~8 GB | 文件摘要 |

### 7-7-4 DGX Spark 推薦參數組合

```bash
lms server start \
  --flash-attention \
  --kv-cache-quantization q8_0 \
  --context-length 8192 \
  --eval-batch-size 4 \
  --gpu-layers 999 \
  --threads 20
```

---

## 7-8 卸載與清理

```bash
# 停止伺服器
lms server stop

# 刪除模型
lms rm qwen3-8b-Q4_K_M

# 清理快取
lms cache clean

# 移除 AppImage
rm ~/lmstudio-linux-arm64-latest.AppImage
rm -rf ~/squashfs-root
```

---

## 7-9 本章小結

::: success ✅ 你現在知道了
- LM Studio 提供 GUI 和 CLI 兩種操作方式
- lms CLI 可以管理模型、啟動 API 服務
- LM Link 讓你輕鬆跨網路分享模型
- Flash Attention 和 KV Cache 量化是最重要的優化技術
- 推薦的參數組合可以在 DGX Spark 上榨出最大效能
:::

::: tip 🚀 下一章預告
接下來我們要看看更輕量、更原生的推論方案 — llama.cpp。它能讓你直接編譯並執行 GGUF 模型，不需要額外的執行環境！

👉 [前往第 8 章：llama.cpp 與 Nemotron →](/guide/chapter8/)
:::

::: info 📝 上一章
← [回到第 6 章：Open WebUI](/guide/chapter6/)
:::
