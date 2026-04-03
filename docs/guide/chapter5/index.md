# 第 5 章：Ollama — 在 128 GB 上跑超大模型

::: tip 🎯 本章你將學到什麼
- 用 Claude Code 一鍵安裝 Ollama
- 用 Docker 執行 Ollama（進階做法）
- 下載並執行 120B 等級的超大模型
- 測試多模態模型的影像分析能力
- 把 Ollama 開放成遠端 API 服務
- 讓 Claude Code 使用本機 Ollama 模型
- Ollama 進階設定與效能調校
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。下載 120B 模型可能需要 30-60 分鐘（視網路速度而定）。
:::

---

## 5-1 用 Claude Code 安裝 Ollama

### 5-1-1 什麼是 Ollama？為什麼選它？

::: info 🤔 Ollama 是什麼？
Ollama 是目前最流行的本地 LLM 推論工具。它的核心價值是：

**把複雜的模型部署變成一行指令。**

傳統做法：
```
1. 下載模型權重（幾十 GB）
2. 安裝 Python 和相依套件
3. 寫載入程式碼
4. 設定 GPU 參數
5. 啟動伺服器
6. 除錯...
```

Ollama 做法：
```bash
ollama run qwen3-8b
# 搞定！
```
:::

**Ollama 的核心優勢**：

| 特色 | 說明 |
|------|------|
| **一行指令** | `ollama run <模型名>` 就能開始對話 |
| **自動下載** | 不需要手動找模型檔案，Ollama 自動下載 |
| **GPU 加速** | 自動偵測 GPU 並啟用加速 |
| **OpenAI API** | 提供 `/v1/chat/completions` 端點，相容所有 OpenAI 客戶端 |
| **多模型管理** | 輕鬆下載、切換、刪除模型 |
| **ARM64 支援** | 完美支援 DGX Spark 的 ARM 架構 |

### 5-1-2 安裝與驗證

現在，讓我們用 Claude Code 來安裝 Ollama。打開終端機，輸入：

```bash
claude
```

然後告訴 Claude Code：

> 「幫我安裝 Ollama，並確認服務正常運作。」

Claude Code 會自動執行以下指令：

```bash
# Ollama 官方安裝指令
curl -fsSL https://ollama.com/install.sh | sh
```

::: tip 💡 安裝腳本在做什麼？
上面的 curl 指令會：
1. 偵測你的系統架構（ARM64）
2. 下載對應的二進位檔（約 400 MB）
3. 安裝到 `/usr/local/bin/ollama`
4. 建立 `ollama` 使用者
5. 建立 systemd 服務（`/etc/systemd/system/ollama.service`）
6. 自動啟動服務
:::

安裝完成後，驗證：

```bash
# 確認 Ollama 版本
ollama --version
# 輸出範例：ollama version is 0.5.x

# 確認服務狀態
systemctl status ollama
# 應該看到：Active: active (running)
```

### 5-1-3 確認服務狀態

```bash
# 查看 Ollama 是否在監聽
curl http://localhost:11434
# 應該會看到：Ollama is running

# 查看已下載的模型（目前應該是空的）
ollama list
# 輸出：NAME    ID    SIZE    MODIFIED
# （空的，因為還沒下載任何模型）

# 查看服務日誌
journalctl -u ollama -f
```

### 5-1-4 設定 NVIDIA Sync 遠端存取

如果你用 NVIDIA Sync 管理 DGX Spark（第 2 章已設定），可以在 Sync 中直接看到 Ollama 服務狀態，也可以直接在 Sync 的終端機中執行 Ollama 指令。

---

## 5-2 用 Docker 執行 Ollama

### 5-2-1 為什麼用 Docker 版本？

| 直接安裝 | Docker 安裝 |
|---------|------------|
| 最簡單 | 環境隔離 |
| 系統級服務 | 容易清理（刪容器就好） |
| 適合個人使用 | 適合多人共用環境 |
| 更新需要重新執行腳本 | `docker pull` 即可更新 |

### 5-2-2 用 Claude Code 啟動容器

告訴 Claude Code：

> 「用 Docker 執行 Ollama，掛載本機模型目錄，確保 GPU 加速有開啟。」

Claude Code 會執行：

```bash
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -v /models:/models \
  --restart unless-stopped \
  ollama/ollama:latest
```

**參數解釋**：

| 參數 | 說明 |
|------|------|
| `--gpus all` | 啟用 GPU 加速（必要！） |
| `-p 11434:11434` | 把容器的 11434 port 對應到主機 |
| `-v ollama_data:/root/.ollama` | 持久化模型資料（刪容器不會丟模型） |
| `-v /models:/models` | 掛載本機模型目錄（可選） |
| `--restart unless-stopped` | 開機自動啟動 |

### 5-2-3 掛載本機模型目錄

如果你已經用非 Docker 方式下載了模型，可以掛載進去：

```bash
# 確認本機模型位置
ls ~/.ollama/models

# 用 Docker 執行時掛載
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ~/.ollama:/root/.ollama \
  ollama/ollama:latest
```

---

## 5-3 下載與執行超大模型

### 5-3-1 目前已下載的模型

```bash
# 列出所有已下載的模型
ollama list
```

剛安裝完成時應該是空的。讓我們開始下載模型。

### 5-3-2 模型命名規則

Ollama 的模型命名規則：

```
模型名稱:標籤
```

標籤通常是參數量 + 量化格式：

| 標籤 | 說明 | 大小 |
|------|------|------|
| `8b` | 8B 參數，預設量化（Q4） | ~5 GB |
| `70b` | 70B 參數，預設量化（Q4） | ~40 GB |
| `120b` | 120B 參數，預設量化（Q4） | ~60 GB |
| `70b:fp16` | 70B 參數，FP16 精度 | ~140 GB |

### 5-3-3 Nemotron-3-Super 120B

Nemotron-3-Super 是 NVIDIA 開源的 120B 參數模型，針對對話和指令遵循做了最佳化。

```bash
# 下載 Nemotron-3-Super 120B
ollama pull nemotron-super:120b
```

::: tip 💡 下載時間估算
| 網路速度 | 120B 模型（~60GB）下載時間 |
|---------|--------------------------|
| 100 Mbps | ~1.5 小時 |
| 500 Mbps | ~20 分鐘 |
| 1 Gbps | ~10 分鐘 |

建議在睡前或出門前開始下載。
:::

下載完成後，開始對話：

```bash
ollama run nemotron-super:120b
```

你會在終端機中看到對話介面：

```
>>> 你好，請介紹一下你自己。
```

輸入任何問題來測試。按 `Ctrl+D` 結束對話。

### 5-3-4 GPT-OSS 120B

GPT-OSS 是另一個開源的 120B 模型：

```bash
# 下載 GPT-OSS 120B
ollama pull gpt-oss:120b

# 開始對話
ollama run gpt-oss:120b
```

### 5-3-5 推薦的入門模型

如果你是第一次使用，建議從小模型開始：

```bash
# 快速測試（~5 GB）
ollama pull qwen3-8b
ollama run qwen3-8b

# 進階測試（~40 GB）
ollama pull qwen3-70b
ollama run qwen3-70b

# 終極測試（~60 GB）
ollama pull qwen3.5:122b
ollama run qwen3.5:122b
```

---

## 5-4 多模態模型實測：Qwen3.5 122B

### 5-4-1 文字對話

Qwen3.5 122B 是一個多模態模型，支援文字和影像輸入。

```bash
# 下載並執行
ollama pull qwen3.5:122b
ollama run qwen3.5:122b
```

測試文字對話：

```
>>> 請用三句話解釋量子計算。

量子計算是一種利用量子力學原理（如疊加和糾纏）來處理資訊的計算方式，
與傳統電腦使用位元（0 或 1）不同，量子電腦使用量子位元（qubit），
可以同時表示 0 和 1 的狀態。這使得量子電腦在特定問題上（如因數分解、
最佳化問題）能夠實現指數級的加速。雖然目前量子電腦仍處於早期發展階段，
但已經在密碼學、藥物研發和材料科學等領域展現出巨大的潛力。
```

### 5-4-2 影像分析

Qwen3.5 122B 也支援影像分析。

**方法一：命令列**

```bash
# 用影像檔案測試
ollama run qwen3.5:122b "這張圖片中有什麼？" /path/to/image.jpg
```

**方法二：API**

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:122b",
    "prompt": "這張圖片中有什麼？",
    "images": ["/path/to/image.jpg"],
    "stream": false
  }'
```

**方法三：Base64 編碼**

```bash
# 把圖片轉為 Base64
base64 -w 0 image.jpg > image.b64

# 用 API 傳送
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:122b",
    "prompt": "描述這張圖片",
    "images": ["'$(cat image.b64)'"],
    "stream": false
  }'
```

::: tip 💡 影像分析效能
在 DGX Spark 上，Qwen3.5 122B 的影像分析速度：
- 圖片載入：~2 秒
- 第一次回應：~5 秒
- 完整描述：~10 秒

這在 122B 模型中是非常快的。作為比較，同樣的任務在 RTX 4090（24GB）上根本跑不起來，因為模型裝不下。
:::

---

## 5-5 將 Ollama 開放成遠端服務

### 5-5-1 設定 OLLAMA_HOST

預設情況下，Ollama 只監聽 `localhost`。要讓其他裝置也能連線：

```bash
# 編輯 Ollama 服務設定
sudo systemctl edit ollama
```

貼入以下內容：

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

重新啟動服務：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

確認監聽狀態：

```bash
# 應該看到 0.0.0.0:11434
ss -tlnp | grep 11434
```

### 5-5-2 Ollama API 端點一覽

| 端點 | 方法 | 用途 | 範例 |
|------|------|------|------|
| `/api/generate` | POST | 文字生成 | 單次生成 |
| `/api/chat` | POST | 對話（多輪） | 聊天機器人 |
| `/api/embeddings` | POST | 取得嵌入向量 | RAG 檢索 |
| `/api/pull` | POST | 下載模型 | 遠端下載 |
| `/api/tags` | GET | 列出模型 | 查看已下載 |
| `/api/delete` | DELETE | 刪除模型 | 清理空間 |
| `/api/copy` | POST | 複製模型 | 備份 |
| `/api/show` | POST | 查看模型資訊 | 查看參數 |

### 5-5-3 從個人電腦呼叫遠端 Ollama

```bash
# 測試遠端 Ollama
curl http://DGX_Spark_IP:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:122b",
    "prompt": "你好！",
    "stream": false
  }'
```

### 5-5-4 用 Python 呼叫遠端 Ollama

**方法一：requests**

```python
import requests

url = "http://DGX_Spark_IP:11434/api/chat"
payload = {
    "model": "qwen3.5:122b",
    "messages": [
        {"role": "user", "content": "請用中文介紹你自己"}
    ],
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json()["message"]["content"])
```

**方法二：OpenAI SDK（推薦）**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://DGX_Spark_IP:11434/v1",
    api_key="ollama"  # Ollama 不需要 API key，但這個欄位必填
)

response = client.chat.completions.create(
    model="qwen3.5:122b",
    messages=[{"role": "user", "content": "你好！"}]
)

print(response.choices[0].message.content)
```

---

## 5-6 讓 Claude Code 使用本機 Ollama 模型

### 5-6-1 手動設定環境變數

```bash
# 設定環境變數
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 啟動 Claude Code
claude
```

### 5-6-2 寫入設定檔（永久生效）

```bash
# 編輯 Zsh 設定檔
nano ~/.zshrc

# 加入以下內容
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 重新載入
source ~/.zshrc
```

### 5-6-3 切換回雲端模型

如果你需要切換回雲端版 Claude：

```bash
# 清除環境變數
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_API_KEY

# 重新啟動 Claude Code
claude
```

---

## 5-7 Ollama 進階設定

### 5-7-1 自訂推論參數

建立 `Modelfile` 來自訂模型行為：

```dockerfile
FROM qwen3-8b

# 設定系統提示詞
SYSTEM """你是一個專業的程式設計助手。
請用繁體中文回答問題。
程式碼請加入型別提示和文件字串。"""

# 設定預設參數
PARAMETER temperature 0.3
PARAMETER num_ctx 8192
PARAMETER top_p 0.9
```

建立自訂模型：

```bash
ollama create my-coder -f Modelfile
ollama run my-coder
```

### 5-7-2 記憶體用量控制

```bash
# 限制 Ollama 使用的 GPU 記憶體
sudo systemctl edit ollama
```

```ini
[Service]
Environment="OLLAMA_MAX_VRAM=64000000000"
```

這會限制 Ollama 最多使用 64GB GPU 記憶體。

### 5-7-3 管理模型

```bash
# 列出所有模型
ollama list

# 查看模型資訊
ollama show qwen3-8b

# 刪除模型
ollama rm qwen3-8b

# 複製模型（備份）
ollama cp qwen3-8b qwen3-8b-backup

# 匯出模型
ollama cp qwen3-8b /backup/qwen3-8b.gguf
```

---

## 5-8 常見問題與疑難排解

### 5-8-1 模型下載失敗

```bash
# 檢查網路連線
ping ollama.com

# 清除快取重試
rm -rf ~/.ollama/models
ollama pull qwen3-8b
```

### 5-8-2 Ollama 服務無法啟動

```bash
# 查看日誌
journalctl -u ollama -n 50

# 常見原因：
# 1. Port 11434 被佔用
# 2. GPU 驅動問題
# 3. 記憶體不足
```

### 5-8-3 推論速度太慢

```bash
# 確認 GPU 有被使用
nvidia-smi

# 如果 GPU 使用率很低，可能是模型沒完全載入到 GPU
# 嘗試減少上下文長度
ollama run qwen3-8b --num-ctx 4096
```

---

## 5-9 本章小結

::: success ✅ 你現在知道了
- Ollama 是最簡單的本地 LLM 推論工具，一行指令就能安裝
- 在 DGX Spark 上可以跑 120B 等級的超大模型
- Qwen3.5 122B 支援文字和影像的多模態輸入
- Ollama 提供 OpenAI 相容 API，任何相容的軟體都能直接使用
- Claude Code 可以設定使用本機 Ollama 模型
- Modelfile 可以自訂模型行為和參數
:::

::: tip 🚀 下一章預告
Ollama 讓我們能在命令列中跟 AI 對話。但如果你想要一個漂亮的網頁介面，可以管理多個模型、上傳文件做 RAG、甚至讓 AI 使用工具呢？下一章的 Open WebUI 就是答案！

👉 [前往第 6 章：Open WebUI — 瀏覽器裡的 AI 助手 →](/guide/chapter6/)
:::

::: info 📝 上一章
← [回到第 4 章：遠端桌面與網路存取](/guide/chapter4/)
:::
