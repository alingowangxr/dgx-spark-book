# 第 5 章：Ollama — 在 128 GB 上跑超大模型

::: tip 🎯 本章你將學到什麼
- 用 Claude Code 一鍵安裝 Ollama
- 用 Docker 執行 Ollama（進階做法）
- 下載並執行 120B 等級的超大模型
- 測試多模態模型的影像分析能力
- 把 Ollama 開放成遠端 API 服務
- 讓 Claude Code 使用本機 Ollama 模型
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。下載 120B 模型可能需要 30-60 分鐘（視網路速度而定）。
:::

---

## 5-1 用 Claude Code 安裝 Ollama

### 5-1-1 安裝與驗證

::: info 🤔 什麼是 Ollama？
Ollama 是目前最流行的本地 LLM 推論工具。它的好處是：
- **超簡單**：一行指令就能下載並跑起模型
- **支援大模型**：在 DGX Spark 上能跑 120B 等級的模型
- **OpenAI 相容 API**：任何支援 OpenAI API 的軟體都能直接接 Ollama
- **免費開源**
:::

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

::: tip 💡 如果你想手動安裝
上面的 curl 指令就是 Ollama 的官方一鍵安裝腳本。它會：
1. 偵測你的系統架構（ARM64）
2. 下載對應的二進位檔
3. 建立 systemd 服務
4. 自動啟動
:::

安裝完成後，驗證：

```bash
# 確認 Ollama 版本
ollama --version

# 確認服務狀態
systemctl status ollama
```

### 5-1-2 確認服務狀態

```bash
# 查看 Ollama 是否在監聽
curl http://localhost:11434
# 應該會看到：Ollama is running

# 查看已下載的模型（目前應該是空的）
ollama list
```

### 5-1-3 設定 NVIDIA Sync 遠端存取

如果你用 NVIDIA Sync 管理 DGX Spark，可以在 Sync 中直接看到 Ollama 服務狀態。這在第 2 章已經設定過了。

---

## 5-2 用 Docker 執行 Ollama

### 5-2-1 用 Claude Code 啟動容器

除了直接安裝，你也可以用 Docker 執行 Ollama。這樣的好處是環境隔離、容易清理。

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
| `--gpus all` | 啟用 GPU 加速 |
| `-p 11434:11434` | 把容器的 11434 port 對應到主機 |
| `-v ollama_data:/root/.ollama` | 持久化模型資料 |
| `-v /models:/models` | 掛載本機模型目錄 |
| `--restart unless-stopped` | 開機自動啟動 |

### 5-2-2 掛載本機模型目錄

如果你已經用非 Docker 方式下載了模型，可以掛載進去給 Docker 版本的 Ollama 使用：

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

### 5-3-2 Nemotron-3-Super 120B

Nemotron-3-Super 是 NVIDIA 開源的 120B 參數模型，針對對話和指令遵循做了最佳化。

```bash
# 下載 Nemotron-3-Super 120B（NVFP4 量化版）
ollama run nemotron-super:120b
```

::: tip 💡 下載時間
120B 模型的 NVFP4 量化版大約 60GB，以一般家用網路速度（100Mbps）大約需要 1-2 小時。建議在睡前或出門前開始下載。
:::

下載完成後，你會在終端機中看到對話介面：

```
>>> 你好，請介紹一下你自己。
```

輸入任何問題來測試。

### 5-3-3 GPT-OSS 120B

GPT-OSS 是另一個開源的 120B 模型：

```bash
# 下載 GPT-OSS 120B
ollama run gpt-oss:120b
```

---

## 5-4 多模態模型實測：Qwen3.5 122B

### 5-4-1 文字對話

Qwen3.5 122B 是一個多模態模型，支援文字和影像輸入。

```bash
# 下載並執行
ollama run qwen3.5:122b
```

測試文字對話：

```
>>> 請用三句話解釋量子計算。
```

### 5-4-2 影像分析

Qwen3.5 122B 也支援影像分析。在 Ollama 中，你可以這樣做：

```bash
# 用影像檔案測試
ollama run qwen3.5:122b "這張圖片中有什麼？" --image /path/to/image.jpg
```

或者用 API 方式：

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

::: tip 💡 影像分析能力
在 DGX Spark 上，Qwen3.5 122B 的影像分析速度約為：
- 圖片載入：~2 秒
- 第一次回應：~5 秒
- 完整描述：~10 秒

這在 122B 模型中是非常快的。
:::

---

## 5-5 將 Ollama 開放成遠端服務

### 5-5-1 設定 OLLAMA_HOST

預設情況下，Ollama 只監聽 `localhost`。要讓其他裝置也能連線，需要修改設定。

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

### 5-5-2 Ollama API 端點一覽

Ollama 提供了完整的 REST API：

| 端點 | 方法 | 用途 |
|------|------|------|
| `/api/generate` | POST | 文字生成 |
| `/api/chat` | POST | 對話（支援多輪） |
| `/api/embeddings` | POST | 取得嵌入向量 |
| `/api/pull` | POST | 下載模型 |
| `/api/tags` | GET | 列出已下載的模型 |
| `/api/delete` | DELETE | 刪除模型 |

### 5-5-3 從個人電腦呼叫遠端 Ollama

在你的個人電腦上：

```bash
# 測試遠端 Ollama（把 IP 換成 DGX Spark 的 IP）
curl http://DGX_Spark_IP:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:122b",
    "prompt": "你好！",
    "stream": false
  }'
```

### 5-5-4 用 Python 呼叫遠端 Ollama

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

或者用 OpenAI 相容的端點：

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

### 5-6-1 用 ollama launch 快速啟動

如果你已經安裝了 Claude Code 的 `ollama launch` 功能：

```bash
claude ollama launch
```

這會自動設定 Claude Code 使用本機的 Ollama 模型作為後端。

### 5-6-2 手動設定遠端 Claude Code

```bash
# 設定環境變數
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 啟動 Claude Code
claude
```

### 5-6-3 寫入設定檔（永久生效）

```bash
# 編輯 Zsh 設定檔
nano ~/.zshrc

# 加入以下內容
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 重新載入
source ~/.zshrc
```

這樣每次啟動 Claude Code 時都會自動使用本機的 Ollama 模型。

::: tip 💡 為什麼要這樣做？
這樣你可以在沒有網路的情況下也能使用 AI 輔助，而且完全免費（不需要 API 額度）。

不過要注意，本機模型的能力可能不如雲端版 Claude，複雜任務可能需要切換回雲端模型。
:::

---

## 5-7 本章小結

::: success ✅ 你現在知道了
- Ollama 是最簡單的本地 LLM 推論工具，一行指令就能安裝
- 在 DGX Spark 上可以跑 120B 等級的超大模型
- Qwen3.5 122B 支援文字和影像的多模態輸入
- Ollama 提供 OpenAI 相容 API，任何相容的軟體都能直接使用
- Claude Code 可以設定使用本機 Ollama 模型
:::

::: tip 🚀 下一章預告
Ollama 讓我們能在命令列中跟 AI 對話。但如果你想要一個漂亮的網頁介面，可以管理多個模型、上傳文件做 RAG、甚至讓 AI 使用工具呢？下一章的 Open WebUI 就是答案！

👉 [前往第 6 章：Open WebUI — 瀏覽器裡的 AI 助手 →](/guide/chapter6/)
:::

::: info 📝 上一章
← [回到第 4 章：遠端桌面與網路存取](/guide/chapter4/)
:::
