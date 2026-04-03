# 第 6 章：Open WebUI — 瀏覽器裡的 AI 助手

::: tip 🎯 本章你將學到什麼
- 用 Claude Code 一鍵部署 Open WebUI
- 管理多個模型、切換對話
- 使用 RAG 讓 AI 讀你的文件
- 啟用 Tool Calling 讓 AI 使用工具和執行程式碼
- 自訂主題和外掛
- 日常維護與更新
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。部署約需 5 分鐘。
:::

---

## 6-1 用 Claude Code 部署 Open WebUI

### 6-1-1 什麼是 Open WebUI？

::: info 🤔 為什麼需要 Open WebUI？
Ollama 的命令列介面很好用，但如果你想要：
- 類似 ChatGPT 的漂亮網頁介面
- 管理多個模型，一鍵切換
- 上傳文件讓 AI 閱讀（RAG）
- 讓 AI 使用工具（搜尋、程式碼執行）
- 多人共用（建立不同帳號）

Open WebUI 就是答案。它是目前最流行的開源 LLM 網頁介面，支援 Ollama、OpenAI API、以及任何 OpenAI 相容的端點。
:::

### 6-1-2 確認 Docker 權限

```bash
# 確認 Docker 正常運作
docker ps

# 如果出現權限錯誤
sudo usermod -aG docker $USER
newgrp docker
```

### 6-1-3 用 Claude Code 拉取映像檔並啟動

告訴 Claude Code：

> 「用 Docker 部署 Open WebUI，連線到我本機的 Ollama，設定自動重啟。」

Claude Code 會執行：

```bash
docker run -d \
  --name open-webui \
  --network host \
  -v open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

**參數解釋**：

| 參數 | 說明 |
|------|------|
| `--network host` | 使用主機網路（直接存取 Ollama） |
| `-v open-webui:/app/backend/data` | 持久化對話紀錄和設定 |
| `-e OLLAMA_BASE_URL` | 告訴 Open WebUI 去哪找 Ollama |
| `--restart unless-stopped` | 開機自動啟動 |

### 6-1-4 首次登入與建立帳號

用瀏覽器打開：

```
http://DGX_Spark_IP:8080
```

第一次開啟會要求你建立管理員帳號：
1. 輸入姓名
2. 輸入 Email
3. 設定密碼

::: warning ⚠️ 安全性
如果 DGX Spark 對外開放了（例如用 Tailscale），建議：
- 設定強密碼
- 關閉公開註冊（在管理設定中）
:::

---

## 6-2 模型管理與基本對話

### 6-2-1 認識主畫面

登入後，你會看到：

- **左側欄**：對話歷史紀錄（按時間排序）
- **中間**：對話視窗（類似 ChatGPT 介面）
- **上方**：模型選擇下拉
- **下方**：輸入框 + 附件按鈕 + 工具按鈕

### 6-2-2 選擇與下載模型

點擊上方的模型選擇器，你會看到 Ollama 中已下載的模型。

如果模型還沒下載，點擊模型名稱旁邊的 **Download** 按鈕，Open WebUI 會自動呼叫 Ollama 下載。

::: tip 💡 模型下載進度
下載大模型時，你可以在左側欄看到下載進度。120B 模型可能需要 30-60 分鐘。
:::

### 6-2-3 管理員控制台與模型管理

點擊左下角的 **設定圖示** → **Admin Panel**：

| 選項 | 功能 |
|------|------|
| **Models** | 管理模型（顯示、隱藏、排序、編輯系統提示詞） |
| **Users** | 管理使用者（建立、刪除、設定角色） |
| **Settings** | 系統設定（介面、RAG、工具、連線） |
| **Documents** | 上傳文件（RAG 知識庫） |

### 6-2-4 基本對話與模型切換

在輸入框中輸入問題，按 Enter 送出。

要切換模型，點擊上方的模型名稱，選擇另一個模型即可。

::: tip 💡 模型切換不需要重新載入
Open WebUI 會自動處理模型切換，你不需要重新整理頁面。新的對話會使用新模型，舊的對話紀錄保留原模型。
:::

### 6-2-5 Arena 模式：模型對比

Open WebUI 的 Arena 模式讓你同時用兩個模型回答同一個問題，方便比較品質。

1. 點擊模型選擇器
2. 選擇「Arena」
3. 選擇兩個要比較的模型
4. 輸入問題，兩個模型會同時回答
5. 投票選出較好的回答

---

## 6-3 RAG：讓模型讀你的文件

### 6-3-1 在對話中上傳文件

最簡單的方式：

1. 在對話視窗中，點擊輸入框旁的 📎 圖示
2. 選擇檔案（支援 PDF、TXT、MD、DOCX、CSV、HTML）
3. 檔案上傳後，系統自動建立向量索引
4. 在對話中提問，AI 會參考文件內容回答

### 6-3-2 建立知識庫

如果要建立可重複使用的知識庫：

1. 進入 **Admin Panel → Documents**
2. 點擊 **Upload**
3. 上傳多個文件
4. 設定知識庫名稱和描述
5. 在對話中選擇要使用的知識庫

### 6-3-3 RAG 設定調校

在 **Admin Panel → Settings → RAG** 中可以調整：

| 設定 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| **Top K** | 檢索最相關的 K 個片段 | 3-5 | 太少可能遺漏，太多會干擾 |
| **Chunk Size** | 每個片段的大小 | 500-1000 | 太小缺乏上下文 |
| **Chunk Overlap** | 片段重疊比例 | 10-20% | 避免資訊被切斷 |
| **Embedding Model** | 嵌入模型 | BGE-M3（中文） | 影響檢索準確率 |

---

## 6-4 Tool Calling：讓模型使用工具

### 6-4-1 啟用函式呼叫

Open WebUI 支援讓 AI 呼叫外部工具。

在 **Admin Panel → Tools** 中可以啟用：
- **Web Search**：網路搜尋
- **Code Interpreter**：程式碼執行
- **Image Generation**：圖片生成
- **File Browser**：檔案瀏覽

### 6-4-2 程式碼執行（Code Interpreter）

啟用 Code Interpreter 後，AI 可以：
- 寫 Python 程式並執行
- 分析資料
- 產生圖表
- 處理檔案

**啟用方式**：
1. **Admin Panel → Tools**
2. 找到 **Code Interpreter**
3. 點擊啟用

::: warning ⚠️ 安全提醒
Code Interpreter 會在伺服器上執行程式碼。如果多人使用，建議：
- 限制可存取的目錄
- 設定資源上限
- 定期審計執行的程式碼
:::

### 6-4-3 網路搜尋

啟用網路搜尋後，AI 可以即時搜尋網路資訊。

**設定方式**：
1. **Admin Panel → Tools → Web Search**
2. 選擇搜尋引擎：
   - **SearXNG**（開源，推薦）
   - **Google Custom Search**
   - **DuckDuckGo**

**SearXNG 部署**：

```bash
docker run -d \
  --name searxng \
  --network host \
  -v ~/searxng:/etc/searxng \
  searxng/searxng:latest
```

然後在 Open WebUI 中設定 SearXNG 的 URL：`http://localhost:8080`

---

## 6-5 自訂主題和外掛

### 6-5-1 更換主題

在 **Settings → Interface** 中可以：
- 切換亮色/暗色模式
- 選擇配色方案
- 調整字型大小

### 6-5-2 自訂系統提示詞

在 **Admin Panel → Models** 中，點擊任何模型的編輯按鈕，可以設定：

```
System Prompt:
你是一個專業的 AI 助手，擅長回答技術問題。
請用繁體中文回答。
程式碼請加入型別提示和文件字串。
```

### 6-5-3 安裝外掛

Open WebUI 支援外掛系統：

1. **Admin Panel → Settings → Functions**
2. 點擊 **Import** 匯入外掛
3. 外掛可以是 Python 腳本，定義自訂工具

---

## 6-6 更新與維護

### 6-6-1 更新容器

```bash
# 停止現有容器
docker stop open-webui
docker rm open-webui

# 拉取最新版本
docker pull ghcr.io/open-webui/open-webui:main

# 重新啟動（用同樣的指令）
docker run -d \
  --name open-webui \
  --network host \
  -v open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

::: tip 💡 用 Watchtower 自動更新
```bash
docker run -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --interval 86400 \
  open-webui
```
Watchtower 會每天自動檢查並更新 Open WebUI。
:::

### 6-6-2 完整移除

```bash
# 停止並移除容器
docker stop open-webui
docker rm open-webui

# 移除映像檔
docker rmi ghcr.io/open-webui/open-webui:main

# 移除資料卷（會刪除所有對話紀錄和設定！）
docker volume rm open-webui
```

### 6-6-3 記憶體管理

Open WebUI 本身佔用記憶體不多（約 200-500MB），但如果同時跑多個模型，記憶體用量會增加。

```bash
# 查看 Open WebUI 的記憶體使用量
docker stats open-webui --no-stream

# 查看 Ollama 的記憶體使用量
docker stats ollama --no-stream
```

---

## 6-7 常見問題與疑難排解

### 6-7-1 無法連線到 Ollama

```bash
# 確認 Ollama 正在執行
systemctl status ollama

# 確認 Ollama 監聽 0.0.0.0
curl http://localhost:11434/api/tags

# 如果只有 localhost，修改 Ollama 設定
sudo systemctl edit ollama
# 加入：Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

### 6-7-2 對話紀錄消失

```bash
# 確認 Docker volume 存在
docker volume ls | grep open-webui

# 如果 volume 不存在，可能是用錯了啟動指令
# 重新啟動時確保使用相同的 -v 參數
```

### 6-7-3 RAG 檢索不準確

1. 換用更好的 Embedding 模型（BGE-M3）
2. 增加 Top K（3 → 5）
3. 增加 Chunk Size（500 → 1000）
4. 檢查文件品質

---

## 6-8 本章小結

::: success ✅ 你現在知道了
- Open WebUI 是一個漂亮的網頁版 AI 介面，類似 ChatGPT
- 支援多模型管理、RAG 文件檢索、Tool Calling
- Arena 模式可以比較不同模型的表現
- RAG 讓 AI 能讀你的文件，回答有依據
- Code Interpreter 讓 AI 能寫程式並執行
- Watchtower 可以自動更新容器
:::

::: tip 🚀 下一章預告
如果你想要一個更輕量、不需要 Docker 的模型服務呢？LM Studio 提供了 GUI 和 CLI 兩種操作方式，還能跨網路分享你的模型！

👉 [前往第 7 章：LM Studio — Headless 模型服務 →](/guide/chapter7/)
:::

::: info 📝 上一章
← [回到第 5 章：Ollama](/guide/chapter5/)
:::
