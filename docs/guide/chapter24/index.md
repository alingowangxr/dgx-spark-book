# 第 24 章：開發環境與 AI 輔助程式開發

::: tip 🎯 本章你將學到什麼
- VS Code 遠端開發與 Ollama 整合
- Vibe Coding：用 Claude Code 搭配本機 Ollama 模型
:::

---

## 24-1 VS Code 遠端開發與 Ollama 整合

### 24-1-1 透過 NVIDIA Sync 連入 DGX Spark

在 NVIDIA Sync 中找到 DGX Spark，點擊 Connect。

### 24-1-2 VS Code 遠端連線

1. 在 VS Code 中安裝 **Remote - SSH** 擴充
2. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`
3. 輸入 `使用者名稱@DGX_Spark_IP`

### 24-1-3 在 VS Code 中接上 Ollama 模型

安裝 **Continue** 擴充：
1. VS Code Extensions → 搜尋 `Continue`
2. 安裝

### 24-1-4 新增 Ollama 模型提供者

在 Continue 的設定中：

```json
{
  "models": [
    {
      "title": "Ollama - Qwen3-8B",
      "provider": "ollama",
      "model": "qwen3-8b",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

### 24-1-5 設定 Ollama 連線端點

如果 Ollama 在 DGX Spark 上，而你的 VS Code 在本機：

```json
{
  "models": [
    {
      "title": "DGX Spark Ollama",
      "provider": "ollama",
      "model": "qwen3-8b",
      "apiBase": "http://DGX_Spark_IP:11434"
    }
  ]
}
```

### 24-1-6 選擇模型開始寫程式

在 VS Code 右側的 Continue 面板中：
1. 選擇模型
2. 輸入問題
3. AI 會給出程式碼建議

### 24-1-7 AI 輔助程式開發實戰

```
你：「幫我寫一個 Python 函式，讀取 CSV 檔案並計算每個分類的平均值」

AI：
```python
import pandas as pd

def calculate_category_averages(csv_path, category_col, value_col):
    df = pd.read_csv(csv_path)
    return df.groupby(category_col)[value_col].mean()
```
```

### 24-1-8 疑難排解

**Q：Continue 無法連線到 Ollama？**

```bash
# 確認 Ollama 監聽 0.0.0.0
curl http://DGX_Spark_IP:11434/api/tags
```

---

## 24-2 Vibe Coding：用 Claude Code 搭配本機 Ollama 模型

### 24-2-1 架構概覽

```
你（自然語言描述需求）
  → Claude Code（理解需求、規劃架構）
    → Ollama 本機模型（提供程式碼建議）
      → Claude Code（整合、測試、除錯）
        → 完成的程式
```

### 24-2-2 安裝 Ollama 與下載程式開發模型

```bash
# 下載適合程式開發的模型
ollama pull qwen3-8b-coder
ollama pull deepseek-coder
```

### 24-2-3 安裝 Claude Code

第 3 章已安裝。

### 24-2-4 一行指令啟動 Claude Code

```bash
claude
```

然後告訴它你想做什麼。

### 24-2-5 進階：手動設定環境變數

```bash
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"
```

### 24-2-6 Vibe Coding 實戰：用自然語言開發

```
你：「幫我建立一個 Flask 網站，有一個首頁、一個 API 端點，
     用 SQLite 儲存資料，部署到 DGX Spark 上。」

Claude Code 會：
1. 建立專案結構
2. 寫 Flask 程式碼
3. 設定資料庫
4. 建立 Dockerfile
5. 啟動服務
6. 測試 API
```

---

## 24-3 本章小結

::: success ✅ 你現在知道了
- VS Code + Continue + Ollama 是強大的遠端開發組合
- Vibe Coding 讓你用自然語言就能開發
- Claude Code 可以自動完成整個專案的建立
:::

::: tip 🚀 下一章預告
最後一章！來看看怎麼把多台 DGX Spark 連在一起，打造你的個人 AI 叢集！

👉 [前往第 25 章：多機互連與分散式運算 →](/guide/chapter25/)
:::

::: info 📝 上一章
← [回到第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用](/guide/chapter23/)
:::
