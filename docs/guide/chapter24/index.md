# 第 24 章：開發環境與 AI 輔助程式開發

::: tip 🎯 本章你將學到什麼
- VS Code 遠端開發與 Ollama 整合
- Continue 擴充設定
- Vibe Coding：用 Claude Code 搭配本機 Ollama 模型
- 用自然語言開發完整專案
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 24-1 VS Code 遠端開發與 Ollama 整合

### 24-1-1 為什麼用 VS Code 遠端開發？

::: info 🤔 遠端開發是什麼？
VS Code Remote SSH 讓你的筆電變成 DGX Spark 的「遙控器」：
- 程式碼存在 DGX Spark 上
- 編輯、執行、除錯都在 DGX Spark 上
- 你的筆電只負責顯示畫面
- 享受 DGX Spark 的 128GB 記憶體和 GPU 算力
:::

### 24-1-2 透過 NVIDIA Sync 連入 DGX Spark

如果你已經設定了 NVIDIA Sync（第 2 章），在 Sync 中找到 DGX Spark，點擊 **Open in VS Code** 即可。

### 24-1-3 VS Code 遠端連線（手動方式）

1. 在 VS Code 中安裝 **Remote - SSH** 擴充（搜尋 `ms-vscode-remote.remote-ssh`）
2. 按 `Ctrl+Shift+P`（macOS: `Cmd+Shift+P`）
3. 輸入 `Remote-SSH: Connect to Host`
4. 輸入 `使用者名稱@DGX_Spark_IP`
5. 選擇 Linux 平台
6. 輸入密碼（如果設定了 SSH Key 就不需要）

連線成功後，左下角會顯示 `SSH: DGX_Spark_IP`。

### 24-1-4 在 VS Code 中接上 Ollama 模型

安裝 **Continue** 擴充：

1. 按 `Ctrl+Shift+X` 打開擴充市集
2. 搜尋 `Continue`
3. 點擊 **Install**

Continue 是一個開源的 AI 程式設計助手，支援 Ollama。

### 24-1-5 設定 Ollama 模型提供者

安裝完成後，Continue 會自動開啟設定檔 `~/.continue/config.json`：

```json
{
  "models": [
    {
      "title": "Ollama - Qwen3-8B",
      "provider": "ollama",
      "model": "qwen3-8b",
      "apiBase": "http://localhost:11434"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Ollama - Qwen3-Coder",
    "provider": "ollama",
    "model": "qwen3-coder",
    "apiBase": "http://localhost:11434"
  }
}
```

**如果 Ollama 在 DGX Spark 上，而你的 VS Code 在本機**：

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
2. 輸入問題或指令
3. AI 會給出程式碼建議

**快捷鍵**：
- `Ctrl+L`（macOS: `Cmd+L`）：打開 Continue 對話面板
- `Ctrl+I`（macOS: `Cmd+I`）：快速編輯（選取程式碼後按）
- `Tab`：接受自動完成建議

### 24-1-7 AI 輔助程式開發實戰

**場景一：寫一個新函式**

```
你：「幫我寫一個 Python 函式，讀取 CSV 檔案並計算每個分類的平均值，
     用 pandas 實作，加入錯誤處理」

Continue：
```python
import pandas as pd
from typing import Optional

def calculate_category_averages(
    csv_path: str,
    category_col: str,
    value_col: str
) -> Optional[pd.DataFrame]:
    """讀取 CSV 並計算每個分類的平均值。

    Args:
        csv_path: CSV 檔案路徑
        category_col: 分類欄位名稱
        value_col: 數值欄位名稱

    Returns:
        每個分類的平均值 DataFrame，發生錯誤時回傳 None
    """
    try:
        df = pd.read_csv(csv_path)

        # 驗證欄位存在
        if category_col not in df.columns:
            raise ValueError(f"找不到分類欄位: {category_col}")
        if value_col not in df.columns:
            raise ValueError(f"找不到數值欄位: {value_col}")

        # 計算平均值
        result = df.groupby(category_col)[value_col].agg(['mean', 'count', 'std'])
        result = result.rename(columns={'mean': '平均值', 'count': '數量', 'std': '標準差'})

        return result

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {csv_path}")
        return None
    except Exception as e:
        print(f"錯誤：{e}")
        return None
```
```

**場景二：除錯**

```
你（選取有 bug 的程式碼，按 Ctrl+I）：
「這段程式碼為什麼會報錯？」

Continue：
「問題在第 15 行：你在迴圈中修改了正在迭代的字典。
  建議改用 list(d.items()) 來建立副本：

  for key, value in list(d.items()):
      ...」
```

**場景三：重構**

```
你（選取程式碼，按 Ctrl+I）：
「把這段程式碼重構，加入型別提示和文件字串」

Continue：
「好的，這是重構後的版本：
  ...（給出改進後的程式碼）」
```

### 24-1-8 自動完成（Tab Autocomplete）

Continue 支援類似 GitHub Copilot 的自動完成：

1. 在設定中啟用 `tabAutocompleteModel`
2. 寫程式時，Continue 會自動給出建議
3. 按 `Tab` 接受，按 `Esc` 忽略

**推薦的自動完成模型**：

| 模型 | 記憶體 | 準確率 | 速度 |
|------|--------|--------|------|
| **Qwen3-Coder** | ~8 GB | 高 | 快 |
| DeepSeek-Coder | ~6 GB | 高 | 快 |
| StarCoder2 15B | ~10 GB | 中高 | 中等 |

### 24-1-9 疑難排解

**Q：Continue 無法連線到 Ollama？**

```bash
# 確認 Ollama 正在執行
systemctl status ollama

# 確認 Ollama 監聽 0.0.0.0
curl http://DGX_Spark_IP:11434/api/tags

# 如果只有 localhost，修改設定
sudo systemctl edit ollama
# 加入：Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

**Q：自動完成沒有出現？**

1. 確認 `tabAutocompleteModel` 已設定
2. 確認模型已下載：`ollama list`
3. 查看 Continue 日誌：`Ctrl+Shift+P` → `Continue: Open Logs`

---

## 24-2 Vibe Coding：用 Claude Code 搭配本機 Ollama 模型

### 24-2-1 什麼是 Vibe Coding？

::: info 🤔 Vibe Coding 是什麼？
Vibe Coding 是一個流行詞，意思是「用感覺（vibe）寫程式」— 你不需要懂程式碼細節，只需要用自然語言描述你想做什麼，AI 就會幫你寫出來。

在 DGX Spark 上，Vibe Coding 的組合是：
- **Claude Code**：理解你的需求、規劃架構、寫程式碼
- **Ollama 本機模型**：提供 AI 能力，不需要網路連線
:::

### 24-2-2 架構概覽

```
你（自然語言描述需求）
  → Claude Code（理解需求、規劃架構）
    → Ollama 本機模型（提供 AI 推理）
      → Claude Code（整合、測試、除錯）
        → 完成的程式
```

### 24-2-3 安裝 Ollama 與下載程式開發模型

```bash
# 下載適合程式開發的模型
ollama pull qwen3-8b-coder     # 程式碼生成
ollama pull deepseek-coder      # 另一個好的程式碼模型
ollama pull qwen3-8b            # 一般對話
```

### 24-2-4 安裝 Claude Code

第 3 章已安裝。確認可以正常執行：

```bash
claude --version
```

### 24-2-5 一行指令啟動 Claude Code

```bash
claude
```

### 24-2-6 進階：手動設定環境變數

如果你想讓 Claude Code 永遠使用本機 Ollama：

```bash
# 編輯 Zsh 設定檔
nano ~/.zshrc

# 加入以下內容
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 重新載入
source ~/.zshrc
```

### 24-2-7 Vibe Coding 實戰：用自然語言開發

**專案一：建立一個 Flask 網站**

```
你：「幫我建立一個 Flask 網站，需求如下：
     1. 有一個首頁，顯示歡迎訊息
     2. 有一個 /api/data 端點，回傳 JSON 格式的資料
     3. 用 SQLite 儲存資料
     4. 建立 Dockerfile
     5. 寫一個 README.md 說明如何使用」

Claude Code 會：
  1. 建立專案目錄結構
  2. 寫 app.py（Flask 主程式）
  3. 寫 models.py（資料庫模型）
  4. 寫 requirements.txt
  5. 寫 Dockerfile
  6. 寫 README.md
  7. 安裝相依套件
  8. 啟動服務
  9. 測試 API 端點
```

**專案二：資料分析腳本**

```
你：「幫我寫一個 Python 腳本，分析 /data/sales.csv：
     1. 計算每個月的銷售總額
     2. 找出銷售最好的前 10 個產品
     3. 畫出月度趨勢圖
     4. 匯出結果到 Excel」

Claude Code 會：
  1. 讀取 CSV 檔案結構
  2. 寫分析腳本
  3. 執行並驗證結果
  4. 生成圖表
  5. 匯出 Excel
```

**專案三：自動化腳本**

```
你：「幫我寫一個 bash 腳本，每天凌晨 2 點自動執行：
     1. 備份 /data 目錄到 /backup
     2. 壓縮超過 7 天的備份
     3. 刪除超過 30 天的備份
     4. 寄送報告 email 到 admin@example.com」

Claude Code 會：
  1. 寫 bash 腳本
  2. 設定 crontab
  3. 測試腳本
  4. 確認排程設定
```

### 24-2-8 Vibe Coding 的最佳實踐

| 做法 | 說明 |
|------|------|
| ✅ **給明確的需求** | 越詳細，AI 寫的程式碼越符合你的期望 |
| ✅ **分步驟進行** | 大專案拆成小步驟，每一步確認後再繼續 |
| ✅ **審查程式碼** | AI 寫的程式碼還是要看一下，確保邏輯正確 |
| ✅ **提供上下文** | 告訴 AI 你的環境、框架、程式語言版本 |
| ❌ **不要一次給太多** | 一次描述太多需求，AI 容易遺漏或出錯 |
| ❌ **不要完全信任** | AI 也會出錯，一定要測試 |

---

## 24-3 本章小結

::: success ✅ 你現在知道了
- VS Code Remote SSH 讓你的筆電變成 DGX Spark 的遙控器
- Continue 擴充讓 VS Code 整合 Ollama 模型
- 自動完成功能類似 GitHub Copilot，但完全免費、本地執行
- Vibe Coding 讓你用自然語言就能開發
- Claude Code + Ollama 是最強大的本地 AI 開發組合
:::

::: tip 🚀 下一章預告
最後一章！來看看怎麼把多台 DGX Spark 連在一起，打造你的個人 AI 叢集！

👉 [前往第 25 章：多機互連與分散式運算 →](/guide/chapter25/)
:::

::: info 📝 上一章
← [回到第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用](/guide/chapter23/)
:::
