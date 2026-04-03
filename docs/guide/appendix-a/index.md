# 附錄 A：Claude Code 常用指令速查表

::: tip 📋 快速參考
本章整理了 Claude Code 的常用操作指令，方便快速查找。
:::

---

## 安裝與啟動

```bash
# 全域安裝
npm install -g @anthropic-ai/claude-code

# 確認版本
claude --version

# 啟動互動模式
claude

# 用 npx（免安裝）
npx @anthropic-ai/claude-code

# 更新
npm update -g @anthropic-ai/claude-code
```

## 互動模式操作

### 基本對話

```
# 描述你想做的事
> 幫我安裝 Ollama

# 指定任務細節
> 用 Docker 部署 Open WebUI，連線到本機 Ollama，設定自動重啟

# 除錯
> 這個錯誤是什麼意思？幫我修復：
> （貼上錯誤訊息）

# 檔案操作
> 讀取 config.yaml 並告訴我內容
> 幫我修改 main.py 第 42 行，把 port 從 8080 改為 8081
```

### 專案操作

```
# 掃描專案結構
> 掃描這個目錄，告訴我有哪些檔案和資料夾

# 程式碼審查
> 審查 app/main.py 的程式碼，找出潛在問題

# 重構
> 把 utils.py 重構，加入型別提示和文件字串

# 建立新檔案
> 幫我建立一個 .gitignore 檔案，排除 node_modules 和 .env
```

## 搭配 Ollama 使用（第 24 章）

```bash
# 設定環境變數（使用本機 Ollama）
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 寫入設定檔（永久生效）
echo 'export ANTHROPIC_BASE_URL="http://localhost:11434/v1"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="ollama"' >> ~/.zshrc
source ~/.zshrc
```

## 常見用法範例

| 任務 | 指令範例 |
|------|---------|
| 安裝軟體 | `幫我安裝 <軟體名>` |
| 部署服務 | `用 Docker 部署 <服務名>，設定自動重啟` |
| 除錯 | `幫我看看這個錯誤：<貼上錯誤訊息>` |
| 寫程式 | `幫我寫一個 <功能描述> 的 Python 程式` |
| 修改設定 | `把 <檔案> 中的 <舊值> 改為 <新值>` |
| 查詢資訊 | `告訴我目前 GPU 記憶體使用狀況` |
| 建立專案 | `幫我建立一個 Flask 專案，有首頁和 API 端點` |
| 分析資料 | `分析 /data/sales.csv，計算每月銷售總額` |

## CLAUDE.md 設定檔

在專案根目錄建立 `CLAUDE.md`，告訴 Claude Code 你的環境和偏好：

```markdown
# CLAUDE.md

## 環境
- 系統：DGX OS (Ubuntu 24.04 ARM64)
- GPU：NVIDIA Blackwell, 128GB 統一記憶體
- Docker：已安裝
- Python：使用 uv 管理

## 偏好
- 優先使用 Docker 部署服務
- Python 環境使用 uv 而非 pip
- 所有指令都要解釋在做什麼
- 程式碼加入型別提示和文件字串
- 使用繁體中文溝通

## 安全限制
- 不要刪除 /data 目錄中的檔案
- 不要執行 sudo 指令
- 不要修改系統設定檔
```

## 進階技巧

### 1. 管道模式（非互動）

```bash
# 把指令透過管道傳給 Claude Code
echo "幫我列出目前目錄中的所有 Python 檔案" | claude

# 從檔案讀取指令
claude < prompt.txt
```

### 2. 多步驟任務

```
> 幫我完成以下任務：
> 1. 在當前目錄建立 Flask 專案
> 2. 安裝 Flask 和相關套件
> 3. 建立一個簡單的 API 端點
> 4. 寫測試
> 5. 建立 Dockerfile
> 每完成一步就告訴我進度
```

### 3. 指定輸出格式

```
> 幫我分析這個 CSV 檔案，用表格格式輸出結果：
> - 總筆數
> - 每個分類的數量
> - 平均值、最大值、最小值
```

---

::: info 📝 返回
← [回到第 25 章：多機互連](/guide/chapter25/) | [首頁](/)
:::
