# 附錄 A：Claude Code 常用指令速查表

---

## 安裝與啟動

```bash
# 安裝
npm install -g @anthropic-ai/claude-code

# 啟動
claude

# 用 npx（免安裝）
npx @anthropic-ai/claude-code

# 確認版本
claude --version
```

## 互動模式操作

```
# 基本對話
> 幫我安裝 Ollama

# 指定任務
> 用 Docker 部署 Open WebUI，連線到本機 Ollama

# 除錯
> 這個錯誤是什麼意思？幫我修復

# 檔案操作
> 讀取 config.yaml 並告訴我內容

# 專案操作
> 掃描這個目錄，告訴我有哪些檔案
```

## 搭配 Ollama 使用

```bash
# 設定環境變數
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_API_KEY="ollama"

# 啟動
claude
```

## 常見用法範例

| 任務 | 指令 |
|------|------|
| 安裝軟體 | `幫我安裝 <軟體名>` |
| 部署服務 | `用 Docker 部署 <服務名>` |
| 除錯 | `幫我看看這個錯誤：<貼上錯誤訊息>` |
| 寫程式 | `幫我寫一個 <功能描述> 的 Python 程式` |
| 修改設定 | `把 <檔案> 中的 <舊值> 改為 <新值>` |
| 查詢資訊 | `告訴我目前 GPU 記憶體使用狀況` |

## CLAUDE.md 設定檔

在專案根目錄建立 `CLAUDE.md`：

```markdown
# CLAUDE.md

## 環境
- DGX OS (Ubuntu 24.04 ARM64)
- 128GB 統一記憶體
- Docker 已安裝

## 偏好
- 優先使用 Docker 部署
- Python 用 uv 管理
- 所有指令都要解釋
```

---

::: info 📝 返回
← [回到第 25 章：多機互連](/guide/chapter25/) | [首頁](/)
:::
