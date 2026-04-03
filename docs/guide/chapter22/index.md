# 第 22 章：AI Agent 與安全沙箱

::: tip 🎯 本章你將學到什麼
- AI Agent 概觀
- OpenClaw 一鍵部署本地 AI 代理
- NemoClaw 企業級 Agent
- OpenShell 安全沙箱
:::

---

## 22-1 AI Agent 概觀

::: info 🤔 什麼是 AI Agent？
AI Agent 不只是「回答問題」，它還能「做事」：
- 執行命令
- 讀寫檔案
- 瀏覽網頁
- 呼叫 API
- 與其他 Agent 協作

簡單說：Chatbot 是顧問，Agent 是員工。
:::

---

## 22-2 OpenClaw — 用 Ollama 一鍵部署本地 AI 代理

### 22-2-1 啟動安裝

```bash
docker run -d \
  --name openclaw \
  --network host \
  -v ~/openclaw-data:/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  ghcr.io/community/openclaw:latest
```

### 22-2-2 選擇模型

OpenClaw 支援從 Ollama 選擇模型：
- Qwen3-8B：快速回應
- Nemotron-3-Nano：高品質

### 22-2-3 安全確認與啟動

OpenClaw 會列出它能執行的操作，要求你確認。

### 22-2-4 初次對話與設定

打開 `http://DGX_Spark_IP:3000`，開始跟 Agent 對話。

### 22-2-5 實際使用

```
你：「幫我整理 /data 目錄中所有 .txt 檔案的摘要」
Agent：
  1. 列出所有 .txt 檔案
  2. 逐一閱讀
  3. 生成摘要報告
```

---

## 22-3 NemoClaw — Nemotron 驅動的企業級 Agent

### 22-3-1 前置準備

```bash
# 確保 Ollama 正在執行
systemctl status ollama
```

### 22-3-2 安裝 NemoClaw

```bash
docker run -d \
  --name nemoclaw \
  --network host \
  -v ~/nemoclaw-data:/data \
  ghcr.io/community/nemoclaw:latest
```

### 22-3-3 執行設定精靈

```bash
docker exec nemoclaw python setup.py
```

### 22-3-4 設定本地推論

在設定中指定 Ollama 端點：
```
OLLAMA_BASE_URL: http://localhost:11434
MODEL: nemotron-super:120b
```

### 22-3-5 打開 Dashboard

`http://DGX_Spark_IP:8081`

### 22-3-6 NemoClaw 常見問題

**Q：Agent 無法連線到 Ollama？**

```bash
# 確認 Ollama 監聽 0.0.0.0
curl http://localhost:11434/api/tags
```

---

## 22-4 OpenShell — AI Agent 安全沙箱

### 22-4-1 三大核心元件

1. **Policy Engine**：定義 Agent 能做什麼
2. **Privacy Router**：過濾敏感資訊
3. **Monitor**：即時監控 Agent 行為

### 22-4-2 YAML 宣告式政策

```yaml
# policy.yaml
allow:
  - read_files: ["/data/*", "/home/*"]
  - write_files: ["/output/*"]
  - run_commands: ["ls", "cat", "grep"]
deny:
  - delete_files: true
  - network_access: true
  - sudo: true
```

### 22-4-3 隱私路由器

```yaml
privacy:
  mask_patterns:
    - type: "email"
      pattern: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+"
    - type: "phone"
      pattern: "\\d{2,4}-\\d{6,8}"
```

### 22-4-4 即時監控

```bash
# 啟動監控
openshell monitor --config policy.yaml

# 查看日誌
openshell logs --follow
```

---

## 22-5 本章小結

::: success ✅ 你現在知道了
- AI Agent 不只是聊天，還能執行任務
- OpenClaw 適合個人使用
- NemoClaw 適合企業環境
- OpenShell 確保 Agent 行為安全
:::

::: tip 🚀 第六篇完結！
恭喜！你已經完成了「多模態 AI 與智慧代理」篇。

最後一篇要介紹科學計算、開發工具和多機互連！

👉 [前往第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用 →](/guide/chapter23/)
:::

::: info 📝 上一章
← [回到第 21 章：RAG 與知識圖譜](/guide/chapter21/)
:::
