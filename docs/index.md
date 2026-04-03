---
layout: home
hero:
  name: "DGX Spark 玩透指南"
  text: "個人 AI 超級電腦全面實戰"
  tagline: 從開箱到 AI Agent，一步一步打造你的個人 AI 工作站
  image:
    src: /logo.svg
    alt: DGX Spark
  actions:
    - theme: brand
      text: 🚀 開始閱讀
      link: /guide/chapter1/
    - theme: alt
      text: 📋 模型清單
      link: /guide/models
    - theme: alt
      text: ❓ 常見問題
      link: /guide/faq

features:
  - icon: 🖥️
    title: 硬體與系統建置
    details: 從開箱到系統設定，一步步帶你認識 DGX Spark 的 128GB 統一記憶體與 GB10 超級晶片。
    link: /guide/chapter1/
  - icon: 🧠
    title: LLM 推論入門
    details: Ollama、Open WebUI、LM Studio、llama.cpp — 四種工具帶你跑起 120B 超大模型。
    link: /guide/chapter5/
  - icon: 🚀
    title: 推論進階加速
    details: vLLM、TensorRT-LLM、SGLang、NIM — 七大推論引擎全面比較與實戰。
    link: /guide/chapter9/
  - icon: 🎨
    title: 多媒體 AI 生成
    details: 圖片生成、影片生成、語音合成、音樂創作 — ComfyUI 完整工作流程。
    link: /guide/chapter13/
  - icon: 🔧
    title: 模型微調與訓練
    details: LoRA、QLoRA、Unsloth、Dreambooth、從零預訓練 — 六種 PEFT 方法比較。
    link: /guide/chapter15/
  - icon: 🤖
    title: 多模態 AI 與 Agent
    details: RAG 知識庫、知識圖譜、AI Agent 部署、安全沙箱 — 打造你的智慧系統。
    link: /guide/chapter20/
---

<div class="home-content">

## 為什麼選這本書？

| 特色 | 說明 |
|------|------|
| 🎯 **小白友善** | 每個指令都解釋為什麼要下，不假設任何先備知識 |
| 🖥️ **128GB 統一記憶體** | 體驗一般消費級 GPU 做不到的事：跑 120B 模型、微調 FLUX、預訓練小模型 |
| 🤖 **Claude Code 輔助** | 用 AI 寫 AI，自然語言就能完成複雜部署 |
| 📊 **實測數據** | 所有效能數據都是實際跑出來的，不是抄規格表 |
| 🔄 **持續更新** | 社群驅動，跟上最新的模型與工具 |

## 誰適合讀這本書？

- **完全新手**：沒碰過 Linux？沒關係，從第 1 章開始一步一步來
- **AI 愛好者**：想在本機跑大模型，但受限於顯卡記憶體
- **開發者**：想微調模型、部署 Agent、整合到自己的應用
- **研究人員**：需要預訓練領域模型、做 RAG、知識圖譜

## 快速開始

::: tip 💡 開始前準備
在打開 DGX Spark 之前，請先準備好：
- HDMI Dummy Plug（假負載插頭）
- 鍵盤、滑鼠、顯示器（首次設定用）
- 網路線（建議用 200GbE 高速網路）
- 另一台電腦（用於遠端連線）
:::

準備好了嗎？[點擊這裡開始第一章 →](/guide/chapter1/)

## 章節地圖

```
第一篇：硬體與系統建置    ← 你現在在這裡 📍
  ├─ 第 1 章：DGX Spark 硬體總覽
  ├─ 第 2 章：DGX OS 安裝與首次開機
  ├─ 第 3 章：Linux 環境與 Claude Code
  └─ 第 4 章：遠端桌面與網路存取

第二篇：LLM 推論入門
  ├─ 第 5 章：Ollama
  ├─ 第 6 章：Open WebUI
  ├─ 第 7 章：LM Studio
  └─ 第 8 章：llama.cpp

第三篇：LLM 推論進階
  ├─ 第 9 章：vLLM
  ├─ 第 10 章：TensorRT-LLM
  ├─ 第 11 章：SGLang
  └─ 第 12 章：NIM 與引擎比較

第四篇：多媒體 AI 生成
  ├─ 第 13 章：圖片與影片生成
  └─ 第 14 章：音訊、語音與音樂

第五篇：模型微調與訓練
  ├─ 第 15 章：LoRA / QLoRA 微調
  ├─ 第 16 章：Unsloth
  ├─ 第 17 章：LLaMA Factory / NeMo
  ├─ 第 18 章：影像模型微調
  └─ 第 19 章：預訓練中小型模型

第六篇：多模態 AI 與 Agent
  ├─ 第 20 章：多模態推論
  ├─ 第 21 章：RAG 與知識圖譜
  └─ 第 22 章：AI Agent 與沙箱

第七篇：科學計算與開發工具
  ├─ 第 23 章：CUDA-X 與 JAX
  ├─ 第 24 章：開發環境與 AI 輔助
  ├─ 第 25 章：多機互連
  └─ 第 26 章：綜合實戰：企業級 RAG 知識庫系統 ← 你現在在這裡 📍
```

</div>
