# 🔥 DGX Spark 玩透指南 — 個人 AI 超級電腦全面實戰

> 從開箱到 AI Agent，一步一步打造你的個人 AI 工作站

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![VitePress](https://img.shields.io/badge/Built%20with-VitePress-6c5ce7)](https://vitepress.dev/)
[![GitHub Pages](https://img.shields.io/badge/Deployed%20on-GitHub%20Pages-22863a)](https://alingowangxr.github.io/dgx-spark-book/)
[![Stars](https://img.shields.io/github/stars/alingowangxr/dgx-spark-book?style=social)](https://github.com/alingowangxr/dgx-spark-book)

---

## 📖 關於本書

這是一本為 **完全新手** 設計的 Nvidia DGX Spark 教學書。不需要任何 Linux 或 AI 的先備知識，從開箱到部署 AI Agent，一步一步帶你玩透 128GB 統一記憶體的無限可能。

### ✨ 特色

| 特色 | 說明 |
|------|------|
| 🎯 **小白友善** | 每個指令都解釋「為什麼要下」，不假設先備知識 |
| 🖥️ **128GB 統一記憶體** | 體驗消費級 GPU 做不到的事：跑 120B 模型、微調 FLUX、預訓練小模型 |
| 🤖 **Claude Code 輔助** | 用 AI 寫 AI，自然語言就能完成複雜部署 |
| 📊 **實測數據** | 所有效能數據都是實際跑出來的 |
| 📚 **25 章完整內容** | 從硬體到多機叢集，涵蓋所有實用場景 |
| 🔄 **持續更新** | 社群驅動，跟上最新的模型與工具 |

### 🎯 適合誰讀

- **完全新手**：沒碰過 Linux？沒關係，從第 1 章開始
- **AI 愛好者**：想在本機跑大模型，但受限於顯卡記憶體
- **開發者**：想微調模型、部署 Agent、整合到自己的應用
- **研究人員**：需要預訓練領域模型、做 RAG、知識圖譜

---

## 🌐 線上閱讀

**👉 [立即開始閱讀](https://alingowangxr.github.io/dgx-spark-book/)**

網站已部署在 GitHub Pages，無需安裝，直接在瀏覽器中閱讀！

支援：
- 🔍 全文搜尋
- 📱 手機/平板/桌面響應式設計
- 🌙 暗色模式
- 📑 側邊欄導航

---

## 🚀 本地部署

如果你想在本機運行或貢獻內容：

```bash
# 1. 複製專案
git clone https://github.com/alingowangxr/dgx-spark-book.git
cd dgx-spark-book

# 2. 安裝相依套件
npm install

# 3. 啟動開發伺服器
npm run dev

# 4. 打開瀏覽器
# http://localhost:5173
```

### 建置靜態網站

```bash
npm run build      # 建置到 docs/.vitepress/dist/
npm run preview    # 預覽建置結果
```

---

## 📚 章節總覽

### 第一篇：硬體與系統建置
| 章節 | 標題 |
|------|------|
| [第 1 章](docs/guide/chapter1/index.md) | DGX Spark 硬體總覽 |
| [第 2 章](docs/guide/chapter2/index.md) | DGX OS 安裝與首次開機 |
| [第 3 章](docs/guide/chapter3/index.md) | Linux 環境建置與 Claude Code 安裝 |
| [第 4 章](docs/guide/chapter4/index.md) | 遠端桌面與網路存取 |

### 第二篇：LLM 推論入門
| 章節 | 標題 |
|------|------|
| [第 5 章](docs/guide/chapter5/index.md) | Ollama — 在 128 GB 上跑超大模型 |
| [第 6 章](docs/guide/chapter6/index.md) | Open WebUI — 瀏覽器裡的 AI 助手 |
| [第 7 章](docs/guide/chapter7/index.md) | LM Studio — Headless 模型服務 |
| [第 8 章](docs/guide/chapter8/index.md) | llama.cpp 與 Nemotron — 輕量原生推論 |

### 第三篇：LLM 推論進階
| 章節 | 標題 |
|------|------|
| [第 9 章](docs/guide/chapter9/index.md) | vLLM — 高吞吐量推論伺服器 |
| [第 10 章](docs/guide/chapter10/index.md) | TensorRT-LLM — NVIDIA 原生加速引擎 |
| [第 11 章](docs/guide/chapter11/index.md) | SGLang 與推測性解碼 |
| [第 12 章](docs/guide/chapter12/index.md) | NIM 推論微服務與引擎總比較 |

### 第四篇：多媒體 AI 生成
| 章節 | 標題 |
|------|------|
| [第 13 章](docs/guide/chapter13/index.md) | 圖片與影片生成 |
| [第 14 章](docs/guide/chapter14/index.md) | 音訊、語音與音樂 AI |

### 第五篇：模型微調與訓練
| 章節 | 標題 |
|------|------|
| [第 15 章](docs/guide/chapter15/index.md) | LoRA / QLoRA 微調實戰 |
| [第 16 章](docs/guide/chapter16/index.md) | Unsloth — 最快的微調框架 |
| [第 17 章](docs/guide/chapter17/index.md) | LLaMA Factory、NeMo 與 PyTorch 微調 |
| [第 18 章](docs/guide/chapter18/index.md) | 影像模型微調 — FLUX Dreambooth LoRA |
| [第 19 章](docs/guide/chapter19/index.md) | 預訓練中小型語言模型 |

### 第六篇：多模態 AI 與智慧代理
| 章節 | 標題 |
|------|------|
| [第 20 章](docs/guide/chapter20/index.md) | 多模態推論與即時視覺 AI |
| [第 21 章](docs/guide/chapter21/index.md) | RAG 與知識圖譜 |
| [第 22 章](docs/guide/chapter22/index.md) | AI Agent 與安全沙箱 |

### 第七篇：科學計算、開發工具與擴展
| 章節 | 標題 |
|------|------|
| [第 23 章](docs/guide/chapter23/index.md) | CUDA-X 資料科學、JAX 與特殊領域應用 |
| [第 24 章](docs/guide/chapter24/index.md) | 開發環境與 AI 輔助程式開發 |
| [第 25 章](docs/guide/chapter25/index.md) | 多機互連與分散式運算 |

### 附錄
- [附錄 A](docs/guide/appendix-a/index.md)：Claude Code 常用指令速查表
- [附錄 B](docs/guide/appendix-b/index.md)：NVIDIA 官方 Playbook 與本書章節對照表
- [附錄 C](docs/guide/appendix-c/index.md)：推薦模型清單與效能基準數據
- [附錄 E](docs/guide/appendix-e/index.md)：DGX Spark 硬體規格速查表
- [常見問題 FAQ](docs/guide/faq.md)
- [推薦模型清單](docs/guide/models.md)

---

## 🏗️ 專案結構

```
dgx-spark-book/
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions 自動部署
├── docs/
│   ├── .vitepress/
│   │   ├── config.ts            # VitePress 設定
│   │   └── theme/               # 自訂主題
│   │       ├── index.ts
│   │       └── style.css        # 活潑教學風樣式
│   ├── guide/
│   │   ├── chapter1/            # 第 1 章
│   │   ├── chapter2/            # 第 2 章
│   │   ├── ...
│   │   ├── chapter25/           # 第 25 章
│   │   ├── appendix-a/          # 附錄 A
│   │   ├── appendix-b/          # 附錄 B
│   │   ├── appendix-c/          # 附錄 C
│   │   ├── appendix-e/          # 附錄 E
│   │   ├── models.md            # 推薦模型清單
│   │   └── faq.md               # 常見問題
│   ├── public/
│   │   └── logo.svg             # 網站 Logo
│   └── index.md                 # 首頁
├── package.json
└── README.md
```

---

## 🛠️ 技術棧

- **[VitePress](https://vitepress.dev/)** — 靜態網站生成器
- **[GitHub Actions](https://github.com/features/actions)** — 自動部署到 GitHub Pages
- **[Markdown](https://daringfireball.net/projects/markdown/)** — 內容撰寫格式
- **自訂主題** — 活潑教學風（漸層色彩、提示框、表格美化）

---

## 📊 專案統計

| 項目 | 數值 |
|------|------|
| 總行數 | 20,000+ 行 |
| 章節數 | 25 章 + 4 個附錄 |
| 程式碼範例 | 500+ 個 |
| 表格 | 100+ 個 |
| 疑難排解 Q&A | 50+ 個 |

---

## 📝 授權

本專案以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 授權釋出。

你可以自由地：
- **分享** — 在任何媒介以任何形式複製、發行本作品
- **改編** — 修改、轉換或以本作品為基礎進行創作

只要你遵守以下條件：
- **署名** — 給予適當的 credit
- **相同方式分享** — 如果你改編本作品，必須以相同授權條款發行

---

## 🙏 致謝

- [NVIDIA](https://www.nvidia.com/) — DGX Spark 硬體與軟體支援
- [VitePress](https://vitepress.dev/) — 優秀的靜態網站生成器
- 所有開源社群貢獻者

---

## 📮 回饋與貢獻

歡迎提出 Issue 或 Pull Request！

- 發現錯誤？[開一個 Issue](https://github.com/alingowangxr/dgx-spark-book/issues)
- 想要貢獻內容？[Fork 這個專案](https://github.com/alingowangxr/dgx-spark-book/fork)

---

<div align="center">

**⭐ 如果這本書對你有幫助，請給個 Star！**

Made with ❤️ for the DGX Spark community

</div>
