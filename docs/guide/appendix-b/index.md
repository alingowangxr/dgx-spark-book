# 附錄 B：NVIDIA 官方 Playbook 與本書章節對照表

本附錄提供 NVIDIA 官方 DGX Spark Playbook 文件與本書各章節的完整對照，方便讀者快速定位官方資源與本書教學內容。

---

## 第一篇：系統建置

### 硬體設置與初始化

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| DGX Spark Setup Guide | 第 1 章 | 開箱檢查、硬體認識、規格總覽 | ⭐ | 30 分鐘 |
| DGX Spark Setup Guide | 第 2 章 | 開機流程、初始設定、DGX OS 安裝 | ⭐ | 45 分鐘 |
| DGX Spark Setup Guide | 第 3 章 | 系統更新、驅動程式安裝、CUDA 環境配置 | ⭐⭐ | 60 分鐘 |

### 網路與遠端存取

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Remote Access Configuration | 第 4 章 | SSH 金鑰設定、SSH 連線、終端機工具選擇 | ⭐ | 30 分鐘 |
| Remote Access Configuration | 第 4 章 | Tailscale 安裝與設定、遠端桌面 VNC/RDP | ⭐⭐ | 45 分鐘 |
| Network Best Practices | 第 4 章 | 靜態 IP 設定、DNS 配置、防火牆規則 | ⭐⭐ | 30 分鐘 |

### 開發環境建置

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Docker on DGX Spark | 第 3 章 | NVIDIA Container Toolkit 安裝、Docker 基本操作 | ⭐⭐ | 45 分鐘 |
| Development Environment Setup | 第 3 章 | Python 虛擬環境、Conda 安裝、pip 套件管理 | ⭐ | 30 分鐘 |
| VS Code Remote Development | 第 24 章 | VS Code Server 安裝、Remote-SSH 連線、擴充套件 | ⭐⭐ | 45 分鐘 |

---

## 第二篇：LLM 推論入門

### 基礎推論框架

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Ollama Quick Start | 第 5 章 | Ollama 安裝、模型下載、基本對話 | ⭐ | 20 分鐘 |
| Ollama Quick Start | 第 5 章 | 模型管理、API 呼叫、環境變數設定 | ⭐ | 30 分鐘 |
| Open WebUI Deployment | 第 6 章 | Docker 部署 Open WebUI、使用者管理、外掛安裝 | ⭐⭐ | 45 分鐘 |
| Open WebUI Deployment | 第 6 章 | RAG 管道設定、檔案上傳、知識庫建立 | ⭐⭐⭐ | 60 分鐘 |

### 高效能推論引擎

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| llama.cpp on DGX Spark | 第 8 章 | llama.cpp 編譯、GGUF 模型載入、量化格式選擇 | ⭐⭐ | 45 分鐘 |
| llama.cpp on DGX Spark | 第 8 章 | 伺服器模式啟動、API 端點設定、批次推論 | ⭐⭐ | 30 分鐘 |
| LM Studio Alternative | 第 7 章 | LM Studio 安裝、模型匯入、圖形介面操作 | ⭐ | 20 分鐘 |
| Model Selection Guide | 第 7 章 | 模型尺寸選擇、記憶體用量評估、效能基準測試 | ⭐⭐ | 30 分鐘 |

---

## 第三篇：LLM 推論進階

### 企業級推論服務

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| vLLM Deployment | 第 9 章 | vLLM 安裝、PagedAttention 原理、模型服務化 | ⭐⭐⭐ | 60 分鐘 |
| vLLM Deployment | 第 9 章 | 多模型同時服務、自動擴展、效能調校 | ⭐⭐⭐ | 45 分鐘 |
| vLLM Deployment | 第 9 章 | OpenAI 相容 API、串流輸出、Token 用量統計 | ⭐⭐ | 30 分鐘 |

### NVIDIA 生態系

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| TensorRT-LLM Quick Start | 第 10 章 | TensorRT-LLM 安裝、模型轉換、引擎最佳化 | ⭐⭐⭐ | 90 分鐘 |
| TensorRT-LLM Quick Start | 第 10 章 | INT8/FP8 量化、KV Cache 最佳化、批次推論 | ⭐⭐⭐ | 60 分鐘 |
| TensorRT-LLM Quick Start | 第 10 章 | Triton Inference Server 整合、效能基準測試 | ⭐⭐⭐ | 45 分鐘 |
| SGLang Setup | 第 11 章 | SGLang 安裝、RadixAttention、結構化輸出 | ⭐⭐⭐ | 60 分鐘 |
| SGLang Setup | 第 11 章 | 提示詞編程、多步驟推理、效能調校 | ⭐⭐⭐ | 45 分鐘 |
| NIM Deployment | 第 12 章 | NVIDIA NIM 容器下載、模型部署、API 設定 | ⭐⭐ | 45 分鐘 |
| NIM Deployment | 第 12 章 | 企業模型目錄、授權管理、監控儀表板 | ⭐⭐ | 30 分鐘 |

### 推論效能調校

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Inference Performance Tuning | 第 9-12 章 | 記憶體管理、批次大小調校、並行策略 | ⭐⭐⭐ | 60 分鐘 |
| Benchmarking Guide | 第 9-12 章 | 基準測試工具、效能指標解讀、比較分析 | ⭐⭐ | 45 分鐘 |

---

## 第四篇：多媒體 AI 生成

### 影像生成

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| ComfyUI on DGX Spark | 第 13 章 | ComfyUI 安裝、節點基礎、基本工作流 | ⭐⭐ | 45 分鐘 |
| ComfyUI on DGX Spark | 第 13 章 | FLUX 模型部署、ControlNet 整合、高解析度生成 | ⭐⭐⭐ | 60 分鐘 |
| ComfyUI on DGX Spark | 第 13 章 | 自訂節點安裝、工作流匯入匯出、批次生成 | ⭐⭐ | 30 分鐘 |
| FLUX Dreambooth | 第 18 章 | Dreambooth 微調、個人化模型訓練、推理應用 | ⭐⭐⭐ | 90 分鐘 |

### 音訊 AI

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Audio AI Pipeline | 第 14 章 | Whisper 語音辨識、音訊轉文字、多語言支援 | ⭐⭐ | 30 分鐘 |
| Audio AI Pipeline | 第 14 章 | 語音合成（TTS）、聲音克隆、情感控制 | ⭐⭐ | 45 分鐘 |
| Audio AI Pipeline | 第 14 章 | 音樂生成、音訊後處理、即時串流 | ⭐⭐⭐ | 60 分鐘 |

---

## 第五篇：模型微調與訓練

### LoRA 微調

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| LoRA Fine-tuning | 第 15 章 | LoRA 原理、參數設定、資料集準備 | ⭐⭐⭐ | 60 分鐘 |
| LoRA Fine-tuning | 第 15 章 | 訓練流程、評估指標、模型合併 | ⭐⭐⭐ | 45 分鐘 |
| LoRA Fine-tuning | 第 15 章 | 多任務 LoRA、適配器管理、推論整合 | ⭐⭐⭐ | 30 分鐘 |

### Unsloth 快速微調

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Unsloth Quick Start | 第 16 章 | Unsloth 安裝、2 倍加速原理、記憶體最佳化 | ⭐⭐ | 30 分鐘 |
| Unsloth Quick Start | 第 16 章 | 指令微調、對話格式、資料集轉換 | ⭐⭐ | 45 分鐘 |
| Unsloth Quick Start | 第 16 章 | GGUF 匯出、Ollama 整合、部署流程 | ⭐⭐ | 30 分鐘 |

### NVIDIA NeMo 框架

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| NeMo AutoModel | 第 17 章 | NeMo 框架介紹、AutoModel API、快速微調 | ⭐⭐⭐ | 60 分鐘 |
| NeMo AutoModel | 第 17 章 | 資料管線、分詞器設定、訓練監控 | ⭐⭐⭐ | 45 分鐘 |
| NeMo AutoModel | 第 17 章 | 模型匯出、NIM 部署、效能驗證 | ⭐⭐⭐ | 30 分鐘 |

### 進階訓練

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Pre-training Guide | 第 19 章 | 從頭訓練原理、資料集建構、分散式訓練 | ⭐⭐⭐⭐ | 120 分鐘 |
| Pre-training Guide | 第 19 章 | 學習率排程、梯度累積、檢查點管理 | ⭐⭐⭐⭐ | 60 分鐘 |
| Pre-training Guide | 第 19 章 | 多機訓練、NCCL 通訊、效能最佳化 | ⭐⭐⭐⭐ | 90 分鐘 |

---

## 第六篇：多模態 AI 與智慧代理

### 視覺語言模型

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Live VLM WebUI | 第 20 章 | VLM 模型部署、即時影像分析、WebUI 設定 | ⭐⭐ | 45 分鐘 |
| Live VLM WebUI | 第 20 章 | 攝影機串流、物件辨識、場景理解 | ⭐⭐⭐ | 60 分鐘 |
| Live VLM WebUI | 第 20 章 | 多模態對話、影像問答、自動化流程 | ⭐⭐ | 30 分鐘 |

### RAG 檢索增強生成

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| RAG Pipeline | 第 21 章 | RAG 架構、向量資料庫、嵌入模型 | ⭐⭐ | 45 分鐘 |
| RAG Pipeline | 第 21 章 | 文件切分、索引建立、相似度搜尋 | ⭐⭐ | 30 分鐘 |
| RAG Pipeline | 第 21 章 | 提示詞工程、上下文最佳化、評估指標 | ⭐⭐⭐ | 60 分鐘 |

### AI Agent 智慧代理

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| AI Agent Setup | 第 22 章 | Agent 架構、工具呼叫、函數定義 | ⭐⭐⭐ | 60 分鐘 |
| AI Agent Setup | 第 22 章 | 多步驟規劃、記憶管理、錯誤處理 | ⭐⭐⭐ | 45 分鐘 |
| AI Agent Setup | 第 22 章 | 多 Agent 協作、任務分配、結果整合 | ⭐⭐⭐⭐ | 90 分鐘 |

---

## 第七篇：科學計算與開發工具

### 資料科學

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| RAPIDS on DGX Spark | 第 23 章 | RAPIDS 安裝、cuDF 資料處理、cuML 機器學習 | ⭐⭐ | 45 分鐘 |
| RAPIDS on DGX Spark | 第 23 章 | GPU 加速 Pandas、Dask 整合、效能比較 | ⭐⭐⭐ | 60 分鐘 |
| RAPIDS on DGX Spark | 第 23 章 | 視覺化、資料管線、即時分析 | ⭐⭐ | 30 分鐘 |

### 開發工具

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| VS Code Remote | 第 24 章 | Remote-SSH 連線、Jupyter 整合、偵錯工具 | ⭐⭐ | 30 分鐘 |
| VS Code Remote | 第 24 章 | Git 整合、終端機設定、擴充套件推薦 | ⭐ | 20 分鐘 |
| VS Code Remote | 第 24 章 | Docker 擴充、Dev Container、遠端開發 | ⭐⭐ | 45 分鐘 |

---

## 第八篇：多機互連

### 叢集建置

| Playbook 文件 | 對應章節 | 核心主題 | 難度 | 預估時間 |
|--------------|---------|---------|------|---------|
| Multi-node Setup | 第 25 章 | 網路拓撲、節點配置、SSH 免密登入 | ⭐⭐⭐ | 60 分鐘 |
| Multi-node Setup | 第 25 章 | 分散式檔案系統、時間同步、環境一致性 | ⭐⭐⭐ | 45 分鐘 |
| NCCL Testing | 第 25 章 | NCCL 原理、頻寬測試、延遲測試 | ⭐⭐⭐ | 30 分鐘 |
| NCCL Testing | 第 25 章 | 效能基準、問題診斷、最佳化建議 | ⭐⭐⭐⭐ | 60 分鐘 |

---

## 官方資源連結

| 資源 | 網址 | 說明 |
|------|------|------|
| NVIDIA DGX Spark 官方頁面 | https://www.nvidia.com/dgx-spark | 產品規格與購買 |
| NVIDIA Developer | https://developer.nvidia.com | 開發者工具與 SDK |
| NVIDIA NGC Catalog | https://catalog.ngc.nvidia.com | 容器與模型目錄 |
| NVIDIA NIM | https://www.nvidia.com/nim | 微服務部署平台 |
| NVIDIA NeMo | https://github.com/NVIDIA/NeMo | 對話式 AI 框架 |
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | 高效能推論引擎 |
| Ollama 官方 | https://ollama.com | 輕量級模型推論 |
| vLLM 官方 | https://docs.vllm.ai | 高效能推論服務 |
| SGLang 官方 | https://github.com/sgl-project/sglang | 結構化生成語言 |

---

## Playbook 文件版本追蹤

| 文件名稱 | 最新版本 | 更新日期 | 備註 |
|---------|---------|---------|------|
| DGX Spark Setup Guide | v1.2 | 2025-03 | 支援 DGX OS 24.04 |
| Ollama Quick Start | v2.0 | 2025-02 | 新增 GGUF v3 支援 |
| vLLM Deployment | v1.5 | 2025-03 | 支援 Blackwell 架構 |
| TensorRT-LLM | v0.15 | 2025-03 | FP8 量化最佳化 |
| SGLang Setup | v0.4 | 2025-02 | RadixAttention 更新 |
| NIM Deployment | v1.1 | 2025-03 | 新增更多模型 |
| LoRA Fine-tuning | v1.3 | 2025-02 | 多任務 LoRA 支援 |
| Unsloth Quick Start | v2025.2 | 2025-02 | 2 倍加速更新 |
| NeMo AutoModel | v2.0 | 2025-03 | 全新 API 設計 |
| ComfyUI on DGX Spark | v1.0 | 2025-01 | 初始版本 |
| RAPIDS on DGX Spark | v24.12 | 2025-01 | CUDA 13 支援 |
| Multi-node Setup | v1.0 | 2025-03 | 初始版本 |

---

## 學習路徑建議

### 初學者路徑（預計 2 週）

```
第 1-2 章（系統建置）→ 第 3 章（開發環境）→ 第 4 章（遠端存取）
→ 第 5 章（Ollama）→ 第 6 章（Open WebUI）→ 第 7 章（模型選擇）
```

### 進階開發者路徑（預計 4 週）

```
第 1-4 章（基礎建置）→ 第 8-9 章（llama.cpp / vLLM）
→ 第 10 章（TensorRT-LLM）→ 第 13 章（ComfyUI）
→ 第 15-16 章（LoRA / Unsloth 微調）→ 第 20 章（VLM）
```

### 企業部署路徑（預計 6 週）

```
第 1-4 章（基礎建置）→ 第 9-12 章（進階推論服務）
→ 第 17 章（NeMo 微調）→ 第 21 章（RAG 管線）
→ 第 22 章（AI Agent）→ 第 23 章（RAPIDS）→ 第 25 章（多機互連）
```

---

::: info 📝 返回
← [回到第 25 章：多機互連](/guide/chapter25/) | [首頁](/)
:::
