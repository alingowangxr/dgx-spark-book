# 第 10 章：TensorRT-LLM — NVIDIA 原生加速引擎

::: tip 🎯 本章你將學到什麼
- TensorRT-LLM 的架構和核心技術
- 模型編譯流程：從 Hugging Face 到 TensorRT Engine
- 單機部署與 quickstart 驗證
- 整合 Open WebUI
- 進階效能調校
- 疑難排解
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 10-1 TRT-LLM 架構與核心技術

### 10-1-1 什麼是 TensorRT-LLM？

TensorRT-LLM（簡稱 TRT-LLM）是 NVIDIA 官方開發的大型語言模型推論加速引擎。它是 NVIDIA 長期積累的 TensorRT 技術在 LLM 領域的延伸。

::: info 🤔 TensorRT 是什麼？
TensorRT 是 NVIDIA 的深度学习推理最佳化 SDK，已經存在多年。它會把訓練好的模型「編譯」成針對特定 GPU 最佳化的執行檔（稱為 engine）。

想像一下：
- **一般推論** = 用直譯語言（如 Python）執行程式
- **TensorRT** = 把程式編譯成機器碼，針對你的 CPU 做了所有可能的最佳化

TRT-LLM 就是把這套技術應用到 LLM 上。
:::

**TRT-LLM 的技術堆疊：**

```
┌─────────────────────────────────────┐
│         應用程式層                    │
│   Open WebUI / 自訂前端 / API 客戶端  │
├─────────────────────────────────────┤
│         API 伺服器層                  │
│   Triton Inference Server            │
│   （支援 gRPC + HTTP + OpenAI API）   │
├─────────────────────────────────────┤
│         TRT-LLM Backend              │
│   • Continuous Batching              │
│   • Paged KV Cache                   │
│   • In-flight Batching               │
│   • Speculative Decoding             │
├─────────────────────────────────────┤
│         TensorRT Engine              │
│   • 圖層融合（Layer Fusion）           │
│   • Kernel Auto-Tuning               │
│   • 精度最佳化（FP8/INT8/FP4）         │
│   • 記憶體最佳化                      │
├─────────────────────────────────────┤
│         CUDA + cuBLAS + cuDNN        │
├─────────────────────────────────────┤
│         NVIDIA GPU（Blackwell）        │
└─────────────────────────────────────┘
```

### 10-1-2 TRT-LLM 的核心最佳化技術

| 技術 | 說明 | 效果 |
|------|------|------|
| **圖層融合（Layer Fusion）** | 把多個連續的運算融合成一個 kernel，減少記憶體讀寫 | 速度提升 20-40% |
| **Kernel Auto-Tuning** | 自動測試多種 kernel 實現，選擇最快的 | 針對每種 GPU 找到最佳配置 |
| **In-flight Batching** | 類似 Continuous Batching，但在 GPU kernel 層級實現 | 比 vLLM 的 batching 更高效 |
| **Paged KV Cache** | 與 vLLM 類似的分頁管理 | 減少記憶體碎片 |
| **FP8 / INT8 量化** | 在編譯時進行量化最佳化 | 速度提升 1.5-2x |
| **GEMM 最佳化** | 針對矩陣乘法使用最佳化的 cuBLAS kernel | 核心運算加速 |
| **多 GPU 支援** | Tensor Parallelism + Pipeline Parallelism | 部署超大模型 |

### 10-1-3 支援的模型

TensorRT-LLM 支援的模型家族（持續增加中）：

| 模型家族 | 代表模型 | 參數量 | 備註 |
|---------|---------|--------|------|
| **Llama** | Llama 3.1 8B/70B | 8B-405B | 廣泛支援，最佳化最完整 |
| **Llama** | Llama 3.3 70B | 70B | 最新版本 |
| **Qwen** | Qwen3-8B | 8B | 中文最佳 |
| **Qwen** | Qwen2.5-72B | 72B | 大型中文模型 |
| **Mistral** | Mistral 7B / Mixtral 8x7B | 7B-56B | 輕量高效 |
| **Gemma** | Gemma 2 9B/27B | 9B-27B | Google 開源 |
| **Falcon** | Falcon 40B/180B | 40B-180B | |
| **Baichuan** | Baichuan 2 13B | 13B | 中文 |
| **DeepSeek** | DeepSeek-V3 | 671B | MoE 架構 |
| **Nemotron** | Nemotron-3-Super | 120B | NVIDIA 自家模型 |

::: warning ⚠️ 模型支援限制
TRT-LLM 的模型支援是由 NVIDIA 團隊逐一實現的。不像 vLLM 那樣「社群貢獻即可支援新模型」，TRT-LLM 需要 NVIDIA 工程師為每個模型架構編寫最佳化程式碼。

因此，如果你使用的模型不在支援列表中，TRT-LLM 可能無法使用。使用前請先查閱[官方支援模型列表](https://github.com/NVIDIA/TensorRT-LLM)。
:::

### 10-1-4 TRT-LLM vs. vLLM 的核心差異

::: info 🤔 TRT-LLM vs. vLLM 的核心差異
- **vLLM**：Python 為主，靈活性高，社群活躍，新模型支援快
- **TRT-LLM**：C++/CUDA 為主，極致效能，NVIDIA 官方最佳化，新模型支援較慢

簡單說：vLLM 像是改裝車（好調校、零件好找），TRT-LLM 像是 F1 賽車（極致效能但需要專業團隊維護）。
:::

| 特性 | TRT-LLM | vLLM |
|------|---------|------|
| **開發者** | NVIDIA 官方 | UC Berkeley + 社群 |
| **語言** | C++/CUDA | Python + Triton |
| **效能** | **極致**（編譯期最佳化） | 高（執行期最佳化） |
| **靈活性** | 低（需重新編譯 engine） | 高（換模型即可） |
| **模型支援** | 較少（但持續增加） | 較多（社群貢獻） |
| **部署難度** | 較高（需要編譯 engine） | 較低（直接載入） |
| **啟動時間** | 快（engine 已編譯好） | 慢（需要載入權重） |
| **適合場景** | 生產環境、效能優先、固定模型 | 開發、測試、經常換模型 |
| **量化支援** | FP8、INT8、FP4 | FP8、INT4、NVFP4、MXFP4 |
| **多 GPU** | Tensor + Pipeline Parallelism | Tensor Parallelism |

---

## 10-2 模型編譯流程

### 10-2-1 為什麼需要編譯？

與 vLLM 直接載入 Hugging Face 模型不同，TRT-LLM 需要先把模型「編譯」成 TensorRT engine。

```
Hugging Face 模型（PyTorch 權重）
         ↓
    模型轉換（build.py）
         ↓
TensorRT Engine（最佳化的 binary）
         ↓
    Triton Server 載入
         ↓
    提供推論服務
```

::: info 🤔 編譯的好處
編譯過程會做以下最佳化：
1. **圖層融合**：把多個連續的運算合併，減少記憶體存取
2. **Kernel 選擇**：針對你的 GPU 型號選擇最快的 kernel 實現
3. **記憶體配置**：預先規劃記憶體佈局，避免執行期動態分配
4. **精度轉換**：把權重轉換為最佳精度（FP8、INT8 等）

結果：推論速度比直接載入 PyTorch 模型快 1.5-3 倍。
:::

### 10-2-2 編譯步驟

```bash
# 步驟 1：啟動 TRT-LLM 開發環境
docker run -it \
  --gpus all \
  --network host \
  --shm-size=8g \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  bash

# 步驟 2：在容器內執行編譯
cd /opt/tritonserver/tensorrtllm/examples/llama

# 以 Llama 3.1 8B 為例
python build.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --dtype bfloat16 \
  --use_gpt_attention_plugin bfloat16 \
  --use_gemm_plugin bfloat16 \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_output_len 2048 \
  --output_dir /tmp/llama-engine

# 步驟 3：編譯完成後，engine 會輸出到 /tmp/llama-engine
ls /tmp/llama-engine
# 會看到 rank0.engine 等檔案
```

**編譯參數解釋：**

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--model_dir` | Hugging Face 模型路徑或名稱 | 模型名稱或本地路徑 |
| `--dtype` | 資料精度 | bfloat16（品質）/ float8（速度） |
| `--use_gpt_attention_plugin` | 啟用 Attention plugin | bfloat16 |
| `--use_gemm_plugin` | 啟用 GEMM plugin | bfloat16 |
| `--max_batch_size` | 最大批次大小 | 8-32（越大需要越多記憶體） |
| `--max_input_len` | 最大輸入長度 | 4096 |
| `--max_output_len` | 最大輸出長度 | 2048 |
| `--output_dir` | Engine 輸出目錄 | 自訂路徑 |

::: warning ⚠️ 編譯時間
編譯過程可能需要 10-30 分鐘，取決於模型大小和 GPU 效能。編譯完成後，engine 可以重複使用，不需要每次都編譯。
:::

### 10-2-3 使用量化模型編譯

```bash
# 使用 FP8 量化編譯
python build.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --dtype float8 \
  --use_gpt_attention_plugin float8 \
  --use_gemm_plugin float8 \
  --max_batch_size 16 \
  --max_input_len 4096 \
  --max_output_len 2048 \
  --output_dir /tmp/llama-fp8-engine

# 使用 INT8 量化編譯（需要先校準）
python build.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --use_int8_weights \
  --max_batch_size 16 \
  --max_input_len 4096 \
  --max_output_len 2048 \
  --output_dir /tmp/llama-int8-engine
```

**量化格式比較：**

| 格式 | 編譯時間 | Engine 大小 | 推論速度 | 品質損失 |
|------|---------|------------|---------|---------|
| BF16 | 基準 | 基準 | 基準 | 無 |
| FP8 | 稍長 | 約 1/2 | +30-50% | 極小 |
| INT8 | 最長（需校準） | 約 1/2 | +40-60% | 小 |

---

## 10-3 單機部署

### 10-3-1 確認 Docker 權限與 GPU

```bash
# 檢查 Docker 狀態
docker ps
# 應該沒有錯誤訊息

# 檢查 GPU 狀態
nvidia-smi
# 確認顯示 GPU 型號、驅動版本、可用記憶體

# 測試 Docker GPU 支援
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# 應該顯示與主機相同的 GPU 資訊
```

### 10-3-2 設定 NGC 認證並拉取容器

TensorRT-LLM 的容器映像存放在 NVIDIA GPU Cloud（NGC）上，需要認證才能存取。

```bash
# 設定 NGC API Key（從 https://ngc.nvidia.com 取得）
export NGC_API_KEY="你的API_KEY"

# 登入 NGC Docker Registry
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 拉取 TRT-LLM 容器
docker pull nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3
```

::: info 🤔 如何取得 NGC API Key？
1. 前往 [NGC 網站](https://ngc.nvidia.com)
2. 用 NVIDIA 帳號登入（沒有的話先註冊）
3. 點擊右上角帳號名稱 → **Setup** → **API Key**
4. 點擊 **Generate API Key**
5. 複製並妥善儲存（只會顯示一次！）
:::

### 10-3-3 quickstart 驗證：LLM

TRT-LLM 內建 quickstart 腳本，可以快速驗證環境是否正常：

```bash
docker run --rm \
  --gpus all \
  --network host \
  --shm-size=8g \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  python3 /opt/tritonserver/tensorrtllm/backends/tensorrtllm/scripts/quickstart.py
```

這個腳本會自動執行以下步驟：

```
1. 下載一個小型測試模型（如 GPT-2 small）
   ↓
2. 編譯為 TensorRT engine
   ↓
3. 啟動 Triton Inference Server
   ↓
4. 發送測試請求
   ↓
5. 顯示效能數據（吞吐量、延遲等）
   ↓
6. 清理測試資源
```

**預期輸出：**

```
=== Quickstart Test Results ===
Model: gpt2-small
Input: "Hello, my name is"
Output: " Hello, my name is John and I am a..."
Tokens generated: 50
Generation time: 1.23s
Tokens per second: 40.65
=== Test Passed ===
```

### 10-3-4 quickstart 驗證：多模態（VLM）

如果你需要測試視覺語言模型（Vision-Language Model）：

```bash
docker run --rm \
  --gpus all \
  --network host \
  --shm-size=8g \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  python3 /opt/tritonserver/tensorrtllm/backends/tensorrtllm/scripts/quickstart_vlm.py
```

這個腳本會測試影像理解能力，例如：
- 圖片描述生成
- 視覺問答
- 影像中的文字辨識（OCR）

### 10-3-5 啟動 Triton Inference Server

編譯好 engine 後，用 Triton Inference Server 提供推論服務：

```bash
# 建立模型倉庫目錄結構
mkdir -p /models/llm/1
cp /tmp/llama-engine/* /models/llm/1/

# 建立模型配置檔
cat > /models/llm/config.pbtxt << 'EOF'
name: "llm"
backend: "tensorrtllm"
max_batch_size: 8

parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_batching" }
}

parameters {
  key: "gpt_model_path"
  value: { string_value: "/models/llm/1" }
}

instance_group {
  count: 1
  kind: KIND_GPU
}
EOF

# 啟動 Triton Server
docker run -d \
  --name trt-llm \
  --gpus all \
  --network host \
  --shm-size=8g \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  tritonserver \
  --model-repository=/models \
  --allow-http=true
```

::: info 🤔 config.pbtxt 是什麼？
config.pbtxt 是 Triton Inference Server 的模型配置檔，告訴伺服器：
- 模型名稱和使用的 backend
- 最大批次大小
- 模型路徑和參數
- 硬體分配（GPU/CPU）

這個檔案的格式是 protobuf 的文字版本。
:::

### 10-3-6 測試 API

```bash
# 等待伺服器啟動完成（約 1-2 分鐘）
curl http://localhost:8000/v2/health/ready
# 回應 {"status": "ready"} 表示正常

# 測試推論
curl http://localhost:8000/v2/models/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "你好！請介紹一下你自己。",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**預期回應：**

```json
{
  "model_name": "llm",
  "model_version": "1",
  "text_output": "你好！我是一個大型語言模型...",
  "cum_log_probs": -12.34,
  "output_log_probs": [-1.2, -0.8, -1.1, ...]
}
```

---

## 10-4 使用預編譯引擎（簡化流程）

如果你不想自己編譯引擎，NVIDIA 提供了一些預編譯的 NIM 容器，內部使用 TRT-LLM 作為後端：

```bash
# 登入 NGC
export NGC_API_KEY="你的API_KEY"
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 直接啟動預編譯的 NIM（內部使用 TRT-LLM）
docker run -d \
  --name nim-trtllm \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

這種方式結合了 TRT-LLM 的效能和 NIM 的易用性。

---

## 10-5 整合 Open WebUI

### 10-5-1 連接 TRT-LLM 到 Open WebUI

如果 Open WebUI 已經在執行（第 6 章），只需要添加 TRT-LLM 作為新的模型提供者：

**步驟：**

1. 打開瀏覽器，前往 Open WebUI（通常是 `http://localhost:3000`）
2. 點擊左下角的 **Admin Panel**（管理面板）
3. 進入 **Settings → Connections**（設定 → 連線）
4. 點擊 **Add Connection**（添加連線）
5. 填寫以下資訊：

| 欄位 | 值 |
|------|-----|
| **Connection Name** | TRT-LLM |
| **API Base URL** | `http://localhost:8000/v1` |
| **API Key** | 任意填寫（TRT-LLM 不需要認證） |

6. 點擊 **Save**（儲存）
7. 回到聊天頁面，在模型選擇器中應該能看到 TRT-LLM 的模型

::: tip 💡 如果 Open WebUI 沒有顯示模型
1. 確認 TRT-LLM 的 API 伺服器正在運行：`curl http://localhost:8000/v2/health/ready`
2. 確認 URL 正確（注意是 v1 還是 v2）
3. 重新整理 Open WebUI 頁面
4. 檢查 Open WebUI 日誌：`docker logs open-webui`
:::

### 10-5-2 使用 Triton 的 gRPC 介面

Triton Server 預設同時支援 HTTP 和 gRPC：

| 介面 | Port | 用途 |
|------|------|------|
| HTTP REST API | 8000 | 瀏覽器、curl、Open WebUI |
| gRPC | 8001 | 高效能客戶端（Python/C++ SDK） |
| Metrics | 8002 | Prometheus 監控 |

**用 Python gRPC 客戶端測試：**

```python
import tritonclient.grpc as grpcclient

# 建立客戶端
client = grpcclient.InferenceServerClient(url="localhost:8001")

# 確認伺服器就緒
print(client.is_server_ready())

# 準備輸入
inputs = [grpcclient.InferInput("text_input", [1], "BYTES")]
inputs[0].set_data_from_numpy(["你好！".encode("utf-8")])

outputs = [grpcclient.InferRequestedOutput("text_output")]

# 發送推論請求
response = client.infer(model_name="llm", inputs=inputs, outputs=outputs)
print(response.as_numpy("text_output")[0].decode("utf-8"))
```

---

## 10-6 進階效能調校

### 10-6-1 批次大小調校

| max_batch_size | 記憶體用量 | 吞吐量 | 延遲 | 適合場景 |
|---------------|-----------|--------|------|---------|
| 1 | 最小 | 最低 | 最低 | 單人互動式對話 |
| 4 | 較小 | 中等 | 低 | 小型團隊使用 |
| 8 | 中等 | 高 | 中等 | ✅ 一般推薦 |
| 16 | 較大 | 很高 | 較高 | 高併發場景 |
| 32 | 最大 | 最高 | 高 | 基準測試 |

### 10-6-2 精度選擇

| 精度 | 速度 | 記憶體 | 品質 | 推薦場景 |
|------|------|--------|------|---------|
| BF16 | 基準 | 基準 | 最佳 | 品質優先 |
| FP16 | +10% | -50% | 接近 BF16 | 平衡 |
| FP8 | +30-50% | -50% | 極小損失 | ✅ 推薦 |
| INT8 | +40-60% | -50% | 小損失 | 速度優先 |

### 10-6-3 In-flight Batching 調校

In-flight Batching 是 TRT-LLM 的獨家技術，比傳統的 Continuous Batching 更高效：

```bash
# 在 config.pbtxt 中啟用
parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_batching" }
}

# 設定最大排隊請求數
parameters {
  key: "max_beam_width"
  value: { string_value: "1" }
}
```

---

## 10-7 單機清理

```bash
# 停止並移除容器
docker stop trt-llm
docker rm trt-llm

# 移除映像檔
docker rmi nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# 清理模型倉庫
rm -rf /models

# 清理編譯產生的 engine
rm -rf /tmp/llama-engine
```

---

## 10-8 疑難排解 FAQ

### Q1：Docker pull 失敗？

```bash
# 確認 NGC API Key 正確
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 如果還是失敗，檢查網路
ping nvcr.io

# 常見錯誤和解決方案：
# "unauthorized" → API Key 不正確或已過期
# "connection refused" → 網路問題，檢查防火牆
# "no space left on device" → 磁碟空間不足
```

::: tip 💡 測試網路連線
```bash
# 測試 NGC 網站連線
curl -I https://ngc.nvidia.com

# 測試 Docker Registry 連線
curl -I https://nvcr.io/v2/
```
:::

### Q2：TRT-LLM 編譯模型失敗？

TRT-LLM 需要先把模型編譯為 TensorRT engine。如果編譯失敗：

**步驟 1：確認模型在支援列表中**
```bash
# 查看支援的模型列表
ls /opt/tritonserver/tensorrtllm/examples/
```

**步驟 2：確認有足夠的記憶體**
```bash
# 編譯過程需要大量 CPU 記憶體和 GPU 記憶體
free -h        # 檢查系統記憶體
nvidia-smi     # 檢查 GPU 記憶體
```

**步驟 3：查看編譯日誌**
```bash
# 編譯時加上 verbose 模式
python build.py --model_dir ... --verbose

# 或查看容器日誌
docker logs trt-llm
```

**常見編譯錯誤：**

| 錯誤 | 原因 | 解決方案 |
|------|------|---------|
| `CUDA out of memory` | GPU 記憶體不足 | 降低 max_batch_size |
| `Unsupported model` | 模型不在支援列表中 | 換用支援的模型 |
| `Plugin creation failed` | Plugin 版本不匹配 | 更新 TRT-LLM 版本 |
| `Build timeout` | 編譯時間過長 | 增加 timeout 或檢查資源 |

### Q3：GPU 記憶體不足？

```bash
# 解決方案 1：降低 batch size（在 build.py 中）
--max_batch_size 4

# 解決方案 2：減少上下文長度
--max_input_len 2048
--max_output_len 1024

# 解決方案 3：使用量化
--dtype float8

# 監控記憶體用量
watch -n 1 nvidia-smi
```

### Q4：Triton Server 啟動後無法連線？

```bash
# 檢查伺服器狀態
curl http://localhost:8000/v2/health/ready

# 查看日誌
docker logs trt-llm

# 常見原因：
# 1. 模型載入中（需要等待）
# 2. Engine 路徑不正確
# 3. config.pbtxt 格式錯誤
# 4. Port 被其他行程佔用
```

### Q5：如何查看 TRT-LLM 的效能指標？

```bash
# Triton Server 提供 Prometheus 格式的指標
curl http://localhost:8002/metrics

# 關鍵指標：
# nv_inference_request_success - 成功請求數
# nv_inference_request_failure - 失敗請求數
# nv_inference_compute_infer_duration_us - 推論時間
# nv_inference_compute_output_duration_us - 輸出時間
```

### Q6：TRT-LLM 和 vLLM 該選哪個？

| 你的情況 | 推薦 |
|---------|------|
| 需要最快速度、模型在支援列表中 | TRT-LLM |
| 需要靈活性、經常換模型 | vLLM |
| 不想自己編譯 engine | vLLM 或 NIM |
| 需要最新模型支援 | vLLM |
| 企業級生產環境 | TRT-LLM 或 NIM |

---

## 10-9 本章小結

::: success ✅ 你現在知道了
- TensorRT-LLM 是 NVIDIA 官方的極致效能推論引擎
- TRT-LLM 透過編譯期最佳化（圖層融合、Kernel Auto-Tuning）達到最高效能
- 需要從 NGC 拉取容器映像，並設定 API Key 認證
- 模型需要先編譯為 TensorRT engine 才能使用
- quickstart 腳本可以快速驗證環境是否正常
- In-flight Batching 是 TRT-LLM 的獨家技術，比傳統 batching 更高效
- 可以整合到 Open WebUI 中使用
- TRT-LLM 適合追求極致效能、模型固定的生產環境
:::

::: tip 🚀 下一章預告
接下來我們要介紹另一個強大的推論框架 — SGLang，它帶來了獨特的 RadixAttention 和結構化生成技術，在多輪對話和 RAG 場景中表現出色！

👉 [前往第 11 章：SGLang 與推測性解碼 →](/guide/chapter11/)
:::

::: info 📝 上一章
← [回到第 9 章：vLLM](/guide/chapter9/)
:::
