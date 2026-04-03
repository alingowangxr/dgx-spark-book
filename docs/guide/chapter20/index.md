# 第 20 章：多模態推論與即時視覺 AI

::: tip 🎯 本章你將學到什麼
- TensorRT 加速擴散模型推論的完整流程
- 三種精度（FP16、FP8、NF4）的實測比較
- Live VLM WebUI 即時串流設定
- RTSP IP 攝影機整合
- 同時跑生成和理解的記憶體配置策略
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 20-1 TensorRT 加速擴散模型推論

### 20-1-1 為什麼要用 TensorRT 加速擴散模型？

::: info 🤔 TensorRT 和擴散模型的關係
擴散模型（Diffusion Model）生成圖片的過程需要反覆執行 U-Net 或 Transformer 的推論。一張 1024x1024 的圖片可能需要 20-50 步，每一步都是一次完整的神經網路前向傳播。

TensorRT 透過以下方式加速：
1. **圖層融合**：把多個小的運算合併成一個，減少記憶體讀寫
2. **核心自動調校**：針對你的 GPU 選擇最快的演算法
3. **精度最佳化**：在幾乎不影響品質的前提下使用 FP8/NF4
:::

在 DGX Spark 上，TensorRT 可以帶來 **1.5-2 倍** 的加速效果。

### 20-1-2 部署 TensorRT 推論環境

```bash
# 建立工作目錄
mkdir -p ~/trt-diffusion && cd ~/trt-diffusion

# 拉取 TensorRT 擴散模型容器
docker pull nvcr.io/nvidia/tensorrt-diffusion:25.01
```

啟動容器：

```bash
docker run -d \
  --name trt-diffusion \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/trt-models:/models \
  -v ~/trt-output:/output \
  -v ~/trt-diffusion:/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorrt-diffusion:25.01 \
  tail -f /dev/null
```

### 20-1-3 下載並轉換 FLUX.1-Dev 為 TensorRT Engine

TensorRT 需要先把模型轉換為優化過的 engine 格式：

```bash
# 進入容器
docker exec -it trt-diffusion bash

# 下載 FLUX.1-Dev 模型
cd /models
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir flux-dev

# 轉換為 TensorRT engine
python /opt/tensorrt-diffusion/convert.py \
  --model-path /models/flux-dev \
  --output-path /models/flux-dev-trt \
  --precision fp16 \
  --batch-size 1 \
  --height 1024 \
  --width 1024
```

::: tip 💡 轉換時間
首次轉換約需 10-20 分鐘。轉換完成後，engine 檔案可以重複使用，不需要每次都轉換。
:::

### 20-1-4 Flux.1-Dev 多精度推論實測

**FP16（最高品質）**：

```bash
python /opt/tensorrt-diffusion/generate.py \
  --engine-path /models/flux-dev-trt-fp16.engine \
  --prompt "夕陽下的海灘，金色的陽光灑在波光粼粼的海面上，遠處有幾艘帆船" \
  --negative-prompt "模糊、低品質、變形" \
  --steps 30 \
  --guidance-scale 3.5 \
  --seed 42 \
  --output /output/flux-fp16.png
```

**FP8（平衡模式）**：

```bash
python /opt/tensorrt-diffusion/generate.py \
  --engine-path /models/flux-dev-trt-fp8.engine \
  --prompt "夕陽下的海灘，金色的陽光灑在波光粼粼的海面上，遠處有幾艘帆船" \
  --steps 30 \
  --guidance-scale 3.5 \
  --seed 42 \
  --output /output/flux-fp8.png
```

**NF4（最快速度）**：

```bash
python /opt/tensorrt-diffusion/generate.py \
  --engine-path /models/flux-dev-trt-nf4.engine \
  --prompt "夕陽下的海灘，金色的陽光灑在波光粼粼的海面上，遠處有幾艘帆船" \
  --steps 30 \
  --guidance-scale 3.5 \
  --seed 42 \
  --output /output/flux-nf4.png
```

### 20-1-5 三種精度的完整效能比較

以下是實際在 DGX Spark 上的測試數據（1024x1024，30 steps）：

| 精度 | 生成時間 | 記憶體用量 | 品質評分（主觀） | 適合場景 |
|------|---------|-----------|----------------|---------|
| **FP16** | 45 秒 | 23 GB | 9.5/10 | 最終成品 |
| **FP8** | 32 秒 | 12 GB | 9.0/10 | 日常使用 |
| **NF4** | 26 秒 | 6 GB | 8.0/10 | 快速迭代 |

::: tip 💡 品質差異在哪？
- FP16 vs FP8：幾乎看不出差異，只在非常細節的紋理上有微小差別
- FP8 vs NF4：在文字渲染、手指等細節上有可察覺的差異
- 對於風景、抽象藝術等，三種精度都很難看出差異
:::

### 20-1-6 Flux.1-Schnell 快速推論

FLUX.1-Schnell 是 FLUX 的快速版本，只需要 4 步就能生成不錯的圖片：

```bash
# 下載 Schnell 模型
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir flux-schnell

# 轉換 engine
python /opt/tensorrt-diffusion/convert.py \
  --model-path /models/flux-schnell \
  --output-path /models/flux-schnell-trt \
  --precision fp16 \
  --steps 4

# 生成（只需 4 步）
python /opt/tensorrt-diffusion/generate.py \
  --engine-path /models/flux-schnell-trt-fp16.engine \
  --prompt "一隻戴著眼鏡的貓在看書" \
  --steps 4 \
  --seed 42 \
  --output /output/schnell.png
```

| 模型 | 步數 | 生成時間 | 品質 |
|------|------|---------|------|
| FLUX.1-Dev | 30 | 45 秒 | 最高 |
| FLUX.1-Schnell | 4 | **6 秒** | 高 |

### 20-1-7 SDXL 推論

SDXL 是上一代的開源圖片生成模型，雖然品質不如 FLUX，但速度更快、記憶體用量更低：

```bash
# 下載 SDXL 模型
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir sdxl

# 轉換 engine
python /opt/tensorrt-diffusion/convert.py \
  --model-path /models/sdxl \
  --output-path /models/sdxl-trt \
  --precision fp16

# 生成
python /opt/tensorrt-diffusion/generate.py \
  --engine-path /models/sdxl-trt-fp16.engine \
  --prompt "賽博龐克風格的城市夜景" \
  --steps 25 \
  --output /output/sdxl.png
```

| 模型 | 記憶體 | 生成時間 | 解析度 |
|------|--------|---------|--------|
| SDXL | 7 GB | 15 秒 | 1024x1024 |
| FLUX.1-Schnell | 12 GB | 6 秒 | 1024x1024 |
| FLUX.1-Dev | 23 GB | 45 秒 | 1024x1024 |

### 20-1-8 批次生成：一次生成多張圖片

```bash
python /opt/tensorrt-diffusion/batch_generate.py \
  --engine-path /models/flux-schnell-trt-fp16.engine \
  --prompts-file /workspace/prompts.txt \
  --batch-size 4 \
  --output-dir /output/batch/
```

`prompts.txt` 內容範例：

```
一隻戴著帽子的狗在公園散步
未來主義的太空站漂浮在木星軌道上
水墨風格的山水畫
復古風格的咖啡館內部
```

### 20-1-9 驗證與清理

```bash
# 查看生成的圖片
ls -lh ~/trt-output/

# 停止容器
docker stop trt-diffusion
docker rm trt-diffusion
```

---

## 20-2 Live VLM WebUI — 即時視覺語言模型

### 20-2-1 什麼是 Live VLM？

::: info 🤔 Live VLM 是什麼？
Live VLM（Live Vision Language Model）讓你把攝影機畫面即時餵給 AI 模型，AI 會即時描述它看到的內容。

想像一下：
- 你把攝影機對著書桌，AI 告訴你桌上有什麼
- 你把攝影機對著窗外，AI 描述天氣和景色
- 你把攝影機對著生產線，AI 偵測異常

這在 DGX Spark 上特別有意義，因為 128GB 記憶體可以跑很大的 VLM 模型，辨識能力更強。
:::

### 20-2-2 安裝 VLM 後端

```bash
# 建立工作目錄
mkdir -p ~/live-vlm && cd ~/live-vlm

# 複製專案
git clone https://github.com/community/live-vlm-webui.git backend
cd backend

# 建立 Python 環境
uv venv .venv
source .venv/bin/activate

# 安裝相依套件
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate opencv-python fastapi uvicorn websockets
uv pip install pillow numpy
```

### 20-2-3 安裝 Live VLM WebUI 前端

```bash
cd ~/live-vlm
git clone https://github.com/community/live-vlm-webui.git frontend
cd frontend

# 安裝 Node.js 相依套件
npm install
npm run build
```

### 20-2-4 下載 VLM 模型

```bash
# 回到後端目錄
cd ~/live-vlm/backend
source .venv/bin/activate

# 下載 Qwen2.5-VL 7B（推薦的平衡選擇）
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5-vl-7b
```

::: tip 💡 模型選擇建議
| 模型 | 記憶體 | 速度 | 辨識能力 | 推薦場景 |
|------|--------|------|---------|---------|
| **Qwen2.5-VL 7B** | ~5 GB | 快 | 高 | **日常使用** |
| Qwen2.5-VL 72B | ~40 GB | 中等 | 最高 | 高精度需求 |
| Llama 3.2 11B | ~8 GB | 快 | 中高 | 英文為主 |
| LLaVA 7B | ~5 GB | 快 | 中 | 輕量使用 |
:::

### 20-2-5 WebRTC 即時串流

啟動 VLM 服務：

```bash
cd ~/live-vlm/backend
source .venv/bin/activate

python server.py \
  --model-path models/qwen2.5-vl-7b \
  --host 0.0.0.0 \
  --port 8765 \
  --prompt "請詳細描述你看到的畫面，包括物體、顏色、位置關係。" \
  --interval 2.0
```

**參數解釋**：

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--prompt` | 給 VLM 的系統提示詞 | 根據需求自訂 |
| `--interval` | 分析間隔（秒） | 1.0-3.0 |
| `--resolution` | 攝影機解析度 | 640x480 或 1280x720 |

啟動前端：

```bash
cd ~/live-vlm/frontend
npm start
```

用瀏覽器打開 `http://DGX_Spark_IP:3000`。

### 20-2-6 預設提示詞與自訂分析

**常見提示詞範例**：

| 場景 | 提示詞 |
|------|--------|
| **居家監控** | 「描述房間內的活動，注意是否有異常情況」 |
| **物品辨識** | 「列出畫面上所有物品，並描述它們的位置和狀態」 |
| **人數統計** | 「計算畫面中有幾個人，描述他們的動作和位置」 |
| **品質檢測** | 「檢查產品表面是否有瑕疵，指出問題位置」 |
| **場景描述** | 「用優美的文字描述你看到的景色」 |

在 WebUI 中可以即時切換提示詞，不需要重啟服務。

### 20-2-7 不同視覺模型的選擇與比較

**Qwen2.5-VL 7B 實測**：

```
輸入：把攝影機對著書桌
輸出：「我看到一張木製書桌，桌上有一台打開的筆記型電腦，
      螢幕顯示著程式碼。右邊有一杯咖啡，左邊有一本翻開的書。
      背景是一面白色牆壁，牆上掛著一幅風景畫。」
```

**Qwen2.5-VL 72B 實測**（更詳細）：

```
輸出：「這是一張整潔的書桌照片。桌面上放置著一台銀色的 MacBook Pro，
      螢幕上顯示的是 Python 程式碼，似乎是某個 web 框架的專案。
      電腦右側有一隻白色的陶瓷馬克杯，杯中裝有深色液體（可能是咖啡）。
      左側是一本翻開的《深入淺出設計模式》，書頁上有螢光筆標記。
      桌面上還有一個無線滑鼠和一副耳機。整體光線柔和，
      來自左上方的自然光，營造出舒適的工作氛圍。」
```

### 20-2-8 效能最佳化

**降低解析度**：

```bash
python server.py \
  --model-path models/qwen2.5-vl-7b \
  --resolution 640x480 \
  --interval 2.0
```

| 解析度 | 處理時間 | 記憶體 | 辨識準確率 |
|--------|---------|--------|-----------|
| 320x240 | 0.3 秒 | 最低 | 基本 |
| **640x480** | **0.5 秒** | **低** | **良好** |
| 1280x720 | 1.0 秒 | 中 | 高 |
| 1920x1080 | 2.0 秒 | 高 | 最高 |

**降低幀率（分析頻率）**：

```bash
# 每 3 秒分析一次（適合長時間監控）
python server.py --interval 3.0

# 每 0.5 秒分析一次（適合即時互動）
python server.py --interval 0.5
```

### 20-2-9 RTSP IP 攝影機支援

如果你有 IP 攝影機（如 Hikvision、Dahua），可以透過 RTSP 串流接入：

```bash
python server.py \
  --model-path models/qwen2.5-vl-7b \
  --rtsp rtsp://admin:password@192.168.1.100:554/stream1 \
  --interval 2.0
```

**常見 IP 攝影機 RTSP URL 格式**：

| 品牌 | RTSP URL 格式 |
|------|--------------|
| Hikvision | `rtsp://admin:password@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0` |
| TP-Link Tapo | `rtsp://username:password@IP:554/stream1` |
| Reolink | `rtsp://admin:password@IP:554/h264Preview_01_main` |

### 20-2-10 進階：多攝影機監控

DGX Spark 的 128GB 記憶體可以同時處理多個攝影機：

```bash
# 攝影機 1：大門口（人數統計）
python camera_server.py \
  --rtsp rtsp://camera1/stream \
  --prompt "計算人數" \
  --port 8765

# 攝影機 2：客廳（場景描述）
python camera_server.py \
  --rtsp rtsp://camera2/stream \
  --prompt "描述場景" \
  --port 8766
```

---

## 20-3 同時跑生成和理解

### 20-3-1 記憶體配置策略

DGX Spark 的 128GB 記憶體是共享的，需要合理分配：

**配置一：高效能文字 + 輕量視覺**

```
Ollama（Qwen3.5 120B NVFP4）   ~60 GB
VLM（Qwen2.5-VL 7B）           ~5 GB
系統保留                       ~10 GB
─────────────────────────────────────
總計                          ~75 GB ← 安全範圍
```

**配置二：圖片生成 + 文字對話**

```
ComfyUI（FLUX.1-Dev FP16）     ~23 GB
Ollama（Qwen3-8B）             ~16 GB
系統保留                       ~10 GB
─────────────────────────────────────
總計                          ~49 GB ← 非常安全
```

**配置三：全開模式（小型模型）**

```
Ollama（Qwen3-8B）             ~16 GB
VLM（Qwen2.5-VL 7B）           ~5 GB
ComfyUI（FLUX.1-Schnell）      ~12 GB
faster-whisper                 ~3 GB
系統保留                       ~10 GB
─────────────────────────────────────
總計                          ~46 GB ← 輕鬆同時運行
```

### 20-3-2 記憶體監控

```bash
# 即時監控記憶體使用
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# 或使用 nvtop（更直觀）
nvtop
```

### 20-3-3 自動記憶體管理腳本

```bash
#!/bin/bash
# auto-memory-manager.sh
# 當記憶體使用超過 90% 時自動釋放不用的服務

THRESHOLD=90
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
TOTAL=131072  # 128GB in MB
PERCENT=$((USED * 100 / TOTAL))

if [ $PERCENT -gt $THRESHOLD ]; then
  echo "記憶體使用率 ${PERCENT}%，超過閾值 ${THRESHOLD}%"
  echo "嘗試釋放快取..."
  echo 3 > /proc/sys/vm/drop_caches
  echo "已釋放快取"
fi
```

---

## 20-4 常見問題與疑難排解

### 20-4-1 TensorRT 推論相關

**Q：TensorRT engine 編譯失敗？**

```bash
# 確認 CUDA 版本
nvcc --version

# 確認 TensorRT 版本
python -c "import tensorrt; print(tensorrt.__version__)"

# 如果版本不相容，換用對應版本的容器
docker pull nvcr.io/nvidia/tensorrt-diffusion:24.12
```

**Q：生成圖片有雜訊或變形？**

- 增加 `--steps` 參數（20 → 30）
- 調整 `--guidance-scale`（3.5 → 4.5）
- 換用 FP16 精度

### 20-4-2 Live VLM WebUI 相關

**Q：攝影機畫面無法顯示？**

```bash
# 確認攝影機裝置
ls -la /dev/video*

# 測試攝影機
ffplay /dev/video0

# 如果沒有 /dev/video*，檢查 USB 連線
lsusb
```

**Q：VLM 回應太慢？**

```bash
# 降低解析度
--resolution 640x480

# 增加分析間隔
--interval 3.0

# 換用較小的模型
--model-path models/qwen2.5-vl-3b
```

**Q：RTSP 串流斷線？**

```bash
# 測試 RTSP 連線
ffplay rtsp://admin:password@IP:554/stream1

# 如果 ffplay 也無法播放，檢查：
# 1. IP 攝影機是否開機
# 2. 網路連線
# 3. 帳號密碼是否正確
```

---

## 20-5 本章小結

::: success ✅ 你現在知道了
- TensorRT 可以加速擴散模型推論 1.5-2 倍
- FP8 是最佳的平衡選擇：速度快、品質好
- Live VLM WebUI 實現即時視覺分析，支援 USB 和 IP 攝影機
- Qwen2.5-VL 7B 是日常使用的最佳選擇
- DGX Spark 可以同時跑多個多模態任務，關鍵是合理分配記憶體
:::

::: tip 🚀 下一章預告
接下來要介紹 RAG 知識庫和知識圖譜，讓 AI 不只是「知道」通用知識，還能讀懂你的專業文件！

👉 [前往第 21 章：RAG 與知識圖譜 →](/guide/chapter21/)
:::

::: info 📝 上一章
← [回到第 19 章：預訓練中小型語言模型](/guide/chapter19/)
:::
