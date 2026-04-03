# 第 20 章：多模態推論與即時視覺 AI

::: tip 🎯 本章你將學到什麼
- TensorRT 加速擴散模型推論
- Live VLM WebUI 即時串流
- 同時跑生成和理解
:::

---

## 20-1 TensorRT 加速擴散模型推論

### 20-1-1 部署 TensorRT 推論環境

```bash
docker run -d \
  --name trt-diffusion \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/trt-models:/models \
  -v ~/trt-output:/output \
  nvcr.io/nvidia/tensorrt-diffusion:25.01
```

### 20-1-2 Flux.1-Dev 多精度推論

```bash
# FP16
python generate.py --model flux-dev --precision fp16 --prompt "夕陽下的海灘"

# FP8
python generate.py --model flux-dev --precision fp8 --prompt "夕陽下的海灘"

# NF4
python generate.py --model flux-dev --precision nf4 --prompt "夕陽下的海灘"
```

### 20-1-3 三種精度的效能比較

| 精度 | 生成速度 | 記憶體 | 品質 |
|------|---------|--------|------|
| FP16 | 1.0x | 100% | 最高 |
| FP8 | 1.4x | 50% | 高 |
| NF4 | 1.7x | 25% | 中高 |

### 20-1-4 Flux.1-Schnell 快速推論

```bash
# Schnell 只需要 4 步
python generate.py --model flux-schnell --steps 4 --prompt "快速生圖"
```

### 20-1-5 SDXL 推論

```bash
python generate.py --model sdxl --precision fp16 --prompt "SDXL 測試"
```

### 20-1-6 驗證與清理

```bash
docker stop trt-diffusion
docker rm trt-diffusion
```

---

## 20-2 Live VLM WebUI — 即時視覺語言模型

### 20-2-1 安裝 VLM 後端

```bash
git clone https://github.com/community/live-vlm.git
cd live-vlm
uv pip install -r requirements.txt
```

### 20-2-2 安裝 Live VLM WebUI

```bash
cd frontend
npm install
npm run build
```

### 20-2-3 WebRTC 即時串流

啟動服務：

```bash
python server.py --model qwen3.5-122b --port 8765
```

用瀏覽器打開 `http://DGX_Spark_IP:8765`，允許攝影機存取權限。

你會在網頁中看到：
- 即時攝影機畫面
- VLM 的即時分析文字

### 20-2-4 預設提示詞與自訂分析

在設定中可以修改提示詞：

```
預設：「描述你看到的畫面」
自訂：「計算畫面中有幾個人」
自訂：「偵測是否有異常行為」
```

### 20-2-5 不同視覺模型的選擇

| 模型 | 記憶體 | 速度 | 準確率 |
|------|--------|------|--------|
| Qwen3.5-122B | ~61 GB | 中等 | 最高 |
| Llama 3.2 11B | ~8 GB | 快 | 高 |
| LLaVA 7B | ~5 GB | 快 | 中 |

### 20-2-6 效能最佳化

```bash
# 降低解析度
python server.py --model qwen3.5-122b --resolution 640x480

# 降低幀率
python server.py --model qwen3.5-122b --fps 2
```

### 20-2-7 RTSP IP 攝影機支援

```bash
python server.py --model qwen3.5-122b --rtsp rtsp://camera-ip/stream
```

---

## 20-3 同時跑生成和理解

DGX Spark 的 128GB 記憶體讓你可以同時執行：

```
圖片生成（ComfyUI）    ~25 GB
文字生成（Ollama）     ~60 GB
視覺理解（VLM）       ~61 GB
─────────────────────────────
總計：~146 GB → 需要錯峰使用
```

**建議配置**：

```
白天：文字生成（Ollama 120B）+ 視覺理解（VLM 11B）
晚上：圖片生成（ComfyUI FLUX）
```

或者用較小的模型同時運行：

```
文字生成（Ollama 8B）    ~16 GB
視覺理解（Llama 3.2 11B） ~8 GB
圖片生成（FLUX Schnell） ~12 GB
─────────────────────────────
總計：~36 GB → 輕鬆同時運行！
```

---

## 20-4 常見問題與疑難排解

### 20-4-1 TensorRT 推論相關

**Q：TensorRT engine 編譯失敗？**

確認模型和 TensorRT 版本相容。

### 20-4-2 Live VLM WebUI 相關

**Q：攝影機畫面無法顯示？**

```bash
# 確認攝影機正常運作
ls /dev/video*

# 測試
ffplay /dev/video0
```

---

## 20-5 本章小結

::: success ✅ 你現在知道了
- TensorRT 可以加速擴散模型推論
- Live VLM WebUI 實現即時視覺分析
- DGX Spark 可以同時跑多個多模態任務（用較小模型）
:::

::: tip 🚀 下一章預告
接下來要介紹 RAG 知識庫和知識圖譜，讓 AI 不只是「知道」通用知識，還能讀懂你的專業文件！

👉 [前往第 21 章：RAG 與知識圖譜 →](/guide/chapter21/)
:::

::: info 📝 上一章
← [回到第 19 章：預訓練中小型語言模型](/guide/chapter19/)
:::
