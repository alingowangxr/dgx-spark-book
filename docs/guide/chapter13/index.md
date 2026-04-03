# 第 13 章：圖片與影片生成

::: tip 🎯 本章你將學到什麼
- 用 Open WebUI 體驗圖片 AI
- ComfyUI 安裝與工作流程
- Text-to-Image 和 Text-to-Video
- NVFP4 加速圖片生成
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 13-1 用現有的工具體驗圖片 AI

### 13-1-1 用 Open WebUI + Qwen3.5 做圖片理解

如果你已經在第 5 章下載了 Qwen3.5 122B，它本身就支援圖片理解：

在 Open WebUI 中：
1. 選擇 Qwen3.5 122B 模型
2. 點擊輸入框旁的 📎 圖示
3. 上傳一張圖片
4. 問：「這張圖片中有什麼？」

### 13-1-2 圖片理解 vs. 圖片生成

| | 圖片理解 | 圖片生成 |
|--|---------|---------|
| 輸入 | 圖片 + 文字 | 文字 |
| 輸出 | 文字描述 | 圖片 |
| 模型類型 | VLM（視覺語言模型） | 擴散模型（Diffusion） |
| 記憶體需求 | 中等 | 較高 |

### 13-1-3 Open WebUI 的圖片生成功能

Open WebUI 可以整合圖片生成 API（如 Stable Diffusion）：

1. **Admin Panel → Settings → Images**
2. 設定圖片生成端點
3. 在對話中，AI 會自動呼叫圖片生成 API

### 13-1-4 使用自己開發的程式

你也可以用 Python 直接呼叫圖片生成模型，這在後面會詳細介紹。

---

## 13-2 ComfyUI 基礎

### 13-2-1 為什麼用 ComfyUI

ComfyUI 是一個節點式的圖片生成介面，讓你用「拉線」的方式建立圖片生成工作流程。

| 工具 | 優點 | 缺點 |
|------|------|------|
| **ComfyUI** | 彈性最大、可視化流程 | 學習曲線較陡 |
| AUTOMATIC1111 | 社群最大、外掛最多 | 記憶體用量高 |
| Fooocus | 最簡單 | 彈性有限 |

在 DGX Spark 上，我們選擇 ComfyUI 因為：
- 記憶體管理好
- 支援最新的模型
- 可以建立複雜的工作流程（圖片 + 影片）

### 13-2-2 DGX Spark 上可用的生成模型

| 模型 | 記憶體需求 | 品質 | 速度 |
|------|-----------|------|------|
| FLUX.2 Klein | ~12 GB | 高 | 快 |
| FLUX.1-dev | ~23 GB | 最高 | 中等 |
| FLUX.1-Schnell | ~12 GB | 高 | 最快 |
| SDXL | ~7 GB | 中等 | 快 |

---

## 13-3 安裝 ComfyUI

### 13-3-1 建立環境並安裝

告訴 Claude Code：

> 「幫我安裝 ComfyUI，使用 uv 建立 Python 環境，安裝所有相依套件。」

Claude Code 會執行：

```bash
# 複製 ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 建立 Python 環境
uv venv .venv
source .venv/bin/activate

# 安裝相依套件
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

### 13-3-2 下載模型

```bash
# 建立模型目錄
mkdir -p models/checkpoints

# 下載 FLUX.1-Schnell（較小，適合先測試）
cd models/checkpoints
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors
```

### 13-3-3 啟動 ComfyUI

```bash
cd ~/ComfyUI
source .venv/bin/activate

python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --force-fp16
```

然後用瀏覽器打開 `http://DGX_Spark_IP:8188`。

### 13-3-4 請 Claude Code 幫忙生圖

你也可以直接用 Claude Code 產生 ComfyUI 的工作流程 JSON：

> 「幫我產生一個 ComfyUI 工作流程，使用 FLUX.1-Schnell 模型，生成一張『夕陽下的海灘』圖片，解析度 1024x1024。」

---

## 13-4 Text-to-Image 工作流程

### 13-4-1 使用預設工作流程

打開 ComfyUI 後，預設就有一個 Text-to-Image 工作流程：

```
CLIP Text Encode (Prompt) → KSampler → VAE Decode → Save Image
         ↑
  CLIP Text Encode (Negative Prompt)
```

在提示詞框中輸入描述，點擊 **Queue Prompt** 即可生成。

### 13-4-2 使用 FLUX.2 Klein 工作流程

1. 下載 FLUX.2 Klein 模型
2. 拖入 FLUX.2 Klein 的工作流程 JSON 檔
3. 修改提示詞
4. 點擊 Queue Prompt

### 13-4-3 安裝 ComfyUI Manager

ComfyUI Manager 是一個外掛管理器，讓你輕鬆安裝其他外掛和工作流程。

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 安裝相依套件
cd ComfyUI-Manager
uv pip install -r requirements.txt

# 重新啟動 ComfyUI
```

重啟後，介面中會多出 **Manager** 按鈕。

---

## 13-5 Text-to-Video：文字生影片

### 13-5-1 安裝影片節點和模型

```bash
# 安裝影片生成節點
cd ~/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-CogVideoX-wrapper.git

# 安裝相依套件
cd ComfyUI-CogVideoX-wrapper
uv pip install -r requirements.txt
```

### 13-5-2 用 Claude Code 產生影片工作流程

> 「幫我產生一個 ComfyUI 影片生成工作流程，使用 CogVideoX 模型，生成 5 秒的『海浪拍打岩石』影片。」

### 13-5-3 在 ComfyUI 中生成影片

1. 載入影片生成工作流程
2. 輸入提示詞
3. 設定幀數（建議 16-49 幀）
4. 點擊 Queue Prompt

::: tip 💡 影片生成很耗記憶體
在 DGX Spark 上，CogVideoX 約需要 20GB 記憶體。確保沒有其他大型模型同時運行。
:::

### 13-5-4 RTX Video 4K 升頻

生成低解析度影片後，可以用 AI 升頻到 4K：

```bash
# 安裝 Real-ESRGAN
cd ~/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-VideoHelperSuite.git

# 下載 Real-ESRGAN 模型
cd ~/ComfyUI/models/upscale_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

---

## 13-6 NVFP4 加速

### 13-6-1 效能提升數據

| 格式 | 生成速度（步/秒） | 記憶體用量 |
|------|------------------|-----------|
| FP16 | 1.0x（基準） | 100% |
| FP8 | 1.3x | 50% |
| **NVFP4** | **1.8x** | **25%** |

### 13-6-2 前提：PyTorch CUDA 13.0

NVFP4 需要 PyTorch 搭配 CUDA 13.0：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 13-6-3 下載 NVFP4 模型

```bash
# 下載 FLUX.1-dev NVFP4 版本
cd ~/ComfyUI/models/checkpoints
huggingface-cli download black-forest-labs/FLUX.1-dev-NVFP4 \
  --local-dir .
```

### 13-6-4 在 ComfyUI 中使用 NVFP4 模型

1. 在 ComfyUI 中選擇 NVFP4 模型
2. 確認 `--force-fp16` 參數已移除（NVFP4 不需要）
3. 正常生成圖片

---

## 13-7 清理

### 13-7-1 移除 ComfyUI 和相關環境

```bash
# 停止 ComfyUI（如果在背景執行）
pkill -f "python main.py"

# 移除整個目錄
rm -rf ~/ComfyUI

# 清理 Python 環境
rm -rf ~/.cache/uv
```

---

## 13-8 疑難排解

### 13-8-1 常見問題

**Q：ComfyUI 啟動後無法連線？**

```bash
# 確認 port 8188 沒被佔用
lsof -i :8188

# 換一個 port
python main.py --port 8189
```

**Q：生成圖片時記憶體不足？**

```bash
# 降低解析度
# 在 ComfyUI 的 Empty Latent Image 節點中，改為 512x512

# 或改用較小的模型（FLUX.1-Schnell）
```

---

## 13-9 本章小結

::: success ✅ 你現在知道了
- ComfyUI 是節點式的圖片生成工具，彈性最大
- FLUX 系列是目前最好的開源圖片生成模型
- Text-to-Video 可以用 CogVideoX 實現
- NVFP4 可以加速圖片生成並減少記憶體用量
:::

::: tip 🚀 下一章預告
圖片搞定了，接下來來處理聲音！語音合成、語音辨識、AI 音樂生成，全部在 DGX Spark 上完成！

👉 [前往第 14 章：音訊、語音與音樂 AI →](/guide/chapter14/)
:::

::: info 📝 上一章
← [回到第 12 章：NIM 與引擎比較](/guide/chapter12/)
:::
