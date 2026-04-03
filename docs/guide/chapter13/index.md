# 第 13 章：圖片與影片生成

::: tip 🎯 本章你將學到什麼
- 用 Open WebUI 體驗圖片 AI
- ComfyUI 安裝與工作流程
- Text-to-Image 和 Text-to-Video
- NVFP4 加速圖片生成
- 進階工作流程與自動化
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 13-1 用現有的工具體驗圖片 AI

在深入 ComfyUI 之前，我們先用已經熟悉的工具來體驗圖片 AI，這樣可以讓你快速建立對整個流程的理解。

### 13-1-1 用 Open WebUI + Qwen3.5 做圖片理解

如果你已經在第 5 章下載了 Qwen3.5 122B，它本身就支援圖片理解（VLM，Vision Language Model）。這意味著你可以上傳圖片，讓模型描述圖片內容、回答問題、甚至分析圖片中的文字。

在 Open WebUI 中的操作步驟：

1. 打開 Open WebUI 介面
2. 選擇 Qwen3.5 122B 模型
3. 點擊輸入框旁的 📎 圖示
4. 上傳一張圖片
5. 輸入提示詞，例如：「這張圖片中有什麼？」、「描述圖片中的場景」、「圖片中的文字是什麼？」

::: info 🤔 什麼是 VLM？
VLM（Vision Language Model）是一種同時理解圖片和文字的 AI 模型。它的工作原理是：

1. 圖片首先被轉換成「視覺 token」（類似文字 token 的數字表示）
2. 這些視覺 token 和文字 token 一起送入語言模型
3. 模型同時理解圖片和文字，然後生成回答

Qwen3.5 122B 就是這樣的模型，它不需要額外的圖片處理模組，本身就具備視覺理解能力。
:::

### 13-1-2 圖片理解 vs. 圖片生成

這是兩個完全不同的方向，初學者經常混淆：

| 特性 | 圖片理解 | 圖片生成 |
|------|---------|---------|
| 輸入 | 圖片 + 文字提示詞 | 純文字提示詞 |
| 輸出 | 文字描述 | 圖片 |
| 模型類型 | VLM（視覺語言模型） | 擴散模型（Diffusion Model） |
| 記憶體需求 | 中等（與純文字 LLM 相近） | 較高（需要處理高維度資料） |
| 代表模型 | Qwen3.5-VL、LLaVA、GPT-4V | FLUX、Stable Diffusion、DALL-E |
| 訓練方式 | 在文字 LLM 基礎上加入視覺編碼器 | 從噪聲逐步還原圖片的迭代過程 |

::: info 🤔 擴散模型（Diffusion Model）是什麼？
擴散模型是目前最主流的圖片生成技術。它的核心概念很直觀：

1. **前向過程**：把一張清晰的圖片逐步加入噪聲，直到變成完全隨機的雜訊
2. **反向過程**：訓練一個神經網路學會「去噪聲」，從隨機雜訊逐步還原出清晰的圖片
3. **生成新圖片**：從純隨機雜訊開始，讓模型一步步去噪聲，每次生成的結果都不同

這個過程就像把一杯清水慢慢滴入墨水變成黑水（前向），然後學會從黑水一步步變回清水（反向）。生成圖片時，我們從隨機雜訊開始，讓模型「想像」出一張新圖片。
:::

### 13-1-3 Open WebUI 的圖片生成功能

Open WebUI 不僅能理解圖片，還可以整合圖片生成 API，讓你在對話中直接要求 AI 生成圖片。

設定步驟：

1. 點擊左上角選單 → **Admin Panel**
2. 進入 **Settings → Images**
3. 在 **Image Generation** 區塊中設定：
   - **Engine**：選擇圖片生成引擎（如 Automatic1111、ComfyUI）
   - **API Endpoint**：填入圖片生成服務的 URL（如 `http://localhost:8188`）
   - **Model**：選擇要使用的模型
4. 點擊 **Save** 儲存設定

設定完成後，在對話中輸入「幫我畫一張⋯⋯」，Open WebUI 會自動呼叫圖片生成 API，並將結果顯示在對話中。

::: tip 💡 提示詞技巧
好的提示詞是生成好圖片的關鍵。建議格式：

```
[主體] + [場景/背景] + [風格] + [光線] + [細節]

例如：
「一隻橘色的貓，坐在窗台上，夕陽從窗外灑進來，水彩風格，柔和的光線，毛髮細節清晰」
```

避免使用模糊的描述，如「漂亮的風景」，改為具體的描述，如「瑞士阿爾卑斯山的湖泊，清晨的薄霧，倒影清晰」。
:::

### 13-1-4 使用自己開發的程式

除了使用圖形介面，你也可以用 Python 直接呼叫圖片生成模型。這在自動化流程或整合到自己的應用程式時非常有用。

```python
import torch
from diffusers import StableDiffusionPipeline

# 載入模型（首次執行會自動下載）
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 生成圖片
prompt = "a beautiful sunset over the ocean, photorealistic, 4k"
image = pipe(prompt).images[0]

# 儲存圖片
image.save("sunset.png")
print("圖片已儲存至 sunset.png")
```

這段程式碼會在 DGX Spark 上生成一張 512x512 的圖片。後面我們會介紹更強大的 FLUX 模型和 ComfyUI 工作流程。

---

## 13-2 ComfyUI 基礎

### 13-2-1 為什麼用 ComfyUI

市面上有多種圖片生成工具，各有優缺點：

| 工具 | 優點 | 缺點 | 適合對象 |
|------|------|------|---------|
| **ComfyUI** | 彈性最大、可視化流程、記憶體管理優秀 | 學習曲線較陡 | 進階使用者、開發者 |
| AUTOMATIC1111 | 社群最大、外掛最多、教學最多 | 記憶體用量高、速度較慢 | 初學者、一般使用者 |
| Fooocus | 最簡單、開箱即用 | 彈性有限、不支援複雜流程 | 完全新手 |
| Draw Things (macOS) | macOS 原生優化 | 僅限 Apple 平台 | Mac 使用者 |

在 DGX Spark 上，我們選擇 ComfyUI 的原因：

1. **記憶體管理出色**：ComfyUI 會在使用完每個節點後自動釋放記憶體，這在 128GB 統一記憶體的 DGX Spark 上非常重要
2. **支援最新模型**：FLUX、SD3 等新模型通常最先在 ComfyUI 上獲得支援
3. **可視化工作流程**：用「拉線」的方式建立流程，直觀且容易除錯
4. **可重複性**：每個工作流程都可以儲存為 JSON 檔，方便分享和重現
5. **支援圖片 + 影片**：同一個介面可以處理圖片和影片生成

::: info 🤔 什麼是節點式工作流程？
想像你在廚房做菜：

- **傳統方式**（如 AUTOMATIC1111）：你告訴廚師「做一道義大利麵」，廚師自己決定所有步驟
- **節點式方式**（如 ComfyUI）：你明確指定每個步驟：煮麵 → 調醬 → 混合 → 裝盤，每個步驟都是一個「節點」，用「線」連接

節點式的好處是你可以看到每個步驟的輸出，隨時調整任何環節。例如你可以只替換「調醬」這一步，其他步驟保持不變。
:::

### 13-2-2 DGX Spark 上可用的生成模型

| 模型 | 記憶體需求 | 品質 | 速度 | 解析度 | 說明 |
|------|-----------|------|------|--------|------|
| **FLUX.2 Klein** | ~12 GB | 高 | 快 | 1024x1024 | FLUX 2.0 輕量版，性價比最高 |
| **FLUX.1-dev** | ~23 GB | 最高 | 中等 | 1024x1024 | 開發版，品質最佳但需要較多記憶體 |
| **FLUX.1-Schnell** | ~12 GB | 高 | 最快 | 1024x1024 | 快速版，僅需 4 步即可生成 |
| **SDXL** | ~7 GB | 中等 | 快 | 1024x1024 | 老牌模型，社群資源最豐富 |
| **SD 1.5** | ~4 GB | 中等偏下 | 超快 | 512x512 | 最輕量，適合測試 |

::: tip 💡 模型選擇建議
- **初次體驗**：從 FLUX.1-Schnell 開始，速度快且品質好
- **追求品質**：使用 FLUX.1-dev，但確保沒有其他大型程式在執行
- **日常使用**：FLUX.2 Klein 是最佳平衡點
- **記憶體緊張時**：退回 SDXL
:::

### 13-2-3 ComfyUI 的核心概念

ComfyUI 的工作流程由以下元素組成：

| 元素 | 說明 | 範例 |
|------|------|------|
| **節點（Node）** | 執行特定功能的模組 | 載入模型、編碼提示詞、取樣器 |
| **連接線（Wire）** | 連接節點的資料流 | 將提示詞傳給取樣器 |
| **輸入節點** | 工作流程的起點 | 文字提示詞、模型載入 |
| **輸出節點** | 工作流程的終點 | 儲存圖片、預覽圖片 |
| **參數** | 每個節點的可調設定 | 步數、CFG 值、解析度 |

一個最基本的 Text-to-Image 工作流程包含：

```
Checkpoint Loader（載入模型）
    ↓
CLIP Text Encode (Prompt)（編碼正向提示詞）
CLIP Text Encode (Negative Prompt)（編碼負向提示詞）
    ↓
Empty Latent Image（建立空白畫布）
    ↓
KSampler（取樣器，核心生成步驟）
    ↓
VAE Decode（將潛空間轉換為圖片）
    ↓
Save Image（儲存圖片）
```

---

## 13-3 安裝 ComfyUI

### 13-3-1 建立環境並安裝

告訴 Claude Code：

> 「幫我安裝 ComfyUI，使用 uv 建立 Python 環境，安裝所有相依套件。」

Claude Code 會執行以下步驟：

```bash
# 複製 ComfyUI 原始碼
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 建立獨立的 Python 虛擬環境
uv venv .venv
source .venv/bin/activate

# 安裝 PyTorch（CUDA 13.0 版本，支援 NVFP4）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安裝 ComfyUI 的相依套件
uv pip install -r requirements.txt
```

::: info 🤔 為什麼要用 uv？
`uv` 是一個極快的 Python 套件管理器，比傳統的 `pip` 快 10-100 倍。在 DGX Spark 上，使用 `uv` 可以大幅減少安裝時間。

`uv venv` 建立的是獨立的 Python 環境，不會影響系統的其他 Python 套件。這很重要，因為不同的 AI 工具可能需要不同版本的套件。
:::

### 13-3-2 下載模型

ComfyUI 本身不包含模型，你需要自行下載。模型檔案通常放在 `models/checkpoints/` 目錄下。

```bash
# 建立模型目錄結構
mkdir -p models/checkpoints
mkdir -p models/vae
mkdir -p models/loras
mkdir -p models/controlnet

# 下載 FLUX.1-Schnell（較小，適合先測試）
cd models/checkpoints
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors

# 回到 ComfyUI 根目錄
cd ~/ComfyUI
```

::: warning ⚠️ 模型下載注意事項
- FLUX.1-Schnell 約 23 GB，下載需要一些時間
- 如果下載中斷，可以重新執行 `wget`，它會自動續傳
- 確保你的磁碟空間足夠（建議至少 50 GB 可用空間）
- `.safetensors` 是安全的模型格式，不會執行惡意程式碼
:::

### 13-3-3 啟動 ComfyUI

```bash
cd ~/ComfyUI
source .venv/bin/activate

python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --force-fp16
```

各參數說明：

| 參數 | 說明 | 何時使用 |
|------|------|---------|
| `--listen 0.0.0.0` | 允許從其他裝置連線 | 需要從其他電腦訪問時 |
| `--port 8188` | 指定連接埠 | 預設值，可改為其他埠 |
| `--force-fp16` | 強制使用半精度浮點數 | 節省記憶體，但某些模型可能不相容 |
| `--lowvram` | 極低記憶體模式 | 記憶體不足時使用 |
| `--normalvram` | 正常記憶體模式 | DGX Spark 通常不需要 |

啟動後，用瀏覽器打開 `http://DGX_Spark_IP:8188`。

::: tip 💡 在背景執行 ComfyUI
如果你希望關閉終端機後 ComfyUI 繼續執行：

```bash
# 使用 nohup 在背景執行
nohup python main.py --listen 0.0.0.0 --port 8188 --force-fp16 > comfyui.log 2>&1 &

# 查看日誌
tail -f comfyui.log

# 停止 ComfyUI
pkill -f "python main.py"
```
:::

### 13-3-4 請 Claude Code 幫忙生圖

你也可以直接用 Claude Code 產生 ComfyUI 的工作流程 JSON：

> 「幫我產生一個 ComfyUI 工作流程，使用 FLUX.1-Schnell 模型，生成一張『夕陽下的海灘』圖片，解析度 1024x1024。」

Claude Code 會產生一個 JSON 檔，你可以直接拖入 ComfyUI 中使用。

---

## 13-4 Text-to-Image 工作流程

### 13-4-1 使用預設工作流程

打開 ComfyUI 後，預設就有一個 Text-to-Image 工作流程。讓我們逐一了解每個節點：

**1. Load Checkpoint（載入模型）**
- 選擇你下載的模型（如 FLUX.1-Schnell）
- 這會載入模型權重到記憶體中

**2. CLIP Text Encode (Prompt)（正向提示詞編碼）**
- 輸入你想要生成的圖片描述
- 例如：「a beautiful sunset over the ocean, golden hour, photorealistic」

**3. CLIP Text Encode (Negative Prompt)（負向提示詞編碼）**
- 輸入你不想要出現的內容
- 例如：「blurry, low quality, watermark, text」
- 注意：FLUX 模型通常不需要負向提示詞

**4. Empty Latent Image（空白潛空間圖片）**
- 設定生成圖片的解析度
- width: 1024, height: 1024（FLUX 建議使用 1024x1024）
- batch_size: 1（一次生成幾張）

**5. KSampler（取樣器）**
- 這是整個工作流程的核心
- 重要參數：
  - `seed`：隨機種子，固定值可以重現相同的圖片
  - `steps`：生成步數，越多品質越好但越慢（FLUX-Schnell 只需 4 步）
  - `cfg`：提示詞遵循度，通常 3.5-7
  - `sampler_name`：取樣演算法（推薦 `euler` 或 `dpmpp_2m`）
  - `scheduler`：排程器（推薦 `normal` 或 `karras`）

**6. VAE Decode（VAE 解碼）**
- 將潛空間表示轉換為實際的圖片像素

**7. Save Image（儲存圖片）**
- 將生成的圖片儲存到 `output/` 目錄

在提示詞框中輸入描述，點擊 **Queue Prompt** 即可開始生成。

### 13-4-2 使用 FLUX.2 Klein 工作流程

FLUX.2 Klein 是 FLUX 系列的最新版本，品質和速度都有提升。

1. 下載 FLUX.2 Klein 模型：
```bash
cd ~/ComfyUI/models/checkpoints
huggingface-cli download black-forest-labs/FLUX.2-Klein \
  --local-dir .
```

2. 拖入 FLUX.2 Klein 的工作流程 JSON 檔（可從社群或 Claude Code 取得）

3. 在 **Load Checkpoint** 節點中選擇 FLUX.2 Klein 模型

4. 修改提示詞

5. 點擊 Queue Prompt

::: tip 💡 FLUX 提示詞技巧
FLUX 模型對自然語言的理解能力很強，建議：
- 使用完整的句子而非關鍵字堆砌
- 描述具體的場景、光線、風格
- 可以指定攝影機角度、鏡頭類型
- 例如：「A close-up portrait of an elderly craftsman working on a wooden sculpture, warm workshop lighting, shot on 85mm lens, shallow depth of field」
:::

### 13-4-3 安裝 ComfyUI Manager

ComfyUI Manager 是一個必備的外掛管理器，讓你輕鬆安裝其他外掛、模型和工作流程。

```bash
# 進入自訂節點目錄
cd ~/ComfyUI/custom_nodes

# 下載 ComfyUI Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 安裝相依套件
cd ComfyUI-Manager
uv pip install -r requirements.txt

# 回到 ComfyUI 根目錄並重新啟動
cd ~/ComfyUI
pkill -f "python main.py" 2>/dev/null
source .venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188 --force-fp16
```

重啟後，側邊欄會多出 **Manager** 按鈕。透過 Manager 你可以：

- 瀏覽和安裝社群工作流程
- 一鍵安裝缺少的自訂節點
- 更新所有已安裝的外掛
- 下載模型（整合 HuggingFace 和 Civitai）

### 13-4-4 進階：ControlNet 精確控制

ControlNet 讓你可以用額外的條件來控制圖片生成，例如：

| ControlNet 類型 | 輸入 | 效果 |
|----------------|------|------|
| Canny | 邊緣線稿 | 保持圖片的輪廓結構 |
| Depth | 深度圖 | 保持場景的遠近關係 |
| OpenPose | 姿勢骨架 | 控制人物的姿勢 |
| Scribble | 簡筆畫 | 用草圖控制構圖 |

安裝 ControlNet 節點：

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
uv pip install -r requirements.txt
```

---

## 13-5 Text-to-Video：文字生影片

### 13-5-1 影片生成模型比較

| 模型 | 記憶體需求 | 生成時長 | 幀數 | 品質 | 速度 |
|------|-----------|---------|------|------|------|
| **CogVideoX-5B** | ~20 GB | 2-6 秒 | 16-49 幀 | 高 | 中等 |
| CogVideoX-2B | ~12 GB | 2-6 秒 | 16-49 幀 | 中高 | 快 |
| AnimateDiff | ~8 GB | 2-4 秒 | 16 幀 | 中 | 快 |
| ModelScope | ~10 GB | 2-4 秒 | 24 幀 | 中 | 中等 |

### 13-5-2 安裝影片節點和模型

```bash
# 安裝 CogVideoX 節點
cd ~/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-CogVideoX-wrapper.git

# 安裝相依套件
cd ComfyUI-CogVideoX-wrapper
uv pip install -r requirements.txt

# 下載 CogVideoX-5B 模型
cd ~/ComfyUI/models/diffusion_models
huggingface-cli download THUDM/CogVideoX-5b \
  --local-dir CogVideoX-5b
```

### 13-5-3 用 Claude Code 產生影片工作流程

> 「幫我產生一個 ComfyUI 影片生成工作流程，使用 CogVideoX-5B 模型，生成 5 秒的『海浪拍打岩石』影片，49 幀。」

Claude Code 會產生一個完整的 JSON 工作流程，包含：

1. **Load Diffusion Model**：載入 CogVideoX 模型
2. **CLIP Text Encode**：編碼提示詞
3. **Empty CogVideo Latent**：設定影片參數（幀數、解析度）
4. **CogVideoX Sampler**：影片取樣器
5. **VAE Decode**：解碼為影片幀
6. **Video Combine**：合併為影片檔
7. **Save/Preview Video**：儲存或預覽

### 13-5-4 在 ComfyUI 中生成影片

1. 載入影片生成工作流程
2. 在提示詞框中輸入影片描述
3. 設定參數：
   - `num_frames`：建議 16-49 幀（越多越流暢但越慢）
   - `guidance_scale`：7-9（提示詞遵循度）
   - `num_inference_steps`：25-50（生成步數）
4. 點擊 Queue Prompt

::: tip 💡 影片生成很耗記憶體
在 DGX Spark 上，CogVideoX-5B 約需要 20GB 記憶體。確保：
- 沒有其他大型模型同時運行
- 關閉不必要的應用程式
- 如果記憶體不足，嘗試使用 CogVideoX-2B（~12GB）
:::

### 13-5-5 RTX Video 4K 升頻

生成低解析度影片後，可以用 AI 升頻到 4K：

```bash
# 安裝 VideoHelperSuite（包含升頻功能）
cd ~/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-VideoHelperSuite.git

# 安裝相依套件
cd ComfyUI-VideoHelperSuite
uv pip install -r requirements.txt

# 下載 Real-ESRGAN 升頻模型
mkdir -p ~/ComfyUI/models/upscale_models
cd ~/ComfyUI/models/upscale_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

升頻工作流程：
```
Load Video → Upscale Image (Real-ESRGAN) → Combine Video → Save Video
```

| 升頻模型 | 倍率 | 記憶體 | 速度 | 適用場景 |
|---------|------|--------|------|---------|
| RealESRGAN_x4plus | 4x | ~4 GB | 中等 | 通用 |
| RealESRGAN_x2plus | 2x | ~2 GB | 快 | 輕微升頻 |
| RealESRGAN_anime | 4x | ~4 GB | 中等 | 動漫風格 |

---

## 13-6 NVFP4 加速

### 13-6-1 什麼是 NVFP4？

NVFP4（NVIDIA FP4）是 NVIDIA Blackwell 架構專用的 4-bit 浮點數格式。相比傳統的 FP16（16-bit），NVFP4 將每個數值壓縮到只有 4-bit，大幅減少記憶體用量並提升計算速度。

### 13-6-2 效能提升數據

| 格式 | 位元數 | 生成速度（步/秒） | 記憶體用量 | 品質損失 |
|------|--------|------------------|-----------|---------|
| FP16 | 16-bit | 1.0x（基準） | 100% | 無 |
| FP8 | 8-bit | 1.3x | 50% | 極小 |
| **NVFP4** | **4-bit** | **1.8x** | **25%** | **可忽略** |

::: info 🤔 為什麼 NVFP4 品質損失可忽略？
NVFP4 使用動態縮放技術，在計算時會根據數值的範圍自動調整精度。對於圖片生成這種任務，人眼對微小的數值差異不敏感，因此 4-bit 的精度損失幾乎看不出來。
:::

### 13-6-3 前提：PyTorch CUDA 13.0

NVFP4 需要 PyTorch 搭配 CUDA 13.0：

```bash
# 確保使用正確的 PyTorch 版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 驗證
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
# 應該輸出 13.0 或更高
```

### 13-6-4 下載 NVFP4 模型

```bash
# 下載 FLUX.1-dev NVFP4 版本
cd ~/ComfyUI/models/checkpoints
huggingface-cli download black-forest-labs/FLUX.1-dev-NVFP4 \
  --local-dir .
```

### 13-6-5 在 ComfyUI 中使用 NVFP4 模型

1. 在 ComfyUI 的 **Load Checkpoint** 節點中選擇 NVFP4 模型
2. 確認啟動時 `--force-fp16` 參數已移除（NVFP4 不需要，反而可能影響效能）
3. 正常生成圖片

正確的啟動命令：
```bash
python main.py --listen 0.0.0.0 --port 8188
```

::: tip 💡 NVFP4 使用建議
- 只在 Blackwell 架構上使用（DGX Spark 支援）
- 推論時使用，微調時使用 NF4（見第 15 章）
- 如果生成結果異常，檢查是否錯誤使用了 `--force-fp16`
:::

---

## 13-7 自動化與批次處理

### 13-7-1 使用 API 批次生成

ComfyUI 提供 API 介面，可以用程式批次生成圖片：

```python
import json
import urllib.request
import urllib.parse
import random

# ComfyUI 伺服器位址
server_address = "127.0.0.1:8188"

def queue_prompt(prompt):
    """送出工作流程到 ComfyUI"""
    p = {"prompt": prompt, "client_id": "my-client"}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(
        f"http://{server_address}/prompt", data=data
    )
    return json.loads(urllib.request.urlopen(req).read())

def generate_image(prompt_text, seed=None, filename="output.png"):
    """生成單張圖片"""
    if seed is None:
        seed = random.randint(0, 2**32)
    
    # 定義工作流程（簡化版）
    workflow = {
        "3": {  # KSampler
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "6": {  # 正向提示詞
            "inputs": {"text": prompt_text, "clip": ["4", 0]},
            "class_type": "CLIPTextEncode"
        },
        "7": {  # 負向提示詞
            "inputs": {"text": "blurry, low quality", "clip": ["4", 0]},
            "class_type": "CLIPTextEncode"
        },
        "5": {  # 空白潛空間
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage"
        }
    }
    
    queue_prompt(workflow)
    print(f"已送出任務，seed: {seed}")

# 批次生成
prompts = [
    "a futuristic city at night, neon lights, cyberpunk style",
    "a peaceful Japanese garden with cherry blossoms",
    "an astronaut floating in space, earth in background"
]

for i, prompt in enumerate(prompts):
    generate_image(prompt, filename=f"output_{i}.png")
```

### 13-7-2 從圖片生成提示詞

你也可以用 Qwen3.5-VL 自動描述圖片，然後用描述來生成新圖片：

```python
# 1. 用 VLM 描述圖片
image_description = "一張夕陽下的海灘照片，金色的陽光灑在海面上"

# 2. 用描述生成新圖片
generate_image(
    f"{image_description}, digital art, vibrant colors, 4k resolution"
)
```

---

## 13-8 清理

### 13-8-1 移除 ComfyUI 和相關環境

```bash
# 停止 ComfyUI（如果在背景執行）
pkill -f "python main.py"

# 移除整個 ComfyUI 目錄（包含模型，請確認不再需要）
rm -rf ~/ComfyUI

# 清理 Python 快取
rm -rf ~/.cache/uv
rm -rf ~/.cache/huggingface
```

::: warning ⚠️ 注意
`rm -rf ~/ComfyUI` 會刪除所有下載的模型（可能數十 GB）。如果只想清理環境但保留模型：

```bash
# 只移除 Python 環境
rm -rf ~/ComfyUI/.venv

# 保留模型，移除其他檔案
rm -rf ~/ComfyUI/custom_nodes
rm -rf ~/ComfyUI/output
```
:::

---

## 13-9 疑難排解

### 13-9-1 常見問題 FAQ

**Q1：ComfyUI 啟動後無法連線？**

```bash
# 確認 port 8188 沒被佔用
lsof -i :8188

# 如果有程式佔用，停止它或換一個 port
python main.py --port 8189

# 檢查防火牆設定
sudo ufw allow 8188/tcp
```

**Q2：生成圖片時記憶體不足（OOM）？**

```bash
# 方法 1：降低解析度
# 在 Empty Latent Image 節點中，改為 512x512

# 方法 2：改用較小的模型
# FLUX.1-Schnell（~12GB）取代 FLUX.1-dev（~23GB）

# 方法 3：使用 --lowvram 模式
python main.py --listen 0.0.0.0 --port 8188 --lowvram

# 方法 4：檢查是否有其他程式佔用記憶體
nvidia-smi
```

**Q3：模型下載失敗或速度很慢？**

```bash
# 使用 huggingface-cli 的鏡像站點
export HF_ENDPOINT=https://hf-mirror.com

# 重新下載
huggingface-cli download black-forest-labs/FLUX.1-schnell \
  --local-dir ~/ComfyUI/models/checkpoints
```

**Q4：生成的圖片是全黑或全白的？**

- 檢查 VAE 是否正確載入
- 確認模型的 dtype 設定（FLUX 建議用 bf16 或 fp16）
- 嘗試在啟動時加上 `--force-fp16` 或移除它

**Q5：ComfyUI Manager 無法安裝外掛？**

```bash
# 手動安裝
cd ~/ComfyUI/custom_nodes
git clone <外掛的 GitHub 網址>
cd <外掛目錄>
uv pip install -r requirements.txt

# 重新啟動 ComfyUI
```

**Q6：影片生成時幀數不穩定或閃爍？**

- 增加 `num_inference_steps`（25 → 50）
- 提高 `guidance_scale`（7 → 8.5）
- 使用較大的模型（CogVideoX-5B 取代 2B）
- 固定 `seed` 值以獲得一致的結果

---

## 13-10 本章小結

::: success ✅ 你現在知道了
- ComfyUI 是節點式的圖片生成工具，彈性最大且記憶體管理優秀
- FLUX 系列是目前最好的開源圖片生成模型，Schnell 最快、dev 品質最高
- Text-to-Video 可以用 CogVideoX 實現，支援 2-6 秒的影片生成
- NVFP4 可以加速圖片生成 1.8 倍並減少 75% 記憶體用量
- ComfyUI 支援 API 批次處理，適合自動化工作流程
:::

::: tip 🚀 下一章預告
圖片搞定了，接下來來處理聲音！語音合成、語音辨識、AI 音樂生成，全部在 DGX Spark 上完成！

👉 [前往第 14 章：音訊、語音與音樂 AI →](/guide/chapter14/)
:::

::: info 📝 上一章
← [回到第 12 章：NIM 與引擎比較](/guide/chapter12/)
:::
