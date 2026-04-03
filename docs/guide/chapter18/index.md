# 第 18 章：影像模型微調 — FLUX Dreambooth LoRA

::: tip 🎯 本章你將學到什麼
- Dreambooth 與 LoRA 的核心概念與原理
- 多概念微調的完整流程
- 環境建立與模型下載的詳細步驟
- 準備高品質訓練資料（圖片 + 設定檔）
- Dreambooth LoRA 訓練的完整參數說明
- 微調前後效果比較與評估方法
- 進階技巧與常見問題完整排解
:::

---

## 18-1 影像生成模型微調概念

### 18-1-1 什麼是 Dreambooth？

**Dreambooth** 是 Google Research 在 2022 年提出的一種微調技術。它的核心思想是：把一個「特定概念」注入到已經訓練好的影像生成模型中。

::: info 🤔 為什麼需要 Dreambooth？
一般的影像生成模型（如 FLUX、Stable Diffusion）已經學會了「貓」、「狗」、「汽車」等通用概念。但如果你想要生成：
- **你家那隻叫「咪咪」的橘貓**（不是任意一隻貓）
- **你的公司產品**（市面上還沒有的設計）
- **你自己的臉**（模型訓練時沒見過你）

這些「特定概念」模型原本不知道，Dreambooth 就是用來教模型認識這些新概念的。
:::

**工作原理：**
1. 準備 10-30 張目標概念的圖片
2. 給這個概念一個**獨特的識別詞**（例如 `sks 貓`）
3. 用這些圖片微調模型，讓模型學會把識別詞與這個概念連結
4. 之後只要輸入包含識別詞的提示詞，就能生成該概念的圖片

### 18-1-2 什麼是 LoRA？

**LoRA**（Low-Rank Adaptation）原本是用於語言模型的微調技術，後來被成功應用到影像模型。

```
傳統 Dreambooth（全參數微調）：
┌─────────────────────────────────────────┐
│  更新模型所有參數（數十億個）              │
│  需要大量記憶體（>80 GB）                 │
│  容易遺忘原有知識（catastrophic forgetting）│
└─────────────────────────────────────────┘

Dreambooth + LoRA：
┌─────────────────────────────────────────┐
│  只更新一小部分參數（LoRA adapter）        │
│  記憶體用量大幅降低（~45 GB）              │
│  保留原有知識，只注入新概念                │
│  產出的 adapter 只有 100-300 MB           │
└─────────────────────────────────────────┘
```

::: tip 💡 LoRA 的核心優勢
- **檔案小**：從數十 GB 變成 100-300 MB
- **可切換**：可以隨時載入/卸載不同的 LoRA
- **可疊加**：多個 LoRA 可以同時使用
- **不破壞原模型**：基礎模型保持不變
:::

### 18-1-3 多概念微調

你可以同時在一個 LoRA 中注入多個概念：

| 概念類型 | 範例 | 圖片需求 | 識別詞 |
|---------|------|---------|--------|
| 寵物 | 你的貓「咪咪」 | 15-20 張 | `sks 貓` |
| 產品 | DGX Spark 電腦 | 10-15 張 | `sks 設備` |
| 畫風 | 水彩風格 | 20-30 張 | `sks 風格` |
| 人物 | 你的臉 | 15-25 張 | `sks 人` |
| 場景 | 你的房間 | 10-15 張 | `sks 房間` |

::: warning ⚠️ 多概念注意事項
- 概念之間不要太相似（例如兩隻不同顏色的貓可能互相干擾）
- 每個概念使用不同的識別詞
- 概念越多，需要的訓練步數越多
- 建議先從單一概念開始，熟悉後再嘗試多概念
:::

### 18-1-4 DGX Spark 的記憶體優勢

微調 FLUX.1-dev 的記憶體需求：

| 硬體 | 可用記憶體 | 能否微調 FLUX.1-dev |
|------|-----------|-------------------|
| RTX 4090（消費級旗艦） | 24 GB | ❌ 不夠 |
| RTX 6000 Ada（專業級） | 48 GB | ⚠️ 勉強（需降低解析度） |
| **DGX Spark** | **128 GB** | ✅ **輕鬆搞定** |
| A100 80GB（雲端） | 80 GB | ✅ 可以 |

DGX Spark 的 128GB 統一記憶體讓你能以 **1024×1024 原生解析度** 微調 FLUX.1-dev，這是消費級 GPU 做不到的。

---

## 18-2 環境建立與模型下載

### 18-2-1 下載 NVIDIA 官方 Playbook

NVIDIA 提供了專門為 DGX Spark 最佳化的 FLUX 微調 Playbook：

```bash
# 1. 確保已安裝 NVIDIA NGC CLI
# 如果沒有，先安裝：
# wget https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip ngccli_linux.zip

# 2. 下載 Playbook
ngc registry resource download-version "nvidia-ai-workbench/dgx-spark-flux-finetuning:latest"

# 3. 進入目錄
cd dgx-spark-flux-finetuning
ls
# 應該看到：Dockerfile, train.py, README.md 等檔案
```

::: info 🤔 什麼是 NGC？
NGC（NVIDIA GPU Cloud）是 NVIDIA 的容器和模型註冊中心：
- 提供最佳化的 Docker 容器
- 包含預訓練模型和資料集
- 所有資源都針對 NVIDIA GPU 進行了最佳化
- 需要免費註冊 NGC 帳號
:::

### 18-2-2 設定 Hugging Face Token

FLUX.1-dev 是受限制的模型，需要 Hugging Face 授權才能下載：

```bash
# 1. 安裝 Hugging Face CLI（如果還沒裝）
uv pip install huggingface_hub

# 2. 登入
huggingface-cli login

# 3. 貼上你的 HF Token
# Enter your token (input will not be visible): hf_xxxxxxxxxxxxxxxxxxxx

# 4. 確認登入成功
huggingface-cli whoami
```

::: tip 💡 如何取得 Hugging Face Token？
1. 前往 https://huggingface.co/settings/tokens
2. 點擊「Create new token」
3. 選擇「Read」權限即可
4. 複製產生的 token
:::

### 18-2-3 申請 FLUX.1-dev 存取權限

FLUX.1-dev 需要手動申請存取權限：

```bash
# 1. 前往模型頁面
# https://huggingface.co/black-forest-labs/FLUX.1-dev

# 2. 點擊「Agree and access repository」
# 3. 同意使用條款

# 4. 等待核准（通常幾分鐘到幾小時）
```

::: warning ⚠️ 常見問題
如果下載時出現 `401 Unauthorized` 錯誤：
1. 確認已同意模型的使用條款
2. 確認 Token 有 Read 權限
3. 嘗試重新登入：`huggingface-cli login`
:::

### 18-2-4 下載 FLUX.1-dev 模型

```bash
# 建立模型儲存目錄
mkdir -p ~/flux-models

# 下載模型（約 23 GB，需要一些時間）
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir ~/flux-models/flux-dev \
  --resume-download

# 確認下載完成
ls -lh ~/flux-models/flux-dev/
# 應該看到：
# - diffusion_pytorch_model.safetensors（約 23 GB）
# - scheduler/
# - text_encoder/
# - tokenizer/
```

::: info 🤔 模型檔案結構
FLUX.1-dev 包含以下元件：
| 元件 | 大小 | 用途 |
|------|------|------|
| UNet/DiT 模型 | ~23 GB | 核心生成模型 |
| T5 文字編碼器 | ~10 GB | 理解提示詞 |
| CLIP 編碼器 | ~1.5 GB | 輔助文字理解 |
| Scheduler | 很小 | 控制生成過程 |
| Tokenizer | 很小 | 文字分詞 |
:::

### 18-2-5 建構 Docker 映像

```bash
cd dgx-spark-flux-finetuning

# 建構映像（第一次需要 10-20 分鐘）
docker build -t flux-finetune .

# 確認映像建立成功
docker images | grep flux-finetune
# flux-finetune  latest  xxxxxxx  2 hours ago  15GB
```

::: tip 💡 加速 Docker build
如果 build 很慢，可以：
1. 確保網路連線良好
2. 使用 `--progress=plain` 查看詳細進度
3. 如果中斷，Docker 會快取已完成的步驟，重新 build 會從中斷處繼續

```bash
docker build --progress=plain -t flux-finetune .
```
:::

---

## 18-3 準備訓練資料

訓練資料的品質直接決定了微調的效果。這是最重要的一步。

### 18-3-1 圖片選擇原則

| 原則 | 說明 | 範例 |
|------|------|------|
| **多樣性** | 不同角度、光線、背景 | 正面、側面、特寫、遠景 |
| **高品質** | 清晰、不模糊、解析度高 | 至少 1024×1024 |
| **一致性** | 同一個概念的所有圖片要一致 | 同一隻貓、同一個產品 |
| **無干擾** | 避免其他物體搶走焦點 | 簡單的背景 |
| **數量適中** | 10-30 張為佳 | 太少學不會，太多會過擬合 |

::: warning ⚠️ 常見錯誤
- ❌ 使用模糊或低解析度的圖片
- ❌ 圖片中有太多不相關的物體
- ❌ 不同概念的圖片混在一起（例如兩隻不同的貓）
- ❌ 圖片數量太少（少於 5 張）或太多（超過 50 張）
:::

### 18-3-2 準備概念 1：Toy Jensen（貓咪）

```bash
# 建立目錄結構
mkdir -p ~/flux-training/concept1/images

# 放入 15-20 張貓咪的圖片
# 建議包含：
# - 正面照（3-5 張）
# - 側面照（3-5 張）
# - 不同姿勢（坐、躺、走）
# - 不同光線條件
# - 不同背景

# 確認圖片
ls ~/flux-training/concept1/images/
# cat_01.jpg  cat_02.jpg  ...  cat_15.jpg
```

### 18-3-3 準備概念 2：DGX Spark 電腦

```bash
# 建立目錄
mkdir -p ~/flux-training/concept2/images

# 放入 10-15 張 DGX Spark 的照片
# 建議包含：
# - 正面照
# - 側面照
# - 接線的照片
# - 放在桌面上的照片
# - 與其他物品的尺寸對比

ls ~/flux-training/concept2/images/
# dgx_01.jpg  dgx_02.jpg  ...  dgx_12.jpg
```

### 18-3-4 data.toml 設定檔詳解

`data.toml` 是訓練資料的設定檔，告訴訓練程式每個概念的位置和描述：

```toml
# data.toml — 完整設定說明

[[datasets]]
name = "concept1"                              # 概念名稱（內部使用）
directory = "/training/concept1/images"        # 圖片目錄（容器內路徑）
caption = "一隻叫 sks 的橘貓"                   # 描述文字，包含識別詞
num_repeats = 10                               # 每張圖片在每個 epoch 重複幾次
resolution = 1024                              # 訓練解析度
keep_tokens = 4                                # 保留前面幾個 token 不被 dropout

[[datasets]]
name = "concept2"
directory = "/training/concept2/images"
caption = "一台 sks 的綠色迷你 AI 電腦"
num_repeats = 5
resolution = 1024
keep_tokens = 4
```

::: info 🤔 num_repeats 是什麼？
`num_repeats` 控制每個概念在訓練中的「曝光次數」。

公式：`總訓練步數 = 圖片數量 × num_repeats × epochs`

例如：
- 15 張貓圖片 × num_repeats=10 = 150 步/epoch
- 12 張 DGX 圖片 × num_repeats=5 = 60 步/epoch

**設定原則：**
- 重要的概念設更高的 num_repeats
- 圖片較少的概念設更高的 num_repeats
- 兩個概念的總步數應該接近
:::

::: info 🤔 caption 中的識別詞
識別詞（如 `sks`）是一個不常見的詞，模型原本不知道它的意義。
透過訓練，模型會學會 `sks` 代表你的特定概念。

選擇識別詞的建議：
- 使用不常見的字母組合（sks, kjt, wxy）
- 避免使用真實存在的詞（貓、電腦）
- 保持簡短（2-4 個字母）
:::

### 18-3-5 訓練資料總覽

| 概念 | 圖片數 | 描述（caption） | num_repeats | 每 epoch 步數 |
|------|--------|----------------|-------------|-------------|
| 咪咪（貓） | 15 | 一隻叫 sks 的橘貓 | 10 | 150 |
| DGX Spark | 12 | 一台 sks 的綠色迷你 AI 電腦 | 5 | 60 |
| **總計** | **27** | — | — | **210** |

### 18-3-6 自訂概念範例

你可以替換成任何你想微調的概念：

```toml
# 範例 1：微調你的產品
[[datasets]]
name = "my_product"
directory = "/training/product/images"
caption = "一個 sks 設計的無線藍牙音箱"
num_repeats = 8

# 範例 2：微調特定畫風
[[datasets]]
name = "watercolor_style"
directory = "/training/watercolor/images"
caption = "一幅 sks 風格的水彩畫"
num_repeats = 6

# 範例 3：微調你的角色設計
[[datasets]]
name = "my_character"
directory = "/training/character/images"
caption = "一個叫 sks 的動漫女孩，藍色頭髮"
num_repeats = 10
```

---

## 18-4 Dreambooth LoRA 訓練

### 18-4-1 訓練設定檔

```yaml
# train_config.yaml — 完整訓練參數說明

# === 模型設定 ===
model: "black-forest-labs/FLUX.1-dev"        # 基礎模型
pretrained_model_path: "/models/flux-dev"    # 本地模型路徑
revision: null                               # 模型版本（null=最新）

# === 訓練參數 ===
resolution: 1024                             # 訓練解析度
train_batch_size: 1                          # 批次大小
max_train_steps: 1000                        # 最大訓練步數
learning_rate: 1.0e-4                        # 學習率
lr_scheduler: "cosine"                       # 學習率調度器
lr_warmup_steps: 100                         # 學習率預熱步數
scale_lr: false                              # 是否根據 batch size 縮放學習率

# === LoRA 設定 ===
lora_rank: 16                                # LoRA rank
lora_alpha: 16                               # LoRA 縮放係數
lora_dropout: 0.0                            # LoRA dropout（Dreambooth 通常設 0）

# === 最佳化 ===
optimizer: "adamw"                           # 最佳化器
adam_beta1: 0.9
adam_beta2: 0.999
adam_weight_decay: 0.01
adam_epsilon: 1.0e-8
max_grad_norm: 1.0                           # 梯度裁剪

# === 輸出設定 ===
output_dir: "./flux-lora-output"             # 輸出目錄
checkpointing_steps: 200                     # 每隔多少步儲存 checkpoint
checkpoints_total_limit: 3                   # 最多保留幾個 checkpoint
save_state: true                             # 是否儲存訓練狀態

# === 效能設定 ===
mixed_precision: "bf16"                      # 混合精度
gradient_checkpointing: true                 # 梯度檢查點
enable_xformers_memory_efficient_attention: true  # xFormers 記憶體最佳化
seed: 42                                     # 隨機種子（確保可重複性）
```

### 18-4-2 記憶體需求分析

| 解析度 | 記憶體用量 | 訓練速度 | 品質 | 建議場景 |
|--------|-----------|---------|------|---------|
| 512×512 | ~25 GB | 快 | 低 | 快速測試 |
| 768×768 | ~35 GB | 中等 | 中 | 平衡選擇 |
| **1024×1024** | **~45 GB** | **推薦** | **高** | **正式訓練** |
| 1536×1536 | ~70 GB | 慢 | 最高 | 極致品質 |

::: tip 💡 DGX Spark 推薦設定
DGX Spark 有 128GB 記憶體，建議使用 **1024×1024** 解析度：
- 記憶體用量約 45 GB，還有充足空間
- 品質與速度的最佳平衡
- FLUX.1-dev 原生支援此解析度
:::

### 18-4-3 學習率與訓練步數指南

| 資料量 | 推薦步數 | 學習率 | 預期時間 |
|--------|---------|--------|---------|
| 10 張 | 500-800 | 1e-4 | ~15 分鐘 |
| 15-20 張 | 800-1200 | 1e-4 | ~25 分鐘 |
| 25-30 張 | 1000-1500 | 8e-5 | ~40 分鐘 |

::: warning ⚠️ 過擬合警告
如果訓練步數太多或學習率太高，會發生過擬合：
- 模型只會複製訓練圖片，無法泛化
- 表現：生成的圖片與訓練圖片幾乎一模一樣
- 解決：減少步數、降低學習率、減少 num_repeats
:::

### 18-4-4 啟動訓練

```bash
# 啟動 Docker 容器並開始訓練
docker run -it \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/flux-training:/training \
  -v ~/flux-models:/models \
  flux-finetune \
  python train.py \
  --config /training/train_config.yaml \
  --data_config /training/data.toml
```

訓練過程中的輸出：

```
Step 100/1000 | Loss: 0.8523 | LR: 5.0e-5 | Speed: 2.3 steps/sec
Step 200/1000 | Loss: 0.6234 | LR: 8.5e-5 | Speed: 2.3 steps/sec | Saving checkpoint...
Step 300/1000 | Loss: 0.4567 | LR: 9.8e-5 | Speed: 2.2 steps/sec
Step 400/1000 | Loss: 0.3891 | LR: 1.0e-4 | Speed: 2.3 steps/sec | Saving checkpoint...
...
Step 1000/1000 | Loss: 0.2345 | LR: 0.0 | Speed: 2.3 steps/sec | Training complete!
```

::: info 🤔 如何解讀訓練輸出？
- **Loss**：損失值，應該持續下降。如果開始上升，可能過擬合了
- **LR**：當前學習率，cosine scheduler 會先上升後下降
- **Speed**：每秒訓練步數，用於估算完成時間
- **Saving checkpoint**：自動儲存進度，即使中斷也可以恢復
:::

### 18-4-5 訓練完成

訓練完成後，檢查輸出：

```bash
# 查看輸出目錄
ls ~/flux-training/flux-lora-output/
# checkpoint-200/  checkpoint-400/  checkpoint-600/
# pytorch_lora_weights.safetensors  # 最終的 LoRA 權重

# 檢查 LoRA 檔案大小
ls -lh ~/flux-training/flux-lora-output/pytorch_lora_weights.safetensors
# -rw-r--r-- 1 user user 120M Apr 01 12:00 pytorch_lora_weights.safetensors
```

LoRA adapter 的大小通常在 100-300 MB 之間，取決於 rank 設定。

| LoRA Rank | 預期大小 | 效果 |
|-----------|---------|------|
| 8 | ~50-80 MB | 基礎概念學習 |
| **16** | **~100-150 MB** | **推薦，效果與大小的平衡** |
| 32 | ~200-300 MB | 更好的細節捕捉 |
| 64 | ~400-500 MB | 最佳效果，但可能過擬合 |

---

## 18-5 基礎模型 vs. 微調模型推論比較

### 18-5-1 推論測試

訓練完成後，比較基礎模型和微調模型的效果：

```python
import torch
from diffusers import FluxPipeline

# 載入基礎模型
base_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 測試基礎模型
prompt = "a cat sleeping on a sofa"
image = base_pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
image.save("base_cat.png")
# → 生成一隻普通的貓

# 載入微調後的模型（加上 LoRA）
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 載入 LoRA adapter
pipe.load_lora_weights(
    "./flux-lora-output",
    weight_name="pytorch_lora_weights.safetensors"
)

# 測試微調模型
prompt = "一隻叫 sks 的橘貓在沙發上睡覺，陽光從窗戶照進來"
image = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
image.save("finetuned_cat.png")
# → 生成你家貓的圖片！
```

### 18-5-2 提示詞對比測試

| 提示詞 | 基礎模型 | 微調模型 |
|--------|---------|---------|
| `a cat on a sofa` | 普通的貓 | 你的貓（sks） |
| `a green mini computer on a desk` | 普通的綠色電腦 | DGX Spark（sks） |
| `sks 貓在太空飛行` | 不認識 sks | 你的貓在太空中 |
| `sks 設備在海底` | 不認識 sks | DGX Spark 在海底 |

::: tip 💡 提示詞技巧
- **加入場景描述**：`sks 貓在海灘上奔跑` 比單純 `sks 貓` 效果更好
- **加入風格描述**：`sks 貓，水彩風格` 可以結合不同藝術風格
- **避免過度約束**：太多細節可能限制模型的發揮
:::

### 18-5-3 在 ComfyUI 中使用 LoRA

用第 13 章的 ComfyUI 來測試微調後的模型：

```
步驟 1：複製 LoRA 檔案
cp ~/flux-training/flux-lora-output/pytorch_lora_weights.safetensors \
   ~/ComfyUI/models/loras/flux-lora.safetensors

步驟 2：啟動 ComfyUI
cd ~/ComfyUI
python main.py --listen 0.0.0.0

步驟 3：建立工作流程
1. 加入「Load Checkpoint」節點 → 選擇 FLUX.1-dev
2. 加入「Load LoRA」節點 → 選擇 flux-lora.safetensors
3. 設定 LoRA 權重（strength）：1.0
4. 連線到模型
5. 在 CLIP Text Encode 輸入提示詞
6. 連線到 KSampler → VAE Decode → Save Image
7. 點擊「Queue Prompt」開始生成
```

### 18-5-4 不同 Checkpoint 的比較

訓練過程中會儲存多個 checkpoint，比較它們的效果：

| Checkpoint | 訓練步數 | Loss | 概念遵循度 | 泛化能力 | 建議 |
|-----------|---------|------|-----------|---------|------|
| checkpoint-200 | 200 | 0.62 | 低 | 高 | 概念還沒學會 |
| checkpoint-400 | 400 | 0.39 | 中 | 中 | 開始有樣子 |
| checkpoint-600 | 600 | 0.28 | **高** | **中** | **推薦** |
| checkpoint-800 | 800 | 0.25 | 很高 | 低 | 可能開始過擬合 |
| checkpoint-1000 | 1000 | 0.23 | 最高 | 很低 | 可能過擬合 |

::: tip 💡 選擇最佳 checkpoint
1. 每個 checkpoint 都生成幾張圖片
2. 比較概念遵循度（像不像你的概念）
3. 比較泛化能力（能不能在不同場景中生成）
4. 選擇兩者平衡最好的 checkpoint
5. 通常不是最後一個 checkpoint 最好
:::

---

## 18-6 進階技巧

### 18-6-1 解析度與記憶體的關係

解析度越高，記憶體用量呈**平方**成長：

```
記憶體用量 ∝ 解析度²

512²  = 262,144  像素 → ~25 GB
768²  = 589,824  像素 → ~35 GB（1.4x）
1024² = 1,048,576 像素 → ~45 GB（1.8x）
1536² = 2,359,296 像素 → ~70 GB（2.8x）
```

### 18-6-2 不同影像模型的比較

| 模型 | 參數量 | 微調記憶體 | 生成品質 | 生成速度 | 適合場景 |
|------|--------|-----------|---------|---------|---------|
| **FLUX.1-dev** | **12B** | **~45 GB** | **最高** | **中等** | **專業用途** |
| FLUX.1-Schnell | 12B | ~30 GB | 高 | 快（4 步） | 快速原型 |
| SDXL | 6.6B | ~20 GB | 中等 | 快 | 入門學習 |
| SD 1.5 | 0.86B | ~8 GB | 基礎 | 很快 | 低資源環境 |

::: info 🤔 為什麼 FLUX.1-dev 品質最高？
FLUX.1-dev 使用了：
- **Flow Matching**：比傳統 Diffusion 更先進的生成方法
- **12B 參數**：比 SDXL 大近 2 倍
- **混合注意力機制**：更好的文字理解能力
- **原生 1024×1024 訓練**：高解析度細節更好
:::

### 18-6-3 LoRA Rank 的影響

| Rank | 可訓練參數 | 檔案大小 | 概念學習能力 | 過擬合風險 | 建議場景 |
|------|-----------|---------|-------------|-----------|---------|
| 4 | ~10M | ~20 MB | 低 | 很低 | 簡單概念 |
| 8 | ~20M | ~50 MB | 基礎 | 低 | 快速測試 |
| **16** | **~40M** | **~100 MB** | **推薦** | **中** | **正式訓練** |
| 32 | ~80M | ~200 MB | 更好 | 中高 | 複雜概念 |
| 64 | ~160M | ~400 MB | 最佳 | 高 | 研究用途 |

### 18-6-4 LoRA 權重疊加

你可以同時載入多個 LoRA，創造組合效果：

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 載入多個 LoRA
pipe.load_lora_weights("./cat-lora", weight_name="lora.safetensors", adapter_name="cat")
pipe.load_lora_weights("./style-lora", weight_name="lora.safetensors", adapter_name="style")
pipe.load_lora_weights("./dgx-lora", weight_name="lora.safetensors", adapter_name="dgx")

# 設定各 LoRA 的權重
pipe.set_adapters(["cat", "style"], adapter_weights=[0.8, 0.5])

# 生成：你的貓 + 水彩風格
prompt = "一隻叫 sks 的橘貓，sks 風格"
image = pipe(prompt, guidance_scale=3.5).images[0]
image.save("combined.png")

# 動態調整權重
pipe.set_adapters(["cat", "style"], adapter_weights=[1.0, 0.3])
# → 貓的概念更強，風格更弱

# 卸載某個 LoRA
pipe.set_adapters(["cat"], adapter_weights=[1.0])
# → 只保留貓的概念
```

::: tip 💡 LoRA 疊加技巧
- 權重總和不需要等於 1.0
- 單個 LoRA 權重建議在 0.5-1.0 之間
- 超過 1.0 可能導致畫面異常
- 如果效果不好，嘗試降低權重
:::

### 18-6-5 進階訓練參數

```yaml
# 進階訓練設定
scheduler: "cosine"                    # 學習率調度器（cosine/linear/constant）
warmup_steps: 100                      # 預熱步數
gradient_checkpointing: true           # 梯度檢查點（省記憶體但慢 20%）
mixed_precision: "bf16"                # BF16 混合精度
seed: 42                               # 隨機種子（確保可重複）
prior_preservation: false              # 先驗保留（防止遺忘原有概念）
class_data_dir: null                   # 先驗資料目錄
prior_loss_weight: 1.0                 # 先驗損失權重
```

::: info 🤔 先驗保留（Prior Preservation）是什麼？
先驗保留是一種防止模型遺忘原有知識的技術：
- 訓練時同時使用目標概念的圖片和通用概念的圖片
- 例如：訓練「你的貓」時，也加入一些「普通貓」的圖片
- 這樣模型學會你的貓的同時，不會忘記普通貓長什麼樣

在 DGX Spark 上，由於記憶體充足，通常不需要啟用此功能。
:::

### 18-6-6 訓練監控

```bash
# 使用 TensorBoard 監控訓練
tensorboard --logdir ./flux-lora-output/logs --host 0.0.0.0 --port 6006

# 瀏覽器打開 http://DGX_Spark_IP:6006
# 可以看到：
# - Loss 曲線
# - 學習率變化
# - 訓練速度
# - 記憶體使用
```

---

## 18-7 常見問題與疑難排解

### 18-7-1 記憶體不足（OOM）

```yaml
# 解決方案 1：降低解析度
resolution: 512

# 解決方案 2：降低 batch size
train_batch_size: 1

# 解決方案 3：啟用梯度檢查點
gradient_checkpointing: true

# 解決方案 4：使用 xFormers
enable_xformers_memory_efficient_attention: true

# 解決方案 5：改用 FLUX.1-Schnell（記憶體需求較低）
model: "black-forest-labs/FLUX.1-Schnell"
```

::: tip 💡 記憶體監控指令
```bash
# 即時監控 GPU 記憶體
watch -n 1 nvidia-smi

# 監控 Docker 容器記憶體
docker stats
```
:::

### 18-7-2 FLUX.1-dev 下載失敗

```bash
# 問題 1：401 Unauthorized
# 解決：確認已同意使用條款且 Token 正確
huggingface-cli login

# 問題 2：下載中斷
# 解決：使用 --resume-download 繼續
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir ~/flux-models/flux-dev \
  --resume-download

# 問題 3：磁碟空間不足
# 解決：清理空間
df -h
docker system prune
```

### 18-7-3 生成品質不佳

```
檢查清單：
□ 訓練圖片品質是否足夠？（清晰、高解析度）
□ 訓練圖片數量是否足夠？（至少 10 張）
□ 訓練步數是否足夠？（至少 500 步）
□ 學習率是否合適？（嘗試 5e-5 到 2e-4 之間）
□ LoRA rank 是否足夠？（嘗試增加到 32）
□ 提示詞是否包含正確的識別詞？
□ 是否過擬合？（嘗試減少訓練步數）
□ 是否欠擬合？（嘗試增加訓練步數）
```

::: info 🤔 如何判斷過擬合 vs 欠擬合？
- **欠擬合**（訓練不足）：生成的圖片不像你的概念
  - 解決：增加訓練步數、提高學習率、增加 num_repeats
- **過擬合**（訓練過度）：生成的圖片與訓練圖片一模一樣，無法泛化
  - 解決：減少訓練步數、降低學習率、減少 num_repeats
:::

### 18-7-4 ComfyUI 載入 LoRA 失敗

```
問題 1：找不到 LoRA 檔案
解決：確認檔案放在正確位置
ls ~/ComfyUI/models/loras/
# 應該看到你的 .safetensors 檔案

問題 2：LoRA 格式錯誤
解決：確認是 .safetensors 格式（不是 .ckpt 或 .bin）

問題 3：LoRA 與模型不兼容
解決：確認 LoRA 是為 FLUX.1-dev 訓練的
（SDXL 的 LoRA 不能用在 FLUX 上）

問題 4：載入後沒有效果
解決：
1. 確認 LoRA 權重（strength）設為 1.0
2. 確認提示詞中包含正確的識別詞
3. 嘗試不同的 seed
```

### 18-7-5 訓練中斷如何恢復

```bash
# 如果訓練中斷，可以從最後的 checkpoint 恢復
docker run -it \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/flux-training:/training \
  -v ~/flux-models:/models \
  flux-finetune \
  python train.py \
  --config /training/train_config.yaml \
  --data_config /training/data.toml \
  --resume_from_checkpoint /training/flux-lora-output/checkpoint-600
```

### 18-7-6 訓練速度太慢

```
加速建議：
1. 確認使用了 bf16 混合精度
2. 啟用 gradient_checkpointing（雖然慢 20%，但避免 OOM 重跑）
3. 啟用 xFormers 記憶體最佳化
4. 降低解析度到 768（如果 1024 太慢）
5. 確認沒有其他程式佔用 GPU
   nvidia-smi  # 查看 GPU 使用情況
```

---

## 18-8 本章小結

::: success ✅ 你現在知道了
- Dreambooth LoRA 可以把特定概念（寵物、產品、畫風）注入圖片生成模型
- DGX Spark 的 128GB 記憶體讓 FLUX.1-dev 微調成為可能，這是消費級 GPU 做不到的
- 訓練資料的品質比數量更重要，10-30 張高品質圖片即可
- LoRA rank 影響效果與檔案大小，16 是推薦的起點
- 多個 LoRA 可以疊加使用，創造組合效果
- 選擇最佳 checkpoint 需要平衡概念遵循度和泛化能力
:::

::: tip 🚀 下一章預告
微調是站在巨人的肩膀上。那如果我們想「從零開始」訓練一個模型呢？下一章來看看預訓練！

👉 [前往第 19 章：預訓練中小型語言模型 →](/guide/chapter19/)
:::

::: info 📝 上一章
← [回到第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調](/guide/chapter17/)
:::
