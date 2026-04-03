# 第 18 章：影像模型微調 — FLUX Dreambooth LoRA

::: tip 🎯 本章你將學到什麼
- Dreambooth 與 LoRA 的概念
- 多概念微調
- 環境建立與模型下載
- 準備訓練資料（圖片 + 設定檔）
- Dreambooth LoRA 訓練
- 微調前後比較
:::

---

## 18-1 影像生成模型微調概念

### 18-1-1 Dreambooth 與 LoRA

**Dreambooth** 是一種把「特定概念」注入模型的方法。

舉例：你想讓模型認識你的寵物貓「咪咪」。一般模型不知道「咪咪」長什麼樣，但用 Dreambooth 微調後，你只要輸入「一隻叫咪咪的貓在沙發上睡覺」，它就能生成你家貓的圖片。

**Dreambooth LoRA** = Dreambooth + LoRA，用 LoRA 的方式做 Dreambooth 微調，記憶體用量大幅降低。

### 18-1-2 多概念微調

你可以同時注入多個概念：
- 你的寵物
- 你的產品
- 特定的畫風

### 18-1-3 DGX Spark 的記憶體優勢

微調 FLUX.1-dev 需要約 40-60 GB 記憶體。一般消費級 GPU（24GB）做不到，但 DGX Spark 的 128GB 輕鬆搞定。

---

## 18-2 環境建立與模型下載

### 18-2-1 下載 NVIDIA 官方 Playbook

```bash
ngc registry resource download-version "nvidia-ai-workbench/dgx-spark-flux-finetuning:latest"
```

### 18-2-2 設定 Hugging Face Token

FLUX.1-dev 需要 Hugging Face 授權：

```bash
huggingface-cli login
# 貼上你的 HF Token
```

### 18-2-3 下載模型

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir ~/flux-models/flux-dev
```

### 18-2-4 建構 Docker 映像

```bash
cd dgx-spark-flux-finetuning
docker build -t flux-finetune .
```

---

## 18-3 準備訓練資料

### 18-3-1 Toy Jensen 訓練圖片

準備 10-30 張目標概念的圖片。例如要微調一個角色：

```bash
mkdir -p ~/flux-training/concept1/images
# 放入 10-30 張圖片
```

### 18-3-2 DGX Spark GPU 訓練圖片

同樣的方式，準備 DGX Spark 的照片：

```bash
mkdir -p ~/flux-training/concept2/images
# 放入 DGX Spark 的照片
```

### 18-3-3 data.toml 設定檔

```toml
# data.toml
[[datasets]]
name = "concept1"
directory = "/training/concept1/images"
caption = "一隻叫咪咪的橘貓"
num_repeats = 10

[[datasets]]
name = "concept2"
directory = "/training/concept2/images"
caption = "一台綠色的 DGX Spark 迷你電腦"
num_repeats = 5
```

### 18-3-4 訓練資料總覽

| 概念 | 圖片數 | 描述 | 重複次數 |
|------|--------|------|---------|
| 咪咪（貓） | 20 | 一隻橘貓 | 10 |
| DGX Spark | 15 | 綠色迷你電腦 | 5 |

### 18-3-5 自訂概念

你可以替換成任何你想微調的概念：
- 你的產品
- 你的畫風
- 你的角色設計

---

## 18-4 Dreambooth LoRA 訓練

### 18-4-1 訓練參數

```yaml
# train_config.yaml
model: "black-forest-labs/FLUX.1-dev"
resolution: 1024
train_batch_size: 1
learning_rate: 1.0e-4
max_train_steps: 1000
lora_rank: 16
lora_alpha: 16
output_dir: "./flux-lora-output"
```

### 18-4-2 記憶體需求分析

| 解析度 | 記憶體用量 | 訓練速度 |
|--------|-----------|---------|
| 512x512 | ~25 GB | 快 |
| 768x768 | ~35 GB | 中等 |
| **1024x1024** | **~45 GB** | **推薦** |
| 1536x1536 | ~70 GB | 慢 |

### 18-4-3 啟動訓練

```bash
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

### 18-4-4 訓練完成

訓練完成後，LoRA adapter 約 100-300 MB。

### 18-4-5 Checkpoint 檔案

```bash
ls ~/flux-training/flux-lora-output/
# pytorch_lora_weights.safetensors
```

---

## 18-5 基礎模型 vs. 微調模型推論比較

### 18-5-1 推論提示詞範例

```
# 基礎模型
"a cat sleeping on a sofa"
# → 生成一隻普通的貓

# 微調模型
"一隻叫咪咪的橘貓在沙發上睡覺"
# → 生成你家貓的圖片！
```

### 18-5-2 啟動 ComfyUI

用第 13 章的 ComfyUI，載入微調後的 LoRA：

1. 把 `pytorch_lora_weights.safetensors` 放到 `ComfyUI/models/loras/`
2. 在工作流程中加入 **Load LoRA** 節點
3. 連線到 FLUX 模型

### 18-5-3 不同 checkpoint 的比較

| Checkpoint | 訓練步數 | 概念遵循度 | 泛化能力 |
|-----------|---------|-----------|---------|
| 500 steps | 低 | 高 |
| 1000 steps | 中 | 中 |
| 1500 steps | **高** | 低 |

---

## 18-6 進階技巧

### 18-6-1 解析度與記憶體的關係

解析度越高，記憶體用量呈平方成長。

### 18-6-2 不同影像模型的比較

| 模型 | 微調記憶體 | 品質 | 速度 |
|------|-----------|------|------|
| FLUX.1-dev | ~45 GB | 最高 | 中等 |
| FLUX.1-Schnell | ~30 GB | 高 | 快 |
| SDXL | ~20 GB | 中等 | 快 |

### 18-6-3 LoRA Rank 的影響

| Rank | 檔案大小 | 效果 | 記憶體 |
|------|---------|------|--------|
| 8 | ~50 MB | 基礎 | 低 |
| **16** | **~100 MB** | **推薦** | **中** |
| 32 | ~200 MB | 更好 | 高 |
| 64 | ~400 MB | 最佳 | 很高 |

### 18-6-4 LoRA 權重疊加

你可以同時載入多個 LoRA：

```python
pipeline.load_lora_weights("./cat-lora", weight_name="lora.safetensors", adapter_name="cat")
pipeline.load_lora_weights("./style-lora", weight_name="lora.safetensors", adapter_name="style")

# 設定權重
pipeline.set_adapters(["cat", "style"], adapter_weights=[0.8, 0.5])
```

### 18-6-5 自訂訓練參數

```yaml
# 進階設定
scheduler: "cosine"
warmup_steps: 100
gradient_checkpointing: true
mixed_precision: "bf16"
seed: 42
```

---

## 18-7 常見問題與疑難排解

### 18-7-1 記憶體不足

```yaml
# 降低解析度
resolution: 512

# 或降低 batch size
train_batch_size: 1
```

### 18-7-2 FLUX.1-dev 下載失敗

確認 Hugging Face Token 正確，並且已同意 FLUX.1-dev 的使用條款。

### 18-7-3 生成品質不佳

- 增加訓練步數
- 增加訓練圖片數量
- 調整 LoRA rank
- 檢查提示詞是否正確

### 18-7-4 ComfyUI 載入 LoRA 失敗

確認 LoRA 檔案格式為 `.safetensors`，並且放在正確的目錄。

---

## 18-8 本章小結

::: success ✅ 你現在知道了
- Dreambooth LoRA 可以把特定概念注入圖片生成模型
- DGX Spark 的 128GB 記憶體讓 FLUX.1-dev 微調成為可能
- LoRA rank 影響效果與檔案大小
- 多個 LoRA 可以疊加使用
:::

::: tip 🚀 下一章預告
微調是站在巨人的肩膀上。那如果我們想「從零開始」訓練一個模型呢？下一章來看看預訓練！

👉 [前往第 19 章：預訓練中小型語言模型 →](/guide/chapter19/)
:::

::: info 📝 上一章
← [回到第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調](/guide/chapter17/)
:::
