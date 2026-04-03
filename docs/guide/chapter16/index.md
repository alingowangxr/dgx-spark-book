# 第 16 章：Unsloth — 最快的微調框架

::: tip 🎯 本章你將學到什麼
- Unsloth 的優勢和原理
- 環境建立與安裝
- QLoRA 微調 Qwen3-8B 完整實戰
- 模型匯出為 GGUF 給 Ollama 使用
- 微調 120B 大模型
- Unsloth vs. 標準 PEFT 全面比較
- 訓練監控與除錯
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。
:::

---

## 16-1 Unsloth 介紹與安裝

### 16-1-1 什麼是 Unsloth？

Unsloth 是一個開源的 LLM 微調加速框架，專門針對消費級 GPU 和邊緣裝置（如 DGX Spark）進行了深度優化。它的核心目標是：**讓微調更快、更省記憶體、更簡單**。

::: info 🤔 Unsloth 為什麼比較快？
Unsloth 的加速來自三個關鍵技術：

1. **手寫的 Triton 核心（Hand-written Triton Kernels）**
   - 取代了 PyTorch 的預設運算
   - 針對注意力機制和 LoRA 運算做了手動優化
   - 減少了記憶體讀寫次數

2. **自動 Flash Attention 2**
   - 無需手動設定，Unsloth 自動啟用
   - 將注意力的計算複雜度從 $O(n^2)$ 降到 $O(n)$
   - 對長序列（4096+ token）特別有效

3. **最佳化的梯度檢查點**
   - 智能選擇哪些層需要儲存激活值
   - 在記憶體和速度之間取得最佳平衡

綜合效果：**訓練速度提升 2 倍，記憶體用量減少 30%**。
:::

### 16-1-2 Unsloth vs. 標準 PEFT 特性比較

| 特性 | Unsloth | 標準 PEFT | 差異說明 |
|------|---------|----------|---------|
| 訓練速度 | **2x 快** | 基準 | Unsloth 的 Triton 核心加速 |
| 記憶體用量 | **30% 少** | 基準 | 最佳化的記憶體管理 |
| 支援模型 | 主流模型（Qwen、Llama、Mistral 等） | 所有 Transformers 模型 | Unsloth 需要針對每個模型手動優化 |
| Flash Attention | 自動啟用 | 需手動設定 | Unsloth 自動偵測並啟用 |
| 4-bit 訓練 | ✅ 最佳化 | ✅ | Unsloth 的 4-bit 訓練更快 |
| GGUF 匯出 | ✅ 一鍵 | 需手動轉換 | Unsloth 內建 llama.cpp 支援 |
| 學習曲線 | 低（API 更簡潔） | 中等 | Unsloth 封裝了複雜設定 |
| 自訂性 | 中等 | 高 | 標準 PEFT 可以更細粒度控制 |

### 16-1-3 用 Docker 部署 Unsloth 環境

使用 Docker 是最乾淨的安裝方式，避免套件版本衝突：

```bash
# 拉取 Unsloth 官方映像檔
docker run -d \
  --name unsloth \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/unsloth-training:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  unslothai/unsloth:latest
```

各參數說明：

| 參數 | 說明 | 為什麼重要 |
|------|------|-----------|
| `--gpus all` | 啟用所有 GPU | 沒有這個參數，容器無法使用 GPU |
| `--shm-size=16g` | 增加共享記憶體 | PyTorch DataLoader 需要大量共享記憶體 |
| `-v ~/unsloth-training:/workspace` | 映射工作目錄 | 訓練結果會持久化到主機 |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | 映射 HF 快取 | 避免重複下載模型 |

### 16-1-4 驗證安裝

進入容器並驗證 Unsloth 是否正常運作：

```bash
# 進入容器
docker exec -it unsloth bash

# 驗證 Unsloth
python -c "
import unsloth
print(f'Unsloth 版本: {unsloth.__version__}')
import torch
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU 記憶體: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'BF16 支援: {torch.cuda.is_bf16_supported()}')
"
```

預期輸出：
```
Unsloth 版本: 2025.x.x
CUDA 可用: True
GPU: NVIDIA DGX Spark
GPU 記憶體: 128.0 GB
BF16 支援: True
```

### 16-1-5 不使用 Docker 的安裝方式

如果你偏好直接在主機上安裝：

```bash
# 建立 Python 環境
uv venv ~/unsloth-env
source ~/unsloth-env/bin/activate

# 安裝 Unsloth（CUDA 12.x 版本）
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 安裝額外相依套件
uv pip install \
  xformers \
  trl \
  peft \
  accelerate \
  bitsandbytes \
  datasets \
  huggingface_hub
```

::: warning ⚠️ ARM64 注意事項
Unsloth 的 Triton 核心目前主要針對 x86_64 架構優化。在 ARM64（DGX Spark 使用 ARM 處理器）上：
- Unsloth 會自動偵測並使用備用方案
- 部分優化可能無法啟用
- 仍然比標準 PEFT 快，但加速比可能不到 2x

如果遇到 Triton 編譯錯誤，Unsloth 會自動回退到 PyTorch 的實作。
:::

---

## 16-2 QLoRA 微調實戰：Qwen3-8B

這是最核心的部分，我們將從頭到尾完成一次完整的微調流程。

### 16-2-1 載入模型

Unsloth 的 `FastLanguageModel` 是整個框架的入口點，它封裝了所有最佳化：

```python
from unsloth import FastLanguageModel
import torch

# 載入模型和 tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=4096,           # 最大序列長度
    dtype=None,                    # 自動選擇最佳精度（通常是 bfloat16）
    load_in_4bit=True,             # 啟用 4-bit 量化（QLoRA）
    # 以下是可選參數：
    # token="your_hf_token",       # 如果需要存取需要授權的模型
    # max_lora_rank=64,            # 最大 LoRA rank
    # use_gradient_checkpointing=True,  # 啟用梯度檢查點
)
```

::: info 🤔 FastLanguageModel 自動做了什麼？
當你呼叫 `FastLanguageModel.from_pretrained()` 時，Unsloth 在幕後自動完成了以下最佳化：

1. **模型載入最佳化**：使用 4-bit 量化載入，節省 ~75% 記憶體
2. **Flash Attention 2**：自動啟用，加速注意力計算
3. **RoPE 縮放**：自動設定旋轉位置編碼的縮放因子
4. **SwiGLU 最佳化**：針對 FFN 層使用手寫的 Triton 核心
5. **精度自動選擇**：根據硬體選擇最佳精度（BF16 > FP16 > FP32）

你不需要手動設定任何這些，Unsloth 會自動處理。
:::

### 16-2-2 設定 LoRA

Unsloth 提供了簡化的 `get_peft_model` 方法：

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                                    # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力層
        "gate_proj", "up_proj", "down_proj",       # FFN 層
    ],
    lora_alpha=128,                          # LoRA 縮放係數
    lora_dropout=0,                          # Unsloth 建議設為 0（內部已有正則化）
    bias="none",                             # 不訓練 bias
    use_gradient_checkpointing="unsloth",    # Unsloth 最佳化的梯度檢查點
    random_state=42,                         # 固定隨機種子
    use_rslora=False,                        # 可選：啟用 rsLoRA
    use_dora=False,                          # 可選：啟用 DoRA
)
```

**參數詳細說明**：

| 參數 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| `r` | LoRA rank | 16-128 | 越大表達力越強，但參數越多 |
| `target_modules` | 要套用 LoRA 的層 | 注意力 + FFN | 越多層效果越好 |
| `lora_alpha` | 縮放係數 | `2 * r` | 控制 LoRA 的影響力 |
| `lora_dropout` | Dropout 率 | 0（Unsloth）或 0.05-0.1 | 防止過擬合 |
| `use_gradient_checkpointing` | 梯度檢查點 | `"unsloth"` | 節省記憶體，速度略降 |
| `use_rslora` | 啟用 rsLoRA | `False`（預設）或 `True` | 訓練更穩定 |
| `use_dora` | 啟用 DoRA | `False`（預設）或 `True` | 效果更好 |

::: tip 💡 Unsloth 的 LoRA 設定建議
- `lora_dropout=0`：Unsloth 內部已有正則化機制，不需要額外的 dropout
- `use_gradient_checkpointing="unsloth"`：使用 Unsloth 最佳化的版本，比原版快 30%
- 如果記憶體充足，可以同時啟用 `use_rslora=True` 和 `use_dora=True`
:::

### 16-2-3 準備訓練資料

Unsloth 支援多種資料格式，最常用的是 Alpaca 格式：

```python
from datasets import load_dataset

# 定義 Alpaca 提示詞模板
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def format_samples(examples):
    """
    將資料集格式化為 Alpaca 格式
    
    輸入格式：
    {"instruction": "...", "input": "...", "output": "..."}
    
    輸出格式：
    {"text": "Below is an instruction...\n\n### Instruction:\n...\n\n### Response:\n..."}
    """
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples.get("input", [""] * len(examples["instruction"])),
        examples["output"]
    ):
        # 如果有 input，合併到 instruction 中
        if input_text and input_text.strip():
            full_instruction = f"{instruction}\n\n背景資訊：{input_text}"
        else:
            full_instruction = instruction
        
        # 套用模板
        text = alpaca_prompt.format(full_instruction, output)
        texts.append(text)
    
    return {"text": texts}

# 載入中文 Alpaca 資料集
dataset = load_dataset("shibing624/alpaca-zh", split="train")

# 格式化
dataset = dataset.map(format_samples, batched=True)

# 查看結果
print(f"資料筆數: {len(dataset)}")
print(f"範例:\n{dataset[0]['text'][:200]}...")
```

**使用自訂資料**：

如果你有自訂的 JSON 或 JSONL 檔案：

```python
from datasets import load_dataset

# 從 JSONL 檔案載入
dataset = load_dataset("json", data_files="my-data.jsonl", split="train")

# 從 JSON 檔案載入
dataset = load_dataset("json", data_files="my-data.json", split="train")

# 從目錄載入（多個檔案）
dataset = load_dataset("json", data_dir="./my-data/", split="train")
```

**自訂資料格式範例**（JSONL）：

```jsonl
{"instruction": "解釋什麼是量子計算", "output": "量子計算是一種利用量子力學原理..."}
{"instruction": "寫一首關於秋天的詩", "output": "秋風送爽葉飄黃，..."}
{"instruction": "DGX Spark 適合什麼場景？", "output": "DGX Spark 適合本地部署大型 AI 模型..."}
```

### 16-2-4 開始訓練

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 設定訓練參數
training_args = TrainingArguments(
    per_device_train_batch_size=2,           # 每個裝置的 batch size
    gradient_accumulation_steps=4,           # 梯度累積（等效 batch size = 8）
    warmup_steps=5,                          # 學習率預熱步數
    max_steps=60,                            # 最大訓練步數（示範用，實際建議 500+）
    learning_rate=2e-4,                      # 學習率
    fp16=not is_bfloat16_supported(),        # 如果不支援 BF16 則用 FP16
    bf16=is_bfloat16_supported(),            # DGX Spark 支援 BF16
    logging_steps=1,                         # 每一步都記錄（方便監控）
    optim="adamw_8bit",                      # 8-bit AdamW 優化器（Unsloth 最佳化）
    weight_decay=0.01,                       # 權重衰減
    lr_scheduler_type="cosine",              # 餘弦學習率排程
    seed=42,                                 # 固定隨機種子
    output_dir="./outputs",                  # 輸出目錄
    report_to="tensorboard",                 # 使用 TensorBoard 監控
)

# 建立訓練器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    dataset_num_proc=2,                      # 資料處理的平行進程數
    args=training_args,
)

# 顯示訓練前的 GPU 記憶體使用
print(f"訓練前 GPU 記憶體: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# 開始訓練！
trainer_stats = trainer.train()

# 顯示訓練統計
print(f"\n訓練完成！")
print(f"總耗時: {trainer_stats.metrics['train_runtime']:.0f} 秒")
print(f"平均每步耗時: {trainer_stats.metrics['train_runtime'] / trainer_stats.metrics['total_flos'] * 1e12:.2f} 秒")
print(f"最終 Loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"GPU 記憶體峰值: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

::: tip 💡 Unsloth 的訓練速度
在 DGX Spark 上，Unsloth 的訓練速度比標準 PEFT 快約 **2 倍**，記憶體用量減少約 **30%**。

實際數據比較（Qwen3-8B，60 步）：

| 框架 | 耗時 | GPU 記憶體 | Loss |
|------|------|-----------|------|
| 標準 PEFT | ~120 秒 | ~45 GB | 0.82 |
| **Unsloth** | **~60 秒** | **~32 GB** | **0.80** |
:::

### 16-2-5 訓練參數調優指南

根據你的需求調整訓練參數：

| 場景 | max_steps | learning_rate | batch_size | 說明 |
|------|-----------|--------------|------------|------|
| **快速測試** | 30-60 | 2e-4 | 2 | 驗證流程是否正確 |
| **一般微調** | 200-500 | 2e-4 | 2-4 | 大多數場景的甜蜜點 |
| **深度微調** | 1000-3000 | 1e-4 | 4 | 需要大量資料時 |
| **少量資料** | 100-200 | 5e-5 | 1 | 防止過擬合 |
| **大量資料** | 1-3 epochs | 1e-4 | 4-8 | 用 epoch 而非 step 控制 |

---

## 16-3 推論測試

訓練完成後，測試微調後的模型：

```python
# 切換到推論模式（Unsloth 的最佳化推論）
FastLanguageModel.for_inference(model)

# 使用 Alpaca 模板測試
prompt = alpaca_prompt.format(
    "介紹 DGX Spark 的三個主要優點",  # instruction
    "",                               # response（留空讓模型生成）
)

# 編碼
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=256,              # 最大生成 token 數
    temperature=0.7,                 # 溫度（越低越確定）
    top_p=0.9,                       # nucleus sampling
    do_sample=True,                  # 啟用取樣
    use_cache=True,                  # 使用 KV cache 加速
    repetition_penalty=1.1,          # 重複懲罰
)

# 解碼並列印
result = tokenizer.batch_decode(outputs)[0]
print(result)
```

**批次推論**：

```python
# 多個提示詞同時推論
prompts = [
    alpaca_prompt.format("什麼是 LoRA？", ""),
    alpaca_prompt.format("如何部署一個 AI 客服系統？", ""),
    alpaca_prompt.format("解釋 DGX Spark 的統一記憶體架構", ""),
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)

for i, output in enumerate(tokenizer.batch_decode(outputs)):
    print(f"\n--- 提示詞 {i+1} ---")
    print(output)
```

::: tip 💡 推論參數調優
- **temperature=0.7**：平衡創造性和準確性。需要精確回答時用 0.1-0.3，需要創意時用 0.8-1.0
- **top_p=0.9**：只考慮累積機率達 90% 的 token
- **repetition_penalty=1.1**：防止模型重複說同樣的話
- **max_new_tokens**：根據你的需求調整，太短可能回答不完整，太長可能產生無關內容
:::

---

## 16-4 模型匯出與部署

### 16-4-1 儲存 LoRA Adapter

最輕量的方式，只儲存訓練的 LoRA 權重：

```python
# 儲存 LoRA adapter（通常只有 50-200 MB）
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")

print(f"LoRA adapter 已儲存至 ./lora-adapter")
print(f"大小: {sum(f.stat().st_size for f in Path('./lora-adapter').rglob('*')) / 1e6:.1f} MB")
```

**載入 LoRA adapter**：

```python
# 載入基礎模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# 套用 LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
)

# 載入訓練好的權重
from peft import PeftModel
model = PeftModel.from_pretrained(model, "./lora-adapter")
```

### 16-4-2 合併並匯出 16-bit 格式

將 LoRA 權重合併到基礎模型中，得到一個完整的模型：

```python
# 合併並匯出為 16-bit 格式
model.save_pretrained_merged(
    "./merged-model",
    tokenizer,
    save_method="merged_16bit",    # 16-bit 合併
)

# 其他合併選項：
# save_method="merged_4bit"       # 4-bit 合併（更小）
# save_method="merged_8bit"       # 8-bit 合併（平衡）
```

合併後的模型可以直接使用，不需要額外的 LoRA 載入步驟：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./merged-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./merged-model")
```

### 16-4-3 匯出 GGUF 格式給 Ollama 使用

GGUF 是 llama.cpp 的模型格式，Ollama 使用這種格式來運行模型。Unsloth 可以一鍵匯出：

```python
# 匯出 GGUF 格式
model.save_pretrained_gguf(
    "./gguf-model",
    tokenizer,
    quantization_method="q4_k_m",    # 量化方法
)
```

**可用的量化方法**：

| 量化方法 | 大小（8B 模型） | 品質 | 速度 | 推薦場景 |
|---------|---------------|------|------|---------|
| `q2_k` | ~3 GB | 低 | 最快 | 資源極度受限 |
| `q3_k_m` | ~4 GB | 中低 | 快 | 邊緣裝置 |
| `q4_k_m` | ~5 GB | 中高 | 快 | **一般推薦** |
| `q5_k_m` | ~6 GB | 高 | 中等 | 品質優先 |
| `q6_k` | ~7 GB | 很高 | 中等 | 接近原始品質 |
| `q8_0` | ~9 GB | 極高 | 慢 | 幾乎無損 |
| `f16` | ~16 GB | 最高 | 最慢 | 需要最高品質 |

### 16-4-4 用 Ollama 載入微調後的模型

匯出 GGUF 後，可以直接用 Ollama 載入：

```bash
# 建立 Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./gguf-model/unsloth.Q4_K_M.gguf

# 設定系統提示詞
SYSTEM """你是一個專業的 AI 助手，專門回答關於 DGX Spark 和 AI 部署的問題。
你的回答應該準確、簡潔且有幫助。"""

# 設定參數
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF

# 建立 Ollama 模型
ollama create my-qwen3-finetuned -f Modelfile

# 測試
ollama run my-qwen3-finetuned "介紹 DGX Spark 的主要優點"
```

**完整部署流程**：

```bash
# 1. 確認 GGUF 檔案存在
ls -lh ./gguf-model/unsloth.Q4_K_M.gguf

# 2. 建立 Modelfile
cat > Modelfile << 'EOF'
FROM ./gguf-model/unsloth.Q4_K_M.gguf
SYSTEM 你是一個專業的 AI 助手。
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

# 3. 建立模型
ollama create my-qwen3-finetuned -f Modelfile

# 4. 列出所有模型確認已建立
ollama list

# 5. 測試對話
ollama run my-qwen3-finetuned

# 6. 透過 API 使用
curl http://localhost:11434/api/generate -d '{
  "model": "my-qwen3-finetuned",
  "prompt": "什麼是 LoRA？",
  "stream": false
}'
```

---

## 16-5 Loss 曲線分析

訓練過程中監控 Loss 曲線是確保訓練正常的重要步驟。

### 16-5-1 讀取訓練日誌並繪製

```python
import matplotlib.pyplot as plt
import json

# 讀取訓練日誌
with open("./outputs/trainer_state.json") as f:
    state = json.load(f)

# 提取 Loss 數據
steps = []
losses = []
learning_rates = []

for log in state["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])
    if "learning_rate" in log:
        learning_rates.append(log["learning_rate"])

# 繪製 Loss 曲線
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲線
axes[0].plot(steps, losses, 'b-o', markersize=4, linewidth=1.5)
axes[0].set_xlabel('訓練步數', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Unsloth 訓練 Loss 曲線', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].annotate(
    f'最終 Loss: {losses[-1]:.4f}',
    xy=(steps[-1], losses[-1]),
    fontsize=10,
    color='red'
)

# 學習率曲線
if learning_rates:
    lr_steps = list(range(len(learning_rates)))
    axes[1].plot(lr_steps, learning_rates, 'g-o', markersize=4, linewidth=1.5)
    axes[1].set_xlabel('訓練步數', fontsize=12)
    axes[1].set_ylabel('學習率', fontsize=12)
    axes[1].set_title('學習率排程', fontsize=14)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training-curves.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 16-5-2 使用 TensorBoard 即時監控

```bash
# 啟動 TensorBoard
tensorboard --logdir ./outputs --host 0.0.0.0 --port 6006

# 然後用瀏覽器打開 http://DGX_Spark_IP:6006
```

TensorBoard 可以即時顯示：
- Loss 曲線（即時更新）
- 學習率變化
- GPU 記憶體使用
- 訓練速度（step/sec）

### 16-5-3 如何判斷訓練是否正常？

**正常的 Loss 曲線特徵**：
```
Loss
  │
2.0│ ╲
  │  ╲
1.5│   ╲
  │    ╲
1.0│     ╲___
  │         ╲__
0.5│           ╲__
  │              ╲_
0.0│________________
  0   100  200  300  步數
```
- 初期快速下降
- 中期穩定下降
- 後期趨於平緩

**異常情況**：

| 情況 | 圖形 | 原因 | 解決方案 |
|------|------|------|---------|
| Loss 不下降 | `──────` | 學習率太低、資料有問題 | 提高學習率、檢查資料 |
| Loss 爆炸 | `╱╱╱╱` | 學習率太高、梯度爆炸 | 降低學習率、增加 gradient clipping |
| Loss 震盪 | `╱╲╱╲╱╲` | batch size 太小、學習率太高 | 增加 batch size、降低學習率 |
| Loss 先降後升 | `╲╱` | 過擬合 | 減少訓練步數、增加 dropout |

---

## 16-6 進階：微調 120B 大模型

Unsloth 也支援微調超大模型，但需要足夠的記憶體。在 DGX Spark 的 128GB 統一記憶體上，可以微調 120B 等級的 MoE 模型。

### 16-6-1 載入 120B 模型

```python
from unsloth import FastLanguageModel

# 載入 Qwen3.5-122B（MoE 架構）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-122B-A14B-NVFP4",
    max_seq_length=2048,             # 較短的序列長度以節省記憶體
    dtype=None,
    load_in_4bit=True,               # 必須使用 4-bit
)
```

::: info 🤔 為什麼 120B 模型可以在 128GB 記憶體上運行？
Qwen3.5-122B 是一個 MoE（Mixture of Experts）模型。雖然總參數是 122B，但每次推論只激活約 14B 的參數（Active Parameters）。

加上 4-bit 量化：
- 激活參數：14B × 4-bit = ~7 GB
- 非激活參數：108B × 4-bit = ~54 GB
- LoRA adapter：~200 MB
- 總計：~62 GB（在 128GB 記憶體範圍內）
:::

### 16-6-2 設定 LoRA 並訓練

```python
# 為大模型設定 LoRA（使用較小的 rank 以節省記憶體）
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                            # 較小的 rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=64,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 訓練參數（更保守的設定）
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,   # batch size 設為 1
        gradient_accumulation_steps=8,   # 增加梯度累積
        warmup_steps=10,
        max_steps=200,
        learning_rate=1e-4,              # 較低的學習率（大模型更敏感）
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
        output_dir="./outputs-120b",
    ),
)

trainer.train()
```

::: warning ⚠️ 記憶體需求
微調 120B 模型需要約 80-100 GB 記憶體。確保：
- 沒有其他大型程式在執行
- 關閉不必要的 Docker 容器
- 監控記憶體使用：`watch -n 1 nvidia-smi`
- 如果記憶體不足，降低 `max_seq_length` 或 `r`
:::

### 16-6-3 大模型微調的注意事項

| 項目 | 8B 模型 | 120B 模型 | 說明 |
|------|--------|----------|------|
| `r`（rank） | 64 | 16-32 | 大模型用較小的 rank |
| `learning_rate` | 2e-4 | 1e-4 | 大模型需要更保守的學習率 |
| `max_seq_length` | 4096 | 2048 | 大模型需要更短的序列 |
| `batch_size` | 2-4 | 1 | 大模型只能處理 batch size 1 |
| `max_steps` | 200-500 | 100-300 | 大模型收斂更快 |
| 訓練時間 | 1-2 小時 | 3-6 小時 | 視資料量而定 |

---

## 16-7 NVIDIA 官方 Unsloth Playbook

NVIDIA 提供了針對 DGX Spark 最佳化的 Unsloth Playbook：

```bash
# 從 NGC 下載
ngc registry resource download-version "nvidia-ai-workbench/dgx-spark-unsloth:latest"

# 進入目錄
cd dgx-spark-unsloth_v1.0

# 啟動 Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

官方 Playbook 包含：
- 針對 DGX Spark 硬體最佳化的 Unsloth 設定
- Qwen3 系列模型的完整微調流程
- 效能基準測試和比較
- GGUF 匯出和 Ollama 部署指南

---

## 16-8 Unsloth vs. 標準 PEFT 全面比較

### 16-8-1 效能比較

| 特性 | Unsloth | 標準 PEFT | 差異 |
|------|---------|----------|------|
| 訓練速度 | **2x 快** | 基準 | Triton 核心 + Flash Attention |
| 記憶體用量 | **30% 少** | 基準 | 最佳化的記憶體管理 |
| 支援模型 | 主流模型 | 所有模型 | Unsloth 需針對每個模型優化 |
| Flash Attention | 自動啟用 | 需手動設定 | Unsloth 零設定 |
| 4-bit 訓練 | ✅ 最佳化 | ✅ | Unsloth 的 4-bit 更快 |
| GGUF 匯出 | ✅ 一鍵 | 需手動轉換 | Unsloth 內建 llama.cpp |
| 學習曲線 | 低 | 中等 | Unsloth API 更簡潔 |
| 自訂性 | 中等 | 高 | PEFT 可更細粒度控制 |
| 社群支援 | 活躍 | 最大 | HuggingFace PEFT 社群更大 |
| 文件品質 | 良好 | 優秀 | HuggingFace 文件更完整 |

### 16-8-2 何時選擇 Unsloth？

| 場景 | 推薦 | 原因 |
|------|------|------|
| 快速原型驗證 | **Unsloth** | 設定簡單、速度快 |
| 生產環境部署 | **Unsloth** | 一鍵 GGUF 匯出 |
| 需要最大自訂性 | **標準 PEFT** | 更細粒度控制 |
| 使用冷門模型 | **標準 PEFT** | Unsloth 可能不支援 |
| 教學/學習 | **Unsloth** | API 簡潔，容易上手 |
| 研究實驗 | **兩者皆可** | 視需求而定 |

### 16-8-3 實際效能數據

在 DGX Spark 上微調 Qwen3-8B（60 步，alpaca-zh 資料集）：

```python
# Unsloth 效能數據
unsloth_metrics = {
    "time": 60,           # 秒
    "memory_peak": 32,    # GB
    "loss": 0.80,
    "steps_per_sec": 1.0,
}

# 標準 PEFT 效能數據
peft_metrics = {
    "time": 120,          # 秒
    "memory_peak": 45,    # GB
    "loss": 0.82,
    "steps_per_sec": 0.5,
}

print(f"速度提升: {peft_metrics['time'] / unsloth_metrics['time']:.1f}x")
print(f"記憶體節省: {(1 - unsloth_metrics['memory_peak'] / peft_metrics['memory_peak']) * 100:.0f}%")
```

輸出：
```
速度提升: 2.0x
記憶體節省: 29%
```

---

## 16-9 常見問題與疑難排解

### 16-9-1 Triton 編譯錯誤

**問題**：在 ARM64 上啟動訓練時出現 Triton 相關錯誤。

**原因**：Triton 主要針對 x86_64 架構優化，ARM64 支援有限。

**解決方案**：

```python
# Unsloth 會自動偵測並迴避這個問題
# 如果仍然出現錯誤，可以手動停用 Triton 最佳化：

import os
os.environ["UNSLOTH_USE_TRITON"] = "0"

# 然後重新執行訓練
# 速度會慢一些，但仍然比標準 PEFT 快
```

::: tip 💡 確認 Unsloth 是否使用了 Triton
```python
from unsloth import is_bfloat16_supported

# 檢查 Unsloth 的最佳化狀態
import unsloth
print(f"Unsloth 版本: {unsloth.__version__}")
print(f"BF16 支援: {is_bfloat16_supported()}")

# 檢查 Triton 是否啟用
try:
    import triton
    print(f"Triton: ✅ 已安裝 ({triton.__version__})")
except ImportError:
    print("Triton: ❌ 未安裝（使用備用方案）")
```
:::

### 16-9-2 統一記憶體問題

**問題**：出現 `CUDA error: an illegal memory access was encountered` 或統一記憶體相關錯誤。

**解決方案**：

```python
# 確保所有張量在 GPU 上
model = model.to("cuda")

# 如果問題持續，停用統一記憶體
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 或者在訓練參數中設定
training_args = TrainingArguments(
    # ...
    dataloader_pin_memory=False,     # 停用記憶體釘選
)
```

### 16-9-3 xformers 版本不相容

**問題**：`xformers` 版本與 PyTorch 不相容。

**解決方案**：

```bash
# 安裝相容的 xformers 版本
uv pip install xformers==0.0.29

# 或者讓 Unsloth 自動處理（推薦）
uv pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 16-9-4 GGUF 匯出失敗

**問題**：`model.save_pretrained_gguf()` 執行失敗。

**排查步驟**：

```bash
# 1. 確認 llama.cpp 已正確安裝
python -c "from llama_cpp import llama_model_metadata; print('llama.cpp OK')"

# 2. 如果沒有安裝 llama.cpp
uv pip install llama-cpp-python

# 3. 確認有足夠的磁碟空間
df -h ./gguf-model

# 4. 嘗試不同的量化方法
model.save_pretrained_gguf(
    "./gguf-model",
    tokenizer,
    quantization_method="q4_0",  # 最簡單的量化方法
)
```

### 16-9-5 訓練 Loss 為 NaN

**問題**：訓練開始後 Loss 立刻變成 `NaN`。

**可能原因和解決方案**：

| 原因 | 解決方案 |
|------|---------|
| 學習率太高 | 降低學習率（2e-4 → 1e-4 → 5e-5） |
| 資料中有 NaN | 檢查資料集，過濾空值 |
| 梯度爆炸 | 增加 `max_grad_norm=1.0` |
| 精度問題 | 改用 BF16（`bf16=True`） |
| batch size 太小 | 增加梯度累積步數 |

```python
# 防禦性訓練設定
training_args = TrainingArguments(
    learning_rate=1e-4,              # 更保守的學習率
    max_grad_norm=1.0,               # 梯度裁剪
    bf16=True,                       # 使用 BF16
    gradient_accumulation_steps=8,   # 更大的等效 batch size
    warmup_ratio=0.1,                # 更長的預熱
)
```

### 16-9-6 Ollama 載入模型後回答品質差

**問題**：微調後的模型在 Ollama 中回答品質不如預期。

**排查步驟**：

1. **確認量化方法**：
```bash
# Q4_K_M 是品質和大小的最佳平衡
# 如果品質不佳，嘗試 Q5_K_M 或 Q6_K
model.save_pretrained_gguf(
    "./gguf-model",
    tokenizer,
    quantization_method="q5_k_m",
)
```

2. **調整 Ollama 參數**：
```bash
cat > Modelfile << 'EOF'
FROM ./gguf-model/unsloth.Q5_K_M.gguf
SYSTEM 你是一個專業的 AI 助手。
PARAMETER temperature 0.5
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
EOF
```

3. **確認提示詞格式正確**：
```python
# Unsloth 微調的模型使用與基礎模型相同的提示詞格式
# 確保推論時的格式與訓練時一致
```

---

## 16-10 本章小結

::: success ✅ 你現在知道了
- Unsloth 是最快的開源微調框架，速度比標準 PEFT 快 2 倍，記憶體用量減少 30%
- `FastLanguageModel` API 簡潔，自動啟用 Flash Attention 2 和各種最佳化
- 完整的微調流程：載入模型 → 設定 LoRA → 準備資料 → 訓練 → 推論測試
- 一鍵匯出 GGUF 格式，直接給 Ollama 使用，支援多種量化方法
- 甚至能在 DGX Spark 上微調 120B 等級的 MoE 大模型
- Loss 曲線分析是確保訓練正常的重要工具
- Unsloth 在 ARM64 上會自動使用備用方案，仍然比標準 PEFT 快
:::

::: tip 🚀 下一章預告
除了 Unsloth，還有 LLaMA Factory 和 NeMo 兩個強大的微調框架。下一章我們一次比較三個！

👉 [前往第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調 →](/guide/chapter17/)
:::

::: info 📝 上一章
← [回到第 15 章：LoRA / QLoRA 微調實戰](/guide/chapter15/)
:::
