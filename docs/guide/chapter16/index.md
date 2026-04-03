# 第 16 章：Unsloth — 最快的微調框架

::: tip 🎯 本章你將學到什麼
- Unsloth 的優勢和安裝
- QLoRA 微調 Qwen3-8B 實戰
- 模型匯出為 GGUF 給 Ollama 使用
- 微調 120B 大模型
- Unsloth vs. 標準 PEFT 比較
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。
:::

---

## 16-1 Unsloth 介紹與安裝

### 16-1-1 用 Claude Code 部署 Unsloth Docker 環境

告訴 Claude Code：

> 「用 Docker 部署 Unsloth 環境，支援 Qwen3 模型。」

```bash
docker run -d \
  --name unsloth \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/unsloth-training:/workspace \
  -w /workspace \
  unslothai/unsloth:latest
```

### 16-1-2 驗證安裝

```bash
docker exec unsloth python -c "
import unsloth
print(f'Unsloth 版本: {unsloth.__version__}')
import torch
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 16-2 QLoRA 微調實戰：Qwen3-8B

### 16-2-1 載入模型

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=4096,
    dtype=None,  # 自動選擇
    load_in_4bit=True,
)
```

::: tip 💡 Unsloth 的 FastLanguageModel
Unsloth 的 `FastLanguageModel` 會自動：
- 選擇最佳精度
- 套用 Unsloth 的最佳化
- 啟用 Flash Attention
:::

### 16-2-2 設定 LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### 16-2-3 準備訓練資料

```python
from datasets import load_dataset

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def format_samples(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        texts.append(alpaca_prompt.format(instruction, output))
    return {"text": texts}

dataset = load_dataset("shibing624/alpaca-zh", split="train")
dataset = dataset.map(format_samples, batched=True)
```

### 16-2-4 開始訓練

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="./outputs",
    ),
)

trainer.train()
```

::: tip 💡 Unsloth 的訓練速度
在 DGX Spark 上，Unsloth 的訓練速度比標準 PEFT 快約 **2 倍**，記憶體用量減少約 **30%**。
:::

---

## 16-3 推論測試

```python
FastLanguageModel.for_inference(model)

prompt = alpaca_prompt.format(
    "介紹 DGX Spark 的三個主要優點",
    "",
)
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])
```

---

## 16-4 模型匯出與部署

### 16-4-1 儲存 LoRA Adapter

```python
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")
```

### 16-4-2 合併並匯出 16-bit 格式

```python
model.save_pretrained_merged(
    "./merged-model",
    tokenizer,
    save_method="merged_16bit",
)
```

### 16-4-3 匯出 GGUF 格式給 Ollama

```python
model.save_pretrained_gguf(
    "./gguf-model",
    tokenizer,
    quantization_method="q4_k_m",
)
```

匯出完成後，可以直接用 Ollama 載入：

```bash
# 建立 Ollama Modelfile
cat > Modelfile << EOF
FROM ./gguf-model/unsloth.Q4_K_M.gguf
EOF

# 建立 Ollama 模型
ollama create my-qwen3-finetuned -f Modelfile

# 測試
ollama run my-qwen3-finetuned
```

---

## 16-5 Loss 曲線分析

```python
import matplotlib.pyplot as plt
import json

# 讀取訓練日誌
with open("./outputs/trainer_state.json") as f:
    state = json.load(f)

steps = [log["step"] for log in state["log_history"]]
losses = [log["loss"] for log in state["log_history"] if "loss" in log]

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Unsloth 訓練 Loss 曲線')
plt.grid(True)
plt.show()
```

---

## 16-6 進階：微調 120B 大模型

```python
# Unsloth 也支援微調超大模型（需要足夠記憶體）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-122B-A14B-NVFP4",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=64,
)
```

::: warning ⚠️ 記憶體需求
微調 120B 模型需要約 80-100 GB 記憶體。確保沒有其他大型程式在執行。
:::

---

## 16-7 NVIDIA 官方 Unsloth Playbook

```bash
ngc registry resource download-version "nvidia-ai-workbench/dgx-spark-unsloth:latest"
```

---

## 16-8 Unsloth vs. 標準 PEFT 比較

| 特性 | Unsloth | 標準 PEFT |
|------|---------|----------|
| 訓練速度 | **2x 快** | 基準 |
| 記憶體用量 | **30% 少** | 基準 |
| 支援模型 | 主流模型 | 所有模型 |
| Flash Attention | 自動啟用 | 需手動設定 |
| 4-bit 訓練 | ✅ 最佳化 | ✅ |
| GGUF 匯出 | ✅ 一鍵 | 需手動轉換 |
| 學習曲線 | 低 | 中等 |

---

## 16-9 常見問題與疑難排解

### 16-9-1 Triton 編譯錯誤

在 ARM64 上，Triton 可能不相容。Unsloth 會自動迴避這個問題。

### 16-9-2 統一記憶體問題

如果出現統一記憶體相關錯誤：

```python
# 確保所有張量在 GPU 上
model = model.to("cuda")
```

### 16-9-3 xformers 版本不相容

```bash
# 安裝正確的 xformers 版本
uv pip install xformers==0.0.29
```

---

## 16-10 本章小結

::: success ✅ 你現在知道了
- Unsloth 是最快的開源微調框架，速度比標準 PEFT 快 2 倍
- 可以一鍵匯出 GGUF 格式，直接給 Ollama 使用
- 甚至能微調 120B 等級的大模型
:::

::: tip 🚀 下一章預告
除了 Unsloth，還有 LLaMA Factory 和 NeMo 兩個強大的微調框架。下一章我們一次比較三個！

👉 [前往第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調 →](/guide/chapter17/)
:::

::: info 📝 上一章
← [回到第 15 章：LoRA / QLoRA 微調實戰](/guide/chapter15/)
:::
