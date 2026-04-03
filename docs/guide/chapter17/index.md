# 第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調

::: tip 🎯 本章你將學到什麼
- LLaMA Factory 的 Web UI 和 CLI 微調
- NeMo AutoModel 的 LoRA 和 QLoRA
- PyTorch 原生微調
- 三大框架比較
:::

---

## 17-1 LLaMA Factory

### 17-1-1 安裝 LLaMA Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

uv venv .venv
source .venv/bin/activate
uv pip install -e ".[torch,metrics]"
```

### 17-1-2 查看 YAML 設定檔

LLaMA Factory 用 YAML 管理訓練設定：

```yaml
# examples/train_lora.yaml
model_name_or_path: Qwen/Qwen3-8B
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset: alpaca-zh
template: qwen
output_dir: saves/qwen3-8b/lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3.0
bf16: true
```

### 17-1-3 使用 CLI 執行 LoRA 微調

```bash
llamafactory-cli train examples/train_lora.yaml
```

### 17-1-4 Web UI 微調

```bash
llamafactory-cli webui
```

打開瀏覽器到 `http://DGX_Spark_IP:7860`，用圖形介面設定訓練參數。

### 17-1-5 驗證與模型匯出

```bash
# 匯出模型
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-8B \
  --adapter_name_or_path saves/qwen3-8b/lora \
  --template qwen \
  --export_dir ./exported-model
```

---

## 17-2 NeMo AutoModel

### 17-2-1 部署 NeMo AutoModel 容器

```bash
docker run -d \
  --name nemo \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/nemo-training:/workspace \
  nvcr.io/nvidia/nemo:25.01
```

### 17-2-2 LoRA 微調 Llama 3.1 8B

```python
from nemo.collections.llm import llama3_8b, lora

model = llama3_8b.model()
model.add_adapter(lora.LoraAdapter(
    target_modules=["qkv", "fc1", "fc2"],
    rank=16,
    alpha=32,
))

model.train(
    dataset="alpaca-zh",
    num_epochs=3,
    batch_size=2,
)
```

### 17-2-3 QLoRA 微調 70B 模型

```python
from nemo.collections.llm import llama3_70b, qlora

model = llama3_70b.model(quantization="nf4")
model.add_adapter(qlora.QLoRAAdapter(rank=32))
model.train(dataset="custom", num_epochs=1)
```

### 17-2-4 全參數微調 Qwen3-8B

```python
from nemo.collections.llm import qwen3_8b

model = qwen3_8b.model()
model.train(
    dataset="alpaca-zh",
    num_epochs=3,
    batch_size=1,
    full_finetune=True,
)
```

### 17-2-5 驗證訓練結果

```python
result = model.infer("介紹 DGX Spark")
print(result)
```

---

## 17-3 PyTorch 原生微調

### 17-3-1 部署環境與共用元件

```bash
uv pip install torch transformers datasets accelerate
```

### 17-3-2 全參數微調 3B 模型（Full SFT）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-3B", torch_dtype=torch.bfloat16)
model.to("cuda")

training_args = TrainingArguments(
    output_dir="./full-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

### 17-3-3 LoRA 微調 70B 模型

```python
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-70B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

### 17-3-4 QLoRA 微調 Llama 3.1 70B

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 17-3-5 PyTorch 官方的 DGX Spark 全參數微調

NVIDIA 和 PyTorch 團隊合作推出了 DGX Spark 專用的全參數微調範例：

```bash
git clone https://github.com/pytorch/examples.git
cd examples/dgx-spark
python full_finetune.py --model qwen3-8b --epochs 3
```

---

## 17-4 三大框架比較

| 特性 | LLaMA Factory | NeMo | PyTorch 原生 |
|------|--------------|------|-------------|
| 學習難度 | **最低** | 中等 | 最高 |
| Web UI | ✅ | ❌ | ❌ |
| 支援模型 | 最多 | NVIDIA 模型 | 所有 |
| 分散式訓練 | ✅ | ✅ 最強 | ✅ |
| 企業級功能 | 有限 | **完整** | 需自行實作 |
| 適合對象 | 初學者、快速實驗 | 企業、大規模訓練 | 研究者、完全控制 |

---

## 17-5 常見問題與疑難排解

### 17-5-1 CUDA out of memory

```python
# 降低 batch size
per_device_train_batch_size=1

# 增加 gradient accumulation
gradient_accumulation_steps=8

# 或改用 QLoRA
```

### 17-5-2 Hugging Face 模型下載失敗

```bash
# 設定鏡像
export HF_ENDPOINT=https://hf-mirror.com
```

### 17-5-3 LLaMA Factory Web UI 無法存取

```bash
# 確認 port
lsof -i :7860

# 換 port
llamafactory-cli webui --port 7861
```

### 17-5-4 NeMo AutoModel 容器映像太大

NeMo 容器約 20-30 GB。確保有足夠磁碟空間。

---

## 17-6 本章小結

::: success ✅ 你現在知道了
- LLaMA Factory 最適合快速實驗，有 Web UI
- NeMo 是企業級方案，功能最完整
- PyTorch 原生微調給研究者最大的控制權
:::

::: tip 🚀 下一章預告
文字模型微調完了，接下來來看看怎麼微調圖片生成模型 — FLUX Dreambooth LoRA！

👉 [前往第 18 章：影像模型微調 →](/guide/chapter18/)
:::

::: info 📝 上一章
← [回到第 16 章：Unsloth](/guide/chapter16/)
:::
