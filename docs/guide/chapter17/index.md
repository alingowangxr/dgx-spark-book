# 第 17 章：LLaMA Factory、NeMo 與 PyTorch 微調

::: tip 🎯 本章你將學到什麼
- LLaMA Factory 的 Web UI 和 CLI 微調完整流程
- NeMo AutoModel 的 LoRA、QLoRA 和全參數微調
- PyTorch 原生微調的完整程式碼與最佳實踐
- 三大框架的詳細比較與選擇指南
- 常見問題的完整疑難排解
:::

---

## 17-1 LLaMA Factory

LLaMA Factory 是目前最受歡迎的開源大語言模型微調框架之一。它由 LLaMA-Factory 團隊開發，支援超過 100 種模型，並且提供了 **Web UI** 和 **CLI** 兩種操作方式。對於初學者來說，它是入門微調的最佳選擇。

::: info 🤔 為什麼選擇 LLaMA Factory？
- **零程式碼門檻**：Web UI 讓你用滑鼠點一點就能完成微調
- **支援模型最多**：涵蓋 LLaMA、Qwen、Mistral、Gemma、Yi 等主流模型
- **微調方式齊全**：支援 LoRA、QLoRA、DoRA、全參數微調
- **社群活躍**：GitHub 上有超過 30,000 顆星，問題回覆速度快
:::

### 17-1-1 安裝 LLaMA Factory

在 DGX Spark 上安裝 LLaMA Factory 非常簡單。我們使用 `uv` 來管理 Python 環境，它比傳統的 pip 快 10-100 倍。

```bash
# 1. 克隆原始碼
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 建立虛擬環境
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. 安裝依賴（包含 PyTorch 和評估指標）
uv pip install -e ".[torch,metrics]"
```

::: tip 💡 安裝加速技巧
如果你在中國大陸，可以設定 pip 鏡像加速：

```bash
uv pip install -e ".[torch,metrics]" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
:::

::: warning ⚠️ 常見安裝問題
- **PyTorch CUDA 版本不符**：確保你的 PyTorch CUDA 版本與系統驅動匹配
  ```bash
  nvidia-smi  # 查看 CUDA 版本
  uv pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **記憶體不足**：安裝過程可能需要 5-10 GB 磁碟空間，請確保有足夠空間
:::

安裝完成後，驗證是否成功：

```bash
llamafactory-cli version
# 應該顯示版本號，例如 0.9.2
```

### 17-1-2 了解 LLaMA Factory 的設定架構

LLaMA Factory 使用 YAML 檔案來管理所有訓練設定。理解這些參數的意義是成功微調的關鍵。

以下是一個完整的 YAML 設定檔，每個參數都有詳細註解：

```yaml
# examples/train_lora.yaml — 完整設定說明

# === 模型設定 ===
model_name_or_path: Qwen/Qwen3-8B    # 基礎模型路徑（Hugging Face ID 或本地路徑）
adapter_name_or_path: null           # 已有 LoRA adapter 路徑（可選，用於繼續訓練）
template: qwen                       # 對話模板，必須與模型匹配
flash_attn: auto                     # 使用 Flash Attention 加速（auto/true/false）

# === 訓練模式 ===
stage: sft                           # sft=監督微調, rm=獎勵模型, ppo=強化學習, dpo=偏好優化
do_train: true                       # true=訓練, false=僅評估
finetuning_type: lora                # lora/qlora/full/freeze

# === LoRA 設定 ===
lora_target: all                     # all=所有線性層, 或指定 "q_proj,v_proj"
lora_rank: 8                         # LoRA 矩陣的秩（越大效果越好但記憶體越多）
lora_alpha: 16                       # LoRA 縮放係數（通常設為 rank 的 2 倍）
lora_dropout: 0.1                    # LoRA Dropout 率，防止過擬合

# === 資料集設定 ===
dataset: alpaca-zh                   # 資料集名稱（在 data/dataset_info.json 中定義）
dataset_dir: data                    # 資料集目錄
cutoff_len: 2048                     # 最大序列長度（超過會被截斷）
max_samples: null                    # 最大使用樣本數（null=全部使用）
overwrite_cache: false               # 是否覆蓋已快取的資料

# === 訓練超參數 ===
output_dir: saves/qwen3-8b/lora      # 模型儲存路徑
per_device_train_batch_size: 2       # 每張 GPU 的 batch size
gradient_accumulation_steps: 4       # 梯度累積步數（等效 batch_size = 2×4 = 8）
learning_rate: 2.0e-4                # 學習率（LoRA 通常用 1e-4 ~ 5e-4）
num_train_epochs: 3.0                # 訓練輪數
lr_scheduler_type: cosine            # 學習率調度器（cosine/linear/polynomial）
warmup_ratio: 0.1                    # 學習率預熱比例（前 10% 步數線性增加）
weight_decay: 0.01                   # 權重衰減，防止過擬合
max_grad_norm: 1.0                   # 梯度裁剪，防止梯度爆炸

# === 精度與效能 ===
bf16: true                           # 使用 BF16 混合精度訓練
tf32: true                           # 使用 TF32 加速（Ampere 架構以上支援）
ddp_timeout: 180000000               # 分散式訓練超時時間（毫秒）

# === 記錄與評估 ===
logging_steps: 10                    # 每隔多少步記錄一次日誌
save_steps: 500                      # 每隔多少步儲存 checkpoint
eval_steps: 500                      # 每隔多少步評估一次
report_to: tensorboard               # 記錄工具（tensorboard/wandb/none）
```

::: info 🤔 梯度累積是什麼？
假設你的 GPU 記憶體只能容納 batch_size=2，但你想用 batch_size=8 的效果。

設定 `gradient_accumulation_steps=4`，模型會：
1. 跑 4 次 forward pass（每次 batch=2）
2. 把 4 次的梯度加起來
3. 才執行一次 optimizer.step()

這樣等效 batch_size = 2 × 4 = 8，但記憶體用量只有 batch=2 的大小！
:::

### 17-1-3 準備自訂資料集

LLaMA Factory 支援多種資料格式。最常用的是 **Alpaca 格式**：

```json
[
  {
    "instruction": "解釋什麼是 DGX Spark",
    "input": "",
    "output": "DGX Spark 是 NVIDIA 推出的個人 AI 超級電腦，搭載 Grace Blackwell 架構..."
  },
  {
    "instruction": "將以下文字翻譯成英文",
    "input": "人工智慧正在改變世界",
    "output": "Artificial intelligence is changing the world."
  }
]
```

將資料儲存為 `data/my_dataset.json`，然後在 `data/dataset_info.json` 中註冊：

```json
{
  "my-dataset": {
    "file_name": "my_dataset.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

::: tip 💡 資料集品質建議
- **數量**：LoRA 微調建議至少 500-1000 筆高品質資料
- **多樣性**：涵蓋不同類型的任務（問答、翻譯、摘要等）
- **長度**：每筆資料的 output 建議 50-500 字
- **品質 > 數量**：100 筆高品質資料勝過 1000 筆低品質資料
:::

### 17-1-4 使用 CLI 執行 LoRA 微調

設定好 YAML 和資料集後，一行指令就能開始訓練：

```bash
llamafactory-cli train examples/train_lora.yaml
```

訓練過程中，你會看到類似以下的輸出：

```
{'loss': 1.2345, 'grad_norm': 0.567, 'learning_rate': 1.8e-4, 'epoch': 0.5}
{'loss': 0.9876, 'grad_norm': 0.432, 'learning_rate': 1.5e-4, 'epoch': 1.0}
{'loss': 0.7654, 'grad_norm': 0.321, 'learning_rate': 1.0e-4, 'epoch': 1.5}
...
```

::: info 🤔 如何解讀訓練日誌？
- **loss**：損失值，越低越好。正常情況下應該持續下降
- **grad_norm**：梯度範數，如果突然暴增表示訓練不穩定
- **learning_rate**：當前學習率，會根據 scheduler 變化
- **epoch**：訓練輪數，1.0 表示完整跑過一次資料集

如果 loss 沒有下降，可能的原因：
1. 學習率太高（嘗試降低 10 倍）
2. 資料格式有問題
3. 模板（template）設定錯誤
:::

### 17-1-5 Web UI 微調 — 圖形化操作

如果你不喜歡命令列，LLaMA Factory 提供了完整的 Web UI：

```bash
llamafactory-cli webui --host 0.0.0.0 --port 7860
```

打開瀏覽器訪問 `http://DGX_Spark_IP:7860`，你會看到以下介面：

**步驟 1：選擇模型**
- 在「模型名稱」輸入 `Qwen/Qwen3-8B`
- 選擇微調類型：LoRA / QLoRA / 全參數
- 設定模板：qwen

**步驟 2：設定訓練參數**
- 學習率：`2e-4`（LoRA 推薦值）
- Batch Size：`2`
- 梯度累積：`4`
- 訓練輪數：`3`
- 最大長度：`2048`

**步驟 3：選擇資料集**
- 勾選 `alpaca-zh` 或你的自訂資料集
- 預覽資料確認格式正確

**步驟 4：開始訓練**
- 點擊「開始訓練」按鈕
- 即時監控 loss 曲線和 GPU 使用率

::: tip 💡 Web UI 進階功能
- **即時預覽**：訓練中可以直接測試模型生成效果
- **TensorBoard 整合**：點擊「TensorBoard」按鈕即時查看訓練曲線
- **多模型比較**：可以同時比較不同參數的訓練結果
:::

### 17-1-6 QLoRA 微調 — 用更少記憶體訓練大模型

QLoRA = Quantized LoRA，透過 4-bit 量化讓大模型能在有限記憶體下訓練：

```yaml
# examples/train_qlora.yaml
model_name_or_path: Qwen/Qwen3-70B
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
quantization_bit: 4                  # 啟用 4-bit 量化
dataset: alpaca-zh
template: qwen
output_dir: saves/qwen3-70b/qlora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2.0
bf16: true
```

```bash
llamafactory-cli train examples/train_qlora.yaml
```

::: info 🤔 QLoRA vs LoRA 記憶體比較
| 模型 | LoRA 記憶體 | QLoRA 記憶體 |
|------|------------|-------------|
| Qwen3-8B | ~20 GB | ~8 GB |
| Qwen3-70B | ~140 GB（DGX Spark 不夠） | ~45 GB（DGX Spark 可跑） |
| Llama-3.1-70B | ~140 GB | ~42 GB |

QLoRA 讓 DGX Spark 也能微調 70B 等級的模型！
:::

### 17-1-7 DPO 偏好優化微調

除了 SFT（監督微調），LLaMA Factory 還支援 DPO（Direct Preference Optimization），用於對齊人類偏好：

```yaml
# examples/train_dpo.yaml
model_name_or_path: Qwen/Qwen3-8B
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
dataset: dpo-zh
template: qwen
output_dir: saves/qwen3-8b/dpo
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
bf16: true
```

DPO 資料格式需要「選擇」和「拒絕」兩種回答：

```json
[
  {
    "instruction": "DGX Spark 適合做什麼？",
    "input": "",
    "chosen": "DGX Spark 適合個人開發者進行 AI 模型微調、推理和實驗...",
    "rejected": "DGX Spark 是一台電腦。"
  }
]
```

### 17-1-8 驗證與模型匯出

訓練完成後，你需要驗證效果並匯出模型：

```bash
# 1. 使用 CLI 進行推理測試
llamafactory-cli chat \
  --model_name_or_path Qwen/Qwen3-8B \
  --adapter_name_or_path saves/qwen3-8b/lora \
  --template qwen
```

進入互動式對話模式後，輸入你的測試問題：

```
使用者：請介紹 DGX Spark 的主要特點
AI：DGX Spark 是 NVIDIA 推出的個人 AI 超級電腦，具有以下特點：
    1. 搭載 Grace Blackwell 架構
    2. 128GB 統一記憶體
    3. 支援本地端 AI 推理和微調
    ...
```

```bash
# 2. 匯出模型（將 LoRA 權重合併到基礎模型）
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-8B \
  --adapter_name_or_path saves/qwen3-8b/lora \
  --template qwen \
  --export_dir ./exported-model \
  --export_size 2 \
  --export_legacy_format false
```

```bash
# 3. 匯出後的模型可以獨立使用
ls ./exported-model/
# config.json  model-00001-of-00003.safetensors  tokenizer.json  ...
```

::: tip 💡 是否需要匯出？
- **不需要匯出**：如果你只是自己使用，直接載入 adapter 即可（節省空間）
- **需要匯出**：如果要部署到生產環境或分享給他人，建議匯出
:::

---

## 17-2 NeMo AutoModel

NVIDIA NeMo 是 NVIDIA 官方的企業級大語言模型框架。NeMo AutoModel 提供了高階的 Python API，讓你可以用幾行程式碼完成複雜的微調任務。

::: info 🤔 為什麼選擇 NeMo？
- **NVIDIA 官方支援**：與 GPU 硬體深度整合，效能最佳化
- **企業級功能**：支援 Megatron-LM 分散式訓練、張量並行、管線並行
- **完整生態系**：與 TensorRT-LLM、Triton Inference Server 無縫整合
- **多模態支援**：不僅支援 LLM，還支援語音、視覺模型
:::

### 17-2-1 部署 NeMo AutoModel 容器

NeMo 透過 Docker 容器分發，確保環境一致性：

```bash
# 1. 確保已安裝 NVIDIA Container Toolkit
# 檢查是否正確安裝
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

```bash
# 2. 拉取 NeMo 容器映像（約 20-30 GB，首次下載需要一些時間）
docker pull nvcr.io/nvidia/nemo:25.01

# 3. 啟動容器
docker run -d \
  --name nemo \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/nemo-training:/workspace \
  nvcr.io/nvidia/nemo:25.01

# 4. 進入容器
docker exec -it nemo bash
```

::: warning ⚠️ 容器空間注意事項
- NeMo 容器約 20-30 GB，確保磁碟有至少 50 GB 可用空間
- 訓練產生的 checkpoint 也會佔用空間，建議預留 100 GB
- 使用 `docker system prune` 定期清理不需要的映像
:::

### 17-2-2 LoRA 微調 Llama 3.1 8B

NeMo AutoModel 的 API 設計非常直觀：

```python
from nemo.collections.llm import llama3_8b, lora

# 1. 載入預訓練模型
model = llama3_8b.model()

# 2. 新增 LoRA adapter
model.add_adapter(lora.LoraAdapter(
    target_modules=["qkv", "fc1", "fc2"],  # 要微調的層
    rank=16,                                # LoRA rank
    alpha=32,                               # 縮放係數（通常 = rank × 2）
    dropout=0.1,                            # Dropout 率
))

# 3. 設定訓練參數並開始訓練
model.train(
    dataset="alpaca-zh",                    # 資料集
    num_epochs=3,                           # 訓練輪數
    batch_size=2,                           # batch size
    learning_rate=2e-4,                     # 學習率
    gradient_accumulation_steps=4,          # 梯度累積
    output_dir="./nemo-lora-output",        # 輸出目錄
)
```

::: info 🤔 target_modules 是什麼？
LoRA 不會微調模型的所有參數，而是選擇性地微調特定的線性層：

| target_modules | 說明 | 記憶體用量 | 效果 |
|---------------|------|-----------|------|
| `["qkv"]` | 只微調注意力層 | 最低 | 基礎 |
| `["qkv", "fc1", "fc2"]` | 注意力 + FFN | 中等 | **推薦** |
| `["all"]` | 所有線性層 | 最高 | 最好 |

在 DGX Spark 上，建議使用 `["qkv", "fc1", "fc2"]` 取得效果與資源的最佳平衡。
:::

### 17-2-3 QLoRA 微調 70B 模型

QLoRA 讓 DGX Spark 也能微調 70B 等級的模型：

```python
from nemo.collections.llm import llama3_70b, qlora

# 1. 載入 4-bit 量化的 70B 模型
model = llama3_70b.model(quantization="nf4")

# 2. 新增 QLoRA adapter
model.add_adapter(qlora.QLoRAAdapter(
    rank=32,                              # QLoRA 建議用較大的 rank
    alpha=64,
    target_modules=["qkv", "fc1", "fc2"],
))

# 3. 開始訓練
model.train(
    dataset="custom",                     # 使用自訂資料集
    num_epochs=1,                         # 70B 模型通常 1-2 輪就夠
    batch_size=1,                         # 70B 模型用 batch_size=1
    learning_rate=1e-4,
    gradient_accumulation_steps=8,
    output_dir="./nemo-qlora-70b",
)
```

::: tip 💡 QLoRA 訓練技巧
- 70B 模型的 QLoRA 訓練在 DGX Spark 上約需 40-50 GB 記憶體
- 建議使用 `gradient_accumulation_steps=8` 來模擬更大的 batch size
- 學習率要比 LoRA 更低（1e-4 vs 2e-4），因為量化模型的梯度更不穩定
:::

### 17-2-4 全參數微調 Qwen3-8B

如果你需要最佳效果，可以進行全參數微調：

```python
from nemo.collections.llm import qwen3_8b

# 1. 載入模型（不新增 adapter）
model = qwen3_8b.model()

# 2. 全參數微調
model.train(
    dataset="alpaca-zh",
    num_epochs=3,
    batch_size=1,
    learning_rate=1e-5,                   # 全參數微調用更低的學習率
    gradient_accumulation_steps=8,
    full_finetune=True,                   # 啟用全參數微調
    output_dir="./nemo-full-sft",
)
```

::: warning ⚠️ 全參數微調注意事項
- 8B 模型全參數微調約需 60-80 GB 記憶體，DGX Spark 的 128GB 足夠
- 學習率必須比 LoRA 低 10-20 倍（1e-5 vs 2e-4）
- 訓練時間會比 LoRA 長 3-5 倍
- 建議先用 LoRA 實驗，確認效果後再用全參數微調
:::

### 17-2-5 使用自訂資料集

NeMo 支援多種資料格式，以下是 JSONL 格式的範例：

```python
# 準備資料（JSONL 格式，每行一個 JSON 物件）
# data/train.jsonl
{"input": "解釋什麼是 DGX Spark", "output": "DGX Spark 是 NVIDIA 推出的個人 AI 超級電腦..."}
{"input": "Python 的 list comprehension 是什麼", "output": "List comprehension 是 Python 的一種簡潔語法..."}
```

```python
from nemo.collections.llm import qwen3_8b, lora
from nemo.collections.llm.data import JSONLDataset

# 載入自訂資料集
dataset = JSONLDataset(
    data_path="/workspace/data/train.jsonl",
    seq_length=2048,
)

model = qwen3_8b.model()
model.add_adapter(lora.LoraAdapter(rank=16, alpha=32))

model.train(
    dataset=dataset,
    num_epochs=3,
    batch_size=2,
    output_dir="./nemo-custom-data",
)
```

### 17-2-6 驗證訓練結果

訓練完成後，進行推理測試：

```python
# 基本推理
result = model.infer("介紹 DGX Spark")
print(result)

# 批次推理
prompts = [
    "什麼是人工智慧？",
    "Python 和 C++ 的區別是什麼？",
    "解釋深度學習的基本概念",
]
results = model.infer(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}")
    print("-" * 50)
```

### 17-2-7 模型匯出與部署

```python
# 匯出為 Hugging Face 格式
model.export(
    export_path="./exported-nemo-model",
    export_format="hf",
)

# 匯出為 TensorRT-LLM 格式（用於高效能推理部署）
model.export(
    export_path="./tensorrt-model",
    export_format="tensorrt_llm",
)
```

---

## 17-3 PyTorch 原生微調

PyTorch 原生微調給研究者最大的控制權。雖然程式碼較多，但你可以精確控制訓練的每個環節。

### 17-3-1 環境設定

```bash
# 使用 uv 快速安裝
uv pip install torch transformers datasets accelerate peft bitsandbytes

# 驗證安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 17-3-2 完整的資料載入流程

在開始訓練之前，我們需要先準備資料：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 載入資料集
dataset = load_dataset("json", data_files="my_dataset.json", split="train")

# 2. 載入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer.pad_token = tokenizer.eos_token  # 設定 pad token

# 3. 定義格式化函數
def format_prompt(example):
    """將資料轉換為模型所需的格式"""
    prompt = f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    return prompt

# 4. Tokenize
def tokenize_function(examples):
    prompts = [format_prompt({"instruction": i, "output": o})
               for i, o in zip(examples["instruction"], examples["output"])]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=2048,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # SFT 的 labels 等於 input_ids
    return tokenized

# 5. 處理資料集
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# 6. 分割訓練/驗證集
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
```

::: info 🤔 為什麼 labels 等於 input_ids？
在 SFT（監督微調）中，我們希望模型學習生成正確的 output。
將 labels 設為 input_ids 的副本，Trainer 會自動計算每個 token 的交叉熵損失。
有些實作會將 instruction 部分的 labels 設為 -100（忽略），只計算 output 部分的損失。
:::

### 17-3-3 全參數微調 3B 模型（Full SFT）

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# 1. 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-3B",
    torch_dtype=torch.bfloat16,    # 使用 BF16 節省記憶體
    device_map="auto",             # 自動分配到可用 GPU
)

# 2. 訓練參數
training_args = TrainingArguments(
    output_dir="./full-sft-qwen3-3b",
    num_train_epochs=3,
    per_device_train_batch_size=4,         # DGX Spark 可以用較大的 batch size
    gradient_accumulation_steps=4,
    learning_rate=1e-5,                    # 全參數微調用低學習率
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    tf32=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,                    # 只保留最近 3 個 checkpoint
    eval_strategy="steps",
    eval_steps=500,
    report_to="tensorboard",
    gradient_checkpointing=True,           # 啟用梯度檢查點節省記憶體
    optim="adamw_torch_fused",             # 使用融合最佳化器加速
)

# 3. Data Collator（負責動態 padding）
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    max_length=2048,
)

# 4. 建立 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)

# 5. 開始訓練
trainer.train()

# 6. 儲存模型
trainer.save_model("./full-sft-qwen3-3b/final")
```

### 17-3-4 LoRA 微調 70B 模型

使用 PEFT（Parameter-Efficient Fine-Tuning）庫來實現 LoRA：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 載入模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-70B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. 設定 LoRA 參數
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          # 因果語言模型任務
    r=32,                                  # LoRA rank
    lora_alpha=64,                         # 縮放係數
    lora_dropout=0.1,
    target_modules=[                       # 要微調的模組
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",                           # 不訓練 bias
)

# 3. 套用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 85,196,800 || all params: 70,000,000,000 || trainable%: 0.12%
# 只訓練 0.12% 的參數！

# 4. 訓練（使用與上面相同的 Trainer 設定）
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./lora-qwen3-70b",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,                # LoRA 可以用較高學習率
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
    ),
    train_dataset=split_dataset["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()
trainer.save_model("./lora-qwen3-70b/final")
```

::: tip 💡 LoRA 參數選擇指南
| 資料量 | Rank | Alpha | 學習率 |
|--------|------|-------|--------|
| < 1,000 筆 | 8 | 16 | 5e-4 |
| 1,000-10,000 筆 | 16 | 32 | 2e-4 |
| > 10,000 筆 | 32 | 64 | 1e-4 |
:::

### 17-3-5 QLoRA 微調 Llama 3.1 70B

使用 BitsAndBytes 進行 4-bit 量化：

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# 1. 設定量化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 啟用 4-bit 量化
    bnb_4bit_quant_type="nf4",             # NF4 量化（Normal Float 4-bit）
    bnb_4bit_compute_dtype=torch.bfloat16, # 計算時用 BF16
    bnb_4bit_use_double_quant=True,        # 雙重量化，進一步節省記憶體
)

# 2. 載入量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. 設定 QLoRA（與 LoRA 相同，但 rank 可以更大）
lora_config = LoraConfig(
    r=64,                                  # QLoRA 可以用更大的 rank
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 訓練（同上）
```

::: info 🤔 NF4 量化是什麼？
NF4（Normal Float 4-bit）是一種專門為神經網路權重設計的量化方法：
- 假設權重分佈接近常態分佈
- 使用資訊理論最佳的分佈感知量化
- 比傳統的 4-bit 量化效果更好
- 雙重量化（double quant）可以額外節省約 0.4 bits/param
:::

### 17-3-6 PyTorch 官方的 DGX Spark 全參數微調

NVIDIA 和 PyTorch 團隊合作推出了 DGX Spark 專用的全參數微調範例：

```bash
# 1. 克隆 PyTorch examples 倉庫
git clone https://github.com/pytorch/examples.git
cd examples/dgx-spark

# 2. 安裝依賴
uv pip install -r requirements.txt

# 3. 執行全參數微調
python full_finetune.py \
  --model qwen3-8b \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-5 \
  --bf16

# 4. 執行推理測試
python inference.py \
  --model-path ./output/qwen3-8b-finetuned \
  --prompt "介紹 DGX Spark"
```

這個範例程式碼針對 DGX Spark 的硬體進行了最佳化：
- 使用 `torch.compile()` 加速
- 啟用 Flash Attention 2
- 最佳化的 DataLoader 設定
- 自動記憶體管理

---

## 17-4 三大框架比較

### 17-4-1 功能比較表

| 特性 | LLaMA Factory | NeMo | PyTorch 原生 |
|------|--------------|------|-------------|
| 學習難度 | **最低**（Web UI） | 中等 | 最高 |
| Web UI | ✅ 完整 | ❌ | ❌ |
| 支援模型數量 | **最多**（100+） | NVIDIA 模型 | 所有 HF 模型 |
| 微調方式 | LoRA/QLoRA/DPO/Full | LoRA/QLoRA/Full | 全部支援 |
| 分散式訓練 | ✅（DeepSpeed） | ✅ **最強**（Megatron） | ✅（需自行設定） |
| 企業級功能 | 有限 | **完整** | 需自行實作 |
| 記憶體效率 | 高 | **最高** | 中等 |
| 社群支援 | 活躍（GitHub 30k+ stars） | NVIDIA 官方 | 最大社群 |
| 文件品質 | 良好 | 良好 | **最佳** |
| 適合對象 | 初學者、快速實驗 | 企業、大規模訓練 | 研究者、完全控制 |

### 17-4-2 選擇指南

::: tip 💡 如何選擇框架？

**選擇 LLaMA Factory 如果：**
- 你是初學者，第一次做微調
- 需要快速實驗不同模型和參數
- 喜歡用 Web UI 操作
- 不想寫太多程式碼

**選擇 NeMo 如果：**
- 你在企業環境中使用
- 需要分散式訓練（多 GPU/多節點）
- 需要與 NVIDIA 生態系整合
- 需要生產等級的穩定性

**選擇 PyTorch 原生如果：**
- 你是研究者，需要完全控制
- 需要自訂損失函數或訓練流程
- 想深入了解微調的底層原理
- 需要最大的靈活性
:::

### 17-4-3 效能比較

在 DGX Spark 上的實際訓練時間比較（Qwen3-8B，1000 筆資料，3 epochs）：

| 框架 | 微調方式 | 訓練時間 | 記憶體峰值 |
|------|---------|---------|-----------|
| LLaMA Factory | LoRA | ~45 分鐘 | ~20 GB |
| LLaMA Factory | QLoRA | ~60 分鐘 | ~10 GB |
| NeMo | LoRA | ~40 分鐘 | ~18 GB |
| NeMo | Full | ~2.5 小時 | ~70 GB |
| PyTorch 原生 | LoRA | ~50 分鐘 | ~22 GB |
| PyTorch 原生 | Full | ~3 小時 | ~75 GB |

---

## 17-5 常見問題與疑難排解

### 17-5-1 CUDA Out of Memory

這是最常見的問題。以下是完整的解決方案：

```python
# 方案 1：降低 batch size 並增加梯度累積
per_device_train_batch_size = 1        # 從 2 降到 1
gradient_accumulation_steps = 8        # 從 4 增加到 8

# 方案 2：啟用梯度檢查點（用時間換空間）
training_args.gradient_checkpointing = True

# 方案 3：使用 QLoRA
finetuning_type = "qlora"
quantization_bit = 4

# 方案 4：縮短序列長度
cutoff_len = 1024                      # 從 2048 降到 1024

# 方案 5：減少 LoRA target modules
lora_target = ["q_proj", "v_proj"]     # 從 all 減少到只有注意力層
```

::: tip 💡 記憶體監控
訓練時用另一個終端機監控記憶體：
```bash
watch -n 1 nvidia-smi
```
如果記憶體持續增長不釋放，可能是記憶體洩漏，嘗試設定 `gradient_checkpointing=True`。
:::

### 17-5-2 Hugging Face 模型下載失敗

```bash
# 方案 1：設定鏡像（中國大陸）
export HF_ENDPOINT=https://hf-mirror.com

# 方案 2：設定代理
export HTTPS_PROXY=http://your-proxy:port

# 方案 3：手動下載後使用本地路徑
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b

# 然後在設定中使用本地路徑
model_name_or_path: ./models/qwen3-8b
```

### 17-5-3 LLaMA Factory Web UI 無法存取

```bash
# 1. 確認 port 是否被佔用
lsof -i :7860

# 2. 如果被佔用，換一個 port
llamafactory-cli webui --port 7861

# 3. 確認防火牆設定
sudo ufw allow 7860

# 4. 確認綁定地址（必須是 0.0.0.0 才能從外部存取）
llamafactory-cli webui --host 0.0.0.0 --port 7860
```

### 17-5-4 NeMo AutoModel 容器映像太大

NeMo 容器約 20-30 GB。以下是管理建議：

```bash
# 1. 檢查磁碟使用
df -h
docker system df

# 2. 清理不需要的映像
docker system prune -a

# 3. 只保留必要的容器
docker rm $(docker ps -a -q -f status=exited)
```

### 17-5-5 訓練 loss 不下降

```
可能原因與解決方案：

1. 學習率太高 → 降低 10 倍（2e-4 → 2e-5）
2. 資料格式錯誤 → 檢查 template 是否正確
3. 資料量太少 → 至少需要 500 筆
4. 模型與 template 不匹配 → 確認 template 設定
5. 梯度爆炸 → 設定 max_grad_norm=1.0
```

### 17-5-6 訓練後模型效果不佳

```
檢查清單：
□ 訓練資料是否涵蓋測試場景？
□ 訓練輪數是否足夠？（嘗試增加 epochs）
□ 是否過擬合？（檢查驗證集 loss）
□ LoRA rank 是否太小？（嘗試增加到 16 或 32）
□ 推理時的 prompt 格式是否與訓練時一致？
```

---

## 17-6 本章小結

::: success ✅ 你現在知道了
- LLaMA Factory 最適合快速實驗，有完整的 Web UI，支援 100+ 種模型
- NeMo 是企業級方案，分散式訓練功能最完整，與 NVIDIA 生態系深度整合
- PyTorch 原生微調給研究者最大的控制權，適合自訂訓練流程
- LoRA 是記憶體效率最高的微調方式，QLoRA 讓 70B 模型也能在 DGX Spark 上訓練
- 三大框架各有優劣，選擇時需考慮學習成本、功能需求和部署場景
:::

::: tip 🚀 下一章預告
文字模型微調完了，接下來來看看怎麼微調圖片生成模型 — FLUX Dreambooth LoRA！

👉 [前往第 18 章：影像模型微調 →](/guide/chapter18/)
:::

::: info 📝 上一章
← [回到第 16 章：Unsloth](/guide/chapter16/)
:::
