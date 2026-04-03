# 第 19 章：預訓練中小型語言模型 — 用 128 GB 從零開始訓練

::: tip 🎯 本章你將學到什麼
- 預訓練 vs. 微調的差異
- 預訓練 BERT 和 GPT 系列模型
- autoresearch 自動研究
- NVFP4 預訓練
- 預訓練 Embedding 模型
:::

---

## 19-1 為什麼 DGX Spark 適合預訓練中小型模型

### 19-1-1 預訓練 vs. 微調

| | 預訓練 | 微調 |
|--|--------|------|
| 起點 | 隨機權重 | 已有模型 |
| 資料量 | 大量（GB-TB 級） | 少量（MB 級） |
| 時間 | 數天-數週 | 數小時 |
| 目的 | 學習語言基本能力 | 學習特定領域知識 |
| 成本 | 高 | 低 |

::: info 🤔 為什麼要自己預訓練？
開源模型已經很強了，為什麼還要自己訓練？

答案是：**領域知識**。

如果你是醫療領域，通用模型可能不懂專業術語。如果你有自己的資料，預訓練一個領域模型可以大幅提升準確率。
:::

### 19-1-2 適合預訓練的模型規模

在 DGX Spark 上，適合預訓練的模型規模：

| 模型大小 | 記憶體需求 | 訓練時間（1M tokens） | 適合場景 |
|---------|-----------|---------------------|---------|
| 10M | ~1 GB | ~30 分鐘 | 學習、測試 |
| 100M | ~5 GB | ~2 小時 | 領域 BERT |
| **1B** | **~20 GB** | **~1 天** | **領域 GPT** |
| 3B | ~50 GB | ~3 天 | 進階實驗 |
| 8B+ | 128GB 不夠 | - | 需要多機 |

### 19-1-3 128 GB 的 Batch Size 優勢

預訓練時，更大的 batch size 通常意味著更穩定的訓練和更好的結果。

DGX Spark 的 128GB 記憶體讓你可以用比消費級 GPU 大 4-5 倍的 batch size。

---

## 19-2 預訓練 BERT 系列模型

### 19-2-1 為什麼要自己預訓練 BERT

BERT 是 Encoder-only 模型，適合：
- 文字分類
- 命名實體辨識
- 情感分析
- 搜尋排序

如果你有自己的領域資料，預訓練 BERT 可以大幅提升這些任務的準確率。

### 19-2-2 訓練資料準備

```python
from datasets import load_dataset

# 載入領域資料
dataset = load_dataset("text", data_files={"train": "domain_corpus.txt"})

# 分詞
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
```

### 19-2-3 在 DGX Spark 上預訓練領域 BERT

```python
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model = BertForMaskedLM.from_pretrained("bert-base-chinese")
model.to("cuda")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./domain-bert",
    per_device_train_batch_size=64,  # DGX Spark 可以用很大的 batch size！
    learning_rate=5e-5,
    num_train_epochs=3,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train()
```

### 19-2-4 下游任務評估

```python
# 在分類任務上測試
from transformers import BertForSequenceClassification

classifier = BertForSequenceClassification.from_pretrained(
    "./domain-bert/checkpoint-xxx",
    num_labels=2
)
```

---

## 19-3 預訓練小型 GPT / Decoder-only 模型

### 19-3-1 nanoGPT 與 nanochat

nanoGPT 是 Andrej Karpathy 開發的簡化版 GPT 實作，非常適合學習和實驗。

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

### 19-3-2 在 DGX Spark 上預訓練 GPT

```bash
# 準備資料
python data/openwebtext/prepare.py

# 訓練
python train.py \
  --batch_size=32 \
  --block_size=1024 \
  --n_layer=12 \
  --n_head=12 \
  --n_embd=768 \
  --max_iters=5000 \
  --device=cuda \
  --dtype=bfloat16
```

### 19-3-3 自訂模型架構

```python
# 自訂模型大小
config = {
    "n_layer": 8,      # 層數
    "n_head": 8,       # 注意力頭數
    "n_embd": 512,     # 嵌入維度
    "block_size": 2048, # 上下文長度
    "vocab_size": 50257,
}
```

---

## 19-4 autoresearch — 讓 AI 自動研究預訓練

### 19-4-1 autoresearch 的核心架構

autoresearch 是一個讓 AI 自動設計實驗、訓練模型、分析結果的框架。

```
AI Agent
  │
  ├─ 設計模型架構
  │   ├─ 層數
  │   ├─ 注意力頭數
  │   └─ 嵌入維度
  │
  ├─ 啟動訓練
  │   ├─ 設定超參數
  │   └─ 監控訓練
  │
  └─ 分析結果
      ├─ 評估指標
      └─ 建議下一步
```

### 19-4-2 在 DGX Spark 上部署 autoresearch

```bash
git clone https://github.com/SakanaAI/autoresearch.git
cd autoresearch
uv pip install -r requirements.txt
```

### 19-4-3 啟動自主研究模式

```bash
python autoresearch.py \
  --model-size-range "10M-1B" \
  --max-experiments 100 \
  --output-dir ./results
```

### 19-4-4 DGX Spark 上的 autoresearch 優勢

- 128GB 記憶體可以同時跑多個實驗
- 不需要排隊等雲端資源
- 資料不會離開本地

### 19-4-5 分析實驗結果

```python
import pandas as pd
results = pd.read_csv("./results/experiments.csv")
results.sort_values("validation_loss").head(10)
```

---

## 19-5 NVFP4 預訓練 — 用 Blackwell 原生 FP4 加速訓練

### 19-5-1 NVFP4 預訓練的原理

傳統訓練用 FP16 或 BF16。NVFP4 預訓練在訓練過程中使用 FP4 精度，可以：
- 減少記憶體用量
- 加速訓練
- 降低能耗

### 19-5-2 混合精度策略

```python
# 權重用 FP4，梯度用 BF16
model = MyModel().to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    with torch.autocast("cuda", dtype=torch.float4_e2m1fn):
        loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 19-5-3 Quartet 論文：MXFP4 原生訓練

Quartet 是 NVIDIA 發表的 MXFP4 原生訓練方法，證明了 4-bit 訓練是可行的。

### 19-5-4 在 DGX Spark 上實作 NVFP4 預訓練

```bash
# 需要 PyTorch CUDA 13.0+
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# 訓練
python train_nvf4.py --model 100M --epochs 10
```

---

## 19-6 預訓練 Embedding 模型

### 19-6-1 Embedding 模型的訓練方法

Embedding 模型把文字轉成向量，是 RAG 系統的核心。

訓練方法：
1. 收集（query, positive, negative）三元組
2. 用對比損失（contrastive loss）訓練
3. 讓相關的向量靠近，不相關的遠離

### 19-6-2 在 DGX Spark 上訓練領域 Embedding

```python
from sentence_transformers import SentenceTransformer, losses, InputExample

model = SentenceTransformer("bert-base-chinese")

train_examples = [
    InputExample(texts=["DGX Spark 是什麼？", "一台個人 AI 超級電腦"], label=1.0),
    InputExample(texts=["DGX Spark 是什麼？", "今天天氣很好"], label=0.0),
]

train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save("./domain-embedding")
```

### 19-6-3 搭配第 21 章的 RAG 系統

訓練好的 Embedding 模型可以直接用於第 21 章的 RAG 系統，大幅提升檢索準確率。

---

## 19-7 訓練監控與模型評估

### 19-7-1 TensorBoard 監控

```bash
# 啟動 TensorBoard
tensorboard --logdir ./outputs --host 0.0.0.0 --port 6006

# 瀏覽器打開 http://DGX_Spark_IP:6006
```

### 19-7-2 Weights & Biases 監控

```bash
# 安裝
uv pip install wandb

# 登入
wandb login

# 在訓練程式中加入
import wandb
wandb.init(project="dgx-spark-pretrain")
```

### 19-7-3 模型評估指標

| 指標 | 說明 | 工具 |
|------|------|------|
| Perplexity | 語言模型困惑度 | 越低越好 |
| BLEU | 生成文字與參考文字的相似度 | 越高越好 |
| ROUGE | 摘要品質評估 | 越高越好 |

### 19-7-4 模型匯出與發布

```python
# 匯出為 Hugging Face 格式
model.save_pretrained("./final-model")
tokenizer.save_pretrained("./final-model")

# 上傳到 Hugging Face
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./final-model",
    repo_id="your-username/domain-model",
)
```

---

## 19-8 本章小結

::: success ✅ 你現在知道了
- DGX Spark 適合預訓練 10M-3B 規模的模型
- BERT 適合分類和搜尋任務
- GPT 適合文字生成任務
- autoresearch 讓 AI 自動設計實驗
- NVFP4 可以加速預訓練
- Embedding 模型是 RAG 的核心
:::

::: tip 🚀 第五篇完結！
恭喜！你已經完成了「模型微調與訓練」篇。

接下來要進入最酷的部分 — 多模態 AI 與智慧代理！

👉 [前往第 20 章：多模態推論與即時視覺 AI →](/guide/chapter20/)
:::

::: info 📝 上一章
← [回到第 18 章：影像模型微調](/guide/chapter18/)
:::
