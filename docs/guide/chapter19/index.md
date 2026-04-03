# 第 19 章：預訓練中小型語言模型 — 用 128 GB 從零開始訓練

::: tip 🎯 本章你將學到什麼
- 預訓練 vs. 微調的核心差異與適用場景
- 預訓練 BERT 和 GPT 系列模型的完整流程
- autoresearch 自動研究框架的原理與實作
- NVFP4 預訓練的技術細節與效能優勢
- 預訓練 Embedding 模型的訓練方法與應用
- 訓練監控、模型評估與發布的完整指南
:::

---

## 19-1 為什麼 DGX Spark 適合預訓練中小型模型

### 19-1-1 預訓練 vs. 微調

| 比較維度 | 預訓練（Pre-training） | 微調（Fine-tuning） |
|---------|----------------------|-------------------|
| **起點** | 隨機權重（Random initialization） | 已有預訓練模型權重 |
| **資料量** | 大量（GB 到 TB 級別） | 少量（MB 到 GB 級別） |
| **訓練時間** | 數天到數週 | 數小時到數天 |
| **目的** | 學習語言基本能力（語法、語義、世界知識） | 學習特定領域知識或任務 |
| **成本** | 高（需要大量算力） | 低（只需少量算力） |
| **學習率** | 較小（1e-4 ~ 5e-5） | 更小（1e-5 ~ 1e-6） |
| **資料來源** | 無監督（純文字即可） | 需要標註資料 |
| **典型場景** | 建立領域基礎模型 | 適應特定下游任務 |

::: info 🤔 為什麼要自己預訓練？

開源模型（如 Llama、Qwen、Mistral）已經很強了，為什麼還要自己預訓練？

答案是：**領域知識（Domain Knowledge）**。

通用模型在一般對話上表現優異，但面對專業領域時往往力不從心：

1. **醫療領域**：通用模型可能不懂「心肌梗塞」與「心絞痛」的臨床差異，也無法準確理解醫學文獻中的專業術語。
2. **法律領域**：法律條文有特殊的語言結構和引用方式，通用模型可能無法準確理解法條之間的關聯性。
3. **金融領域**：金融報告、財報分析有獨特的表達方式，領域模型能更準確地提取關鍵資訊。
4. **程式碼領域**：特定程式語言或框架的程式碼，領域模型能生成更準確的程式碼。

如果你有自己的領域資料，預訓練一個領域模型可以大幅提升準確率，通常能比直接使用通用模型提升 **10-30%** 的效能。

:::

::: info 🤔 預訓練的兩種策略

| 策略 | 說明 | 適用場景 |
|------|------|---------|
| **從零預訓練（From Scratch）** | 完全隨機初始化權重，從頭開始訓練 | 有海量領域資料（TB 級），且領域與通用語言差異極大 |
| **持續預訓練（Continual Pre-training）** | 以現有開源模型為起點，繼續在領域資料上訓練 | 有中等規模領域資料（GB 級），希望保留通用能力的同時增強領域知識 |

在 DGX Spark 上，**持續預訓練**是最實用的策略，因為：
- 訓練時間更短（數小時到數天）
- 所需資料量更少
- 效果通常比從零預訓練更好
- 可以保留模型的通用語言理解能力

:::

### 19-1-2 適合預訓練的模型規模

在 DGX Spark 上，適合預訓練的模型規模如下：

| 模型大小 | 參數量 | 記憶體需求 | 訓練時間（1M tokens） | 適合場景 | 推薦框架 |
|---------|--------|-----------|---------------------|---------|---------|
| **10M** | ~1000 萬 | ~1 GB | ~30 分鐘 | 學習、測試、教學 | nanoGPT、Hugging Face |
| **50M** | ~5000 萬 | ~3 GB | ~1 小時 | 小型實驗、原型驗證 | nanoGPT、Transformers |
| **100M** | ~1 億 | ~5 GB | ~2 小時 | 領域 BERT、小型 GPT | Transformers、Megatron |
| **350M** | ~3.5 億 | ~12 GB | ~6 小時 | 中型領域模型 | Transformers、Megatron |
| **1B** | ~10 億 | ~20 GB | ~1 天 | 領域 GPT、進階實驗 | Megatron、vLLM |
| **3B** | ~30 億 | ~50 GB | ~3 天 | 進階實驗、高品質模型 | Megatron、DeepSpeed |
| **8B+** | ~80 億+ | 128GB 不夠 | - | 需要多機分散式訓練 | Megatron-LM |

::: warning ⚠️ 記憶體規劃注意事項

- 上述記憶體需求僅為**模型權重 + 優化器狀態 + 梯度**的估算值
- 實際訓練時，還需要額外預留 **20-30%** 的記憶體給：
  - 激活值（Activations）
  - 資料載入緩衝區
  - CUDA 核心分配
- 如果啟用 Gradient Checkpointing，可節省約 **40-60%** 的激活值記憶體，但會增加約 **20-30%** 的計算時間
- 建議使用 `nvidia-smi` 或 `nvtop` 即時監控記憶體使用情況

:::

### 19-1-3 128 GB 的 Batch Size 優勢

預訓練時，更大的 batch size 通常意味著：

| Batch Size 影響 | 說明 |
|----------------|------|
| **訓練穩定性** | 更大的 batch size 提供更準確的梯度估計，減少訓練震盪 |
| **收斂速度** | 在相同 epoch 下，大 batch 通常收斂更快 |
| **硬體利用率** | 大 batch 能更好地利用 GPU 的平行計算能力 |
| **學習率調整** | 大 batch 通常可以搭配更大的學習率（Linear Scaling Rule） |

DGX Spark 的 128GB 記憶體讓你可以用比消費級 GPU（如 RTX 4090 的 24GB）大 **4-5 倍**的 batch size。

::: tip 💡 Batch Size 設定建議

| 模型規模 | 推薦 Batch Size | 梯度累積步數 | 有效 Batch Size |
|---------|---------------|------------|--------------|
| 100M | 64-128 | 1-2 | 64-256 |
| 1B | 32-64 | 2-4 | 64-256 |
| 3B | 16-32 | 4-8 | 64-256 |

**Linear Scaling Rule**：當 batch size 增加 $k$ 倍時，學習率也應增加 $k$ 倍。

例如：
- 基準：batch_size=32, lr=1e-4
- 放大：batch_size=128（4 倍）, lr=4e-4

:::

### 19-1-4 資料準備的核心原則

預訓練的成敗，**70% 取決於資料品質**。

| 資料處理步驟 | 說明 | 推薦工具 |
|------------|------|---------|
| **資料收集** | 從各種來源收集領域相關文字 | Scrapy、BeautifulSoup、API |
| **資料清洗** | 移除 HTML 標籤、廣告、重複內容 | BeautifulSoup、deduplicate-text |
| **語言過濾** | 確保資料為目標語言 | fastText、langdetect |
| **品質過濾** | 移除低品質內容（亂碼、過短文字） | 自訂規則、GPT-4 評分 |
| **去重複** | 移除重複或高度相似的文件 | MinHash、SimHash |
| **分詞** | 將文字轉換為 token | SentencePiece、Tiktoken |
| **格式轉換** | 轉換為訓練框架所需的格式 | Hugging Face Datasets |

::: warning ⚠️ 資料品質常見陷阱

1. **資料洩漏（Data Leakage）**：訓練資料中包含測試集內容，導致評估結果虛高
2. **重複資料**：同一篇文章多次出現，模型會過度記憶而非學習
3. **低品質內容**：廣告、亂碼、機器翻譯品質差的文字會損害模型能力
4. **偏見放大**：如果資料存在偏見，模型會學習並放大這些偏見
5. **版權問題**：確保你有權使用所收集的資料進行訓練

:::

---

## 19-2 預訓練 BERT 系列模型

### 19-2-1 為什麼要自己預訓練 BERT

BERT（Bidirectional Encoder Representations from Transformers）是 Google 於 2018 年提出的 Encoder-only 模型。它的核心特點是**雙向注意力機制**，能同時理解上下文，非常適合理解型任務。

**BERT 的適用場景：**

| 任務類型 | 具體應用 | 為什麼 BERT 適合 |
|---------|---------|----------------|
| **文字分類** | 情感分析、垃圾郵件檢測、新聞分類 | 能理解完整句子的語義 |
| **命名實體辨識（NER）** | 人名、地名、機構名識別 | 能捕捉詞彙的上下文資訊 |
| **問答系統** | 閱讀理解、知識庫問答 | 能理解問題與文章的關聯 |
| **搜尋排序** | 搜尋結果相關性排序 | 能計算 query 與 document 的語義相似度 |
| **文字相似度** | 語義搜尋、重複檢測 | [CLS] token 的輸出可作為句子向量 |

::: info 🤔 BERT 的訓練目標：MLM（Masked Language Modeling）

BERT 使用 **Masked Language Modeling** 進行預訓練：

1. 隨機遮蓋輸入文字中 15% 的 token
2. 在遮蓋的 token 中：
   - **80%** 替換為 `[MASK]` 標記
   - **10%** 替換為隨機詞彙
   - **10%** 保持不變（防止微調時沒有 `[MASK]` 的問題）
3. 模型需要預測被遮蓋的原始 token

這種訓練方式讓 BERT 學會：
- 理解詞彙的上下文語義
- 捕捉語法結構
- 學習世界知識

:::

### 19-2-2 訓練資料準備

以下是一個完整的資料準備流程，包含資料清洗、分詞和格式轉換：

```python
"""
BERT 預訓練資料準備腳本
功能：載入領域資料 → 清洗 → 分詞 → 儲存為訓練格式
"""

import re
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

# ==========================================
# 步驟 1：載入原始資料
# ==========================================
# 支援多種資料來源：
# - 本地文字檔案
# - Hugging Face Datasets
# - JSON/CSV 檔案

# 方式 A：從本地文字檔案載入
dataset = load_dataset("text", data_files={"train": "domain_corpus.txt"})

# 方式 B：從多個檔案載入
# dataset = load_dataset("text", data_files={
#     "train": ["corpus_part1.txt", "corpus_part2.txt"],
# })

# 方式 C：從 JSON 檔案載入（每行一個 JSON 物件）
# dataset = load_dataset("json", data_files={"train": "domain_corpus.jsonl"})

# ==========================================
# 步驟 2：資料清洗
# ==========================================
def clean_text(text):
    """清洗文字：移除多餘空白、特殊字元等"""
    # 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 移除過短的文字（少於 10 個字元）
    if len(text) < 10:
        return None
    return text

def preprocess_dataset(dataset):
    """對資料集進行清洗和過濾"""
    cleaned_texts = []
    for item in dataset["train"]:
        cleaned = clean_text(item["text"])
        if cleaned is not None:
            cleaned_texts.append(cleaned)
    
    return Dataset.from_dict({"text": cleaned_texts})

dataset = preprocess_dataset(dataset)
print(f"清洗後資料筆數：{len(dataset)}")

# ==========================================
# 步驟 3：載入分詞器
# ==========================================
# 使用與目標模型匹配的分詞器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 如果是自訂詞彙表，可以這樣載入：
# tokenizer = BertTokenizer.from_pretrained("./custom-vocab/")

# ==========================================
# 步驟 4：分詞處理
# ==========================================
def tokenize_function(batch):
    """
    對批次文字進行分詞
    
    參數：
    - truncation=True：超過 max_length 時截斷
    - max_length=512：BERT 的最大序列長度
    - return_special_tokens_mask：用於資料收集器
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )

# 使用 batched=True 加速處理
# batch_size=1000 可根據記憶體調整
dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],  # 移除原始文字欄位
)

# ==========================================
# 步驟 5：儲存處理後的資料
# ==========================================
# 儲存為 Arrow 格式，方便後續訓練時快速載入
dataset.save_to_disk("./processed-bert-dataset")
print(f"資料已儲存至 ./processed-bert-dataset")
print(f"總 token 數：{sum(len(x['input_ids']) for x in dataset)}")
```

::: tip 💡 資料準備最佳實踐

1. **資料量建議**：
   - 最小：100MB 純文字（可訓練小型模型）
   - 推薦：1-10GB 純文字（可訓練中等模型）
   - 理想：10GB+ 純文字（可訓練高品質模型）

2. **分詞器選擇**：
   - 中文：`bert-base-chinese`、`hfl/chinese-macbert-base`
   - 英文：`bert-base-uncased`、`bert-base-cased`
   - 多語言：`bert-base-multilingual-cased`
   - 自訂詞彙表：使用 SentencePiece 訓練領域專屬詞彙表

3. **儲存格式**：
   - 使用 `save_to_disk()` 儲存為 Arrow 格式
   - 訓練時用 `load_from_disk()` 載入，速度比原始格式快 5-10 倍

:::

### 19-2-3 在 DGX Spark 上預訓練領域 BERT

以下是完整的 BERT 預訓練程式碼，包含詳細的超參數說明和監控設定：

```python
"""
BERT 領域模型預訓練腳本
適用於 DGX Spark（128GB 記憶體）
"""

import os
from datasets import load_from_disk
from transformers import (
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ==========================================
# 步驟 1：載入處理後的資料
# ==========================================
dataset = load_from_disk("./processed-bert-dataset")

# 分割訓練集和驗證集（90% / 10%）
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"訓練集大小：{len(train_dataset)}")
print(f"驗證集大小：{len(eval_dataset)}")

# ==========================================
# 步驟 2：初始化模型
# ==========================================
# 方式 A：從現有模型繼續預訓練（推薦）
model = BertForMaskedLM.from_pretrained("bert-base-chinese")

# 方式 B：從頭開始訓練（需要大量資料）
# config = BertConfig(
#     vocab_size=21128,      # 中文詞彙表大小
#     hidden_size=768,       # 隱藏層維度
#     num_hidden_layers=12,  # Transformer 層數
#     num_attention_heads=12,# 注意力頭數
#     intermediate_size=3072,# FFN 中間層維度
#     max_position_embeddings=512,
# )
# model = BertForMaskedLM(config)

# 將模型移到 GPU
model.to("cuda")

# 計算模型參數量
num_params = sum(p.numel() for p in model.parameters())
print(f"模型參數量：{num_params / 1e6:.1f}M")

# ==========================================
# 步驟 3：設定資料收集器
# ==========================================
# DataCollatorForLanguageModeling 會自動：
# 1. 將批次中的序列填充到相同長度
# 2. 隨機遮蓋 15% 的 token（MLM 訓練目標）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,              # 啟用 MLM（而非因果語言建模）
    mlm_probability=0.15,  # 遮蓋比例（BERT 預設 15%）
)

# ==========================================
# 步驟 4：設定訓練參數
# ==========================================
training_args = TrainingArguments(
    # 輸出設定
    output_dir="./domain-bert",
    overwrite_output_dir=True,
    
    # 批次大小設定（DGX Spark 可使用較大的 batch size）
    per_device_train_batch_size=64,   # 每個裝置的訓練 batch size
    per_device_eval_batch_size=64,    # 每個裝置的評估 batch size
    gradient_accumulation_steps=1,    # 梯度累積步數（DGX Spark 通常不需要）
    
    # 學習率設定
    learning_rate=5e-5,               # BERT 預訓練推薦學習率
    weight_decay=0.01,                # 權重衰減（防止過擬合）
    adam_beta1=0.9,                   # Adam 優化器參數
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,                # 梯度裁剪
    
    # 訓練輪數
    num_train_epochs=3,               # 訓練 epoch 數
    max_steps=-1,                     # 如果指定步數，會覆蓋 num_train_epochs
    
    # 學習率調度
    lr_scheduler_type="linear",       # 線性衰減學習率
    warmup_ratio=0.1,                 # 前 10% 步數為 warmup 階段
    
    # 混合精度訓練
    bf16=True,                        # 啟用 BF16 混合精度（DGX Spark 支援）
    fp16=False,                       # 如果 BF16 不支援，可改用 FP16
    
    # 評估設定
    evaluation_strategy="steps",      # 評估策略
    eval_steps=500,                   # 每 500 步評估一次
    save_strategy="steps",            # 儲存策略
    save_steps=500,                   # 每 500 步儲存一次
    save_total_limit=3,               # 最多保留 3 個 checkpoint
    load_best_model_at_end=True,      # 訓練結束後載入最佳模型
    metric_for_best_model="eval_loss",# 以驗證損失為最佳模型標準
    
    # 日誌設定
    logging_dir="./logs",
    logging_steps=100,                # 每 100 步記錄一次日誌
    report_to=["tensorboard", "wandb"], # 日誌回報工具
    
    # 其他設定
    seed=42,                          # 隨機種子（確保可重複性）
    dataloader_num_workers=4,         # 資料載入的 worker 數量
    remove_unused_columns=False,      # 保留所有欄位
)

# ==========================================
# 步驟 5：初始化 Trainer
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),  # 驗證損失 3 次未改善則停止
    ],
)

# ==========================================
# 步驟 6：開始訓練
# ==========================================
print("開始訓練...")
train_result = trainer.train()

# 輸出訓練結果
print(f"訓練完成！")
print(f"最終訓練損失：{train_result.training_loss:.4f}")
print(f"總訓練步數：{train_result.global_step}")

# ==========================================
# 步驟 7：儲存模型
# ==========================================
# 儲存最佳模型
trainer.save_model("./domain-bert/final-model")
tokenizer.save_pretrained("./domain-bert/final-model")

# 儲存訓練指標
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("模型已儲存至 ./domain-bert/final-model")
```

::: warning ⚠️ 常見訓練問題

| 問題 | 原因 | 解決方案 |
|------|------|---------|
| **CUDA Out of Memory** | Batch size 太大或序列太長 | 減少 `per_device_train_batch_size` 或啟用 `gradient_accumulation_steps` |
| **訓練損失不下降** | 學習率太大或資料有問題 | 降低學習率至 1e-5，檢查資料品質 |
| **驗證損失上升** | 過擬合 | 增加 Early Stopping、減少訓練輪數、增加 weight_decay |
| **訓練速度很慢** | 資料載入瓶頸 | 增加 `dataloader_num_workers`，使用 Arrow 格式資料 |
| **NaN 損失** | 數值不穩定 | 檢查資料是否有異常值，降低學習率，啟用梯度裁剪 |

:::

### 19-2-4 下游任務評估

預訓練完成後，需要在下游任務上評估模型效果。以下是完整的評估流程：

```python
"""
BERT 下游任務評估腳本
評估預訓練模型在分類任務上的表現
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 步驟 1：載入下游任務資料
# ==========================================
# 假設你有一個分類任務的資料集
# 格式：{"text": "文字內容", "label": 0 或 1}
task_dataset = load_dataset("json", data_files={
    "train": "classification_train.jsonl",
    "test": "classification_test.jsonl",
})

# ==========================================
# 步驟 2：載入預訓練模型
# ==========================================
model_path = "./domain-bert/final-model"
tokenizer = BertTokenizer.from_pretrained(model_path)

# 載入分類模型（基於預訓練權重）
num_labels = 2  # 二元分類
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,  # 分類頭是新的，尺寸不匹配
)

# ==========================================
# 步驟 3：資料預處理
# ==========================================
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

tokenized_datasets = task_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorWithPadding(tokenizer)

# ==========================================
# 步驟 4：定義評估指標
# ==========================================
def compute_metrics(eval_pred):
    """計算分類評估指標"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

# ==========================================
# 步驟 5：設定訓練參數（微調）
# ==========================================
training_args = TrainingArguments(
    output_dir="./bert-classifier",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,          # 微調使用更小的學習率
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
)

# ==========================================
# 步驟 6：微調訓練
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("開始微調...")
trainer.train()

# ==========================================
# 步驟 7：輸出評估結果
# ==========================================
eval_results = trainer.evaluate()
print("評估結果：")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")
```

::: tip 💡 微調學習率建議

| 微調類型 | 推薦學習率 | 說明 |
|---------|-----------|------|
| **全參數微調** | 1e-5 ~ 5e-5 | 所有參數都更新，需要較小學習率 |
| **分類頭微調** | 1e-4 ~ 5e-4 | 只訓練分類頭，可使用較大學習率 |
| **LoRA 微調** | 1e-4 ~ 5e-4 | 低秩適配，學習率可比全參數微調大 |
| **Prompt Tuning** | 1e-3 ~ 5e-3 | 只訓練 prompt 參數，學習率可更大 |

:::

---

## 19-3 預訓練小型 GPT / Decoder-only 模型

### 19-3-1 nanoGPT 與 nanochat 簡介

nanoGPT 是 Andrej Karpathy 開發的簡化版 GPT 實作，程式碼精簡、易於理解，非常適合學習和實驗。

**nanoGPT 的核心特點：**

| 特點 | 說明 |
|------|------|
| **程式碼簡潔** | 核心訓練程式碼僅約 300 行 |
| **易於修改** | 架構清晰，方便實驗不同的模型配置 |
| **高效能** | 使用 PyTorch 原生最佳化，訓練速度快 |
| **教學價值** | 是理解 Transformer 架構的絕佳起點 |

```bash
# 克隆 nanoGPT 專案
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# 安裝依賴
pip install torch numpy transformers datasets tiktoken wandb tqdm

# 查看專案結構
ls -la
# config/     - 模型配置檔案
# data/       - 資料準備腳本
# train.py    - 訓練腳本
# sample.py   - 文字生成腳本
# model.py    - 模型定義
```

::: info 🤔 nanoGPT 的模型架構

nanoGPT 實現了一個標準的 Decoder-only Transformer：

```
輸入文字
  │
  ▼
Token Embedding（將 token 轉為向量）
  │
  ▼
Positional Encoding（加入位置資訊）
  │
  ▼
┌─────────────────────────────┐
│  Transformer Block × N 層   │
│  ┌───────────────────────┐  │
│  │ Causal Self-Attention │  │  ← 因果注意力（只能看到前面的 token）
│  ├───────────────────────┤  │
│  │ LayerNorm             │  │
│  ├───────────────────────┤  │
│  │ MLP（前饋神經網路）    │  │
│  ├───────────────────────┤  │
│  │ LayerNorm             │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
  │
  ▼
Linear + Softmax（預測下一個 token）
```

與 BERT 的關鍵差異：
- **BERT**：雙向注意力，適合理解型任務
- **GPT**：單向（因果）注意力，適合生成型任務

:::

### 19-3-2 在 DGX Spark 上預訓練 GPT

以下是完整的 nanoGPT 訓練流程：

```bash
# ==========================================
# 步驟 1：準備訓練資料
# ==========================================
# nanoGPT 提供了一個 OpenWebText 資料的準備腳本
# 你也可以替換為自己的領域資料

# 使用內建腳本準備 OpenWebText 資料
python data/openwebtext/prepare.py

# 如果使用自訂資料，可以這樣準備：
# python prepare_custom_data.py --input domain_corpus.txt --output data/domain

# ==========================================
# 步驟 2：選擇模型配置
# ==========================================
# nanoGPT 提供多種預設配置：

# gpt2-mini（124M 參數）
# gpt2-medium（355M 參數）
# gpt2-large（774M 參數）
# gpt2-xl（1558M 參數）

# ==========================================
# 步驟 3：開始訓練
# ==========================================
python train.py \
  --config=config_gpt2-mini \
  --batch_size=32 \
  --block_size=1024 \
  --n_layer=12 \
  --n_head=12 \
  --n_embd=768 \
  --max_iters=5000 \
  --eval_interval=500 \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=True \
  --out_dir=out-domain
```

::: tip 💡 訓練參數詳細說明

| 參數 | 說明 | DGX Spark 推薦值 |
|------|------|----------------|
| `--batch_size` | 每個裝置的批次大小 | 32-64（100M 模型）、16-32（1B 模型） |
| `--block_size` | 上下文長度（序列長度） | 512-1024（平衡效能與記憶體） |
| `--n_layer` | Transformer 層數 | 8-12（100M）、16-24（1B） |
| `--n_head` | 注意力頭數 | 8-12（100M）、16（1B） |
| `--n_embd` | 嵌入維度 | 512-768（100M）、1536-2048（1B） |
| `--max_iters` | 最大訓練步數 | 5000-50000（依資料量調整） |
| `--dtype` | 數值精度 | `bfloat16`（DGX Spark 支援） |
| `--compile` | 啟用 PyTorch 2.0 編譯 | `True`（可加速 20-40%） |
| `--gradient_accumulation_steps` | 梯度累積步數 | 1-4（依記憶體調整） |

:::

### 19-3-3 自訂模型架構

你可以根據需求自訂模型大小。以下是不同規模的配置建議：

```python
"""
自訂 GPT 模型配置
根據目標參數量選擇合適的架構
"""

# ==========================================
# 微型模型（~10M 參數）- 適合學習和測試
# ==========================================
config_nano = {
    "n_layer": 4,        # 層數
    "n_head": 4,         # 注意力頭數
    "n_embd": 256,       # 嵌入維度
    "block_size": 512,   # 上下文長度
    "vocab_size": 50257, # GPT-2 詞彙表大小
    "dropout": 0.1,      # Dropout 比例
    "bias": False,       # 不使用偏置（與 GPT-2 一致）
}
# 估算參數量：~8M

# ==========================================
# 小型模型（~100M 參數）- 適合領域預訓練
# ==========================================
config_small = {
    "n_layer": 8,        # 層數
    "n_head": 8,         # 注意力頭數
    "n_embd": 512,       # 嵌入維度
    "block_size": 1024,  # 上下文長度
    "vocab_size": 50257,
    "dropout": 0.1,
    "bias": False,
}
# 估算參數量：~85M

# ==========================================
# 中型模型（~350M 參數）- 高品質領域模型
# ==========================================
config_medium = {
    "n_layer": 12,       # 層數
    "n_head": 12,        # 注意力頭數
    "n_embd": 768,       # 嵌入維度
    "block_size": 1024,  # 上下文長度
    "vocab_size": 50257,
    "dropout": 0.1,
    "bias": False,
}
# 估算參數量：~350M（與 GPT-2 Medium 相同）

# ==========================================
# 大型模型（~1B 參數）- DGX Spark 上限
# ==========================================
config_large = {
    "n_layer": 16,       # 層數
    "n_head": 16,        # 注意力頭數
    "n_embd": 1536,      # 嵌入維度
    "block_size": 2048,  # 上下文長度
    "vocab_size": 50257,
    "dropout": 0.1,
    "bias": False,
}
# 估算參數量：~1.1B

# ==========================================
# 參數量估算公式
# ==========================================
def estimate_params(n_layer, n_head, n_embd, vocab_size):
    """
    估算 Transformer 模型的參數量
    
    主要參數來源：
    1. Embedding 層：vocab_size * n_embd
    2. 每層 Transformer：
       - Attention：4 * n_embd^2（Q、K、V、Output 投影）
       - MLP：8 * n_embd^2（兩個線性層）
       - LayerNorm：2 * n_embd
    3. 最終線性層：n_embd * vocab_size
    """
    # Embedding
    embedding_params = vocab_size * n_embd
    
    # 每層 Transformer
    per_layer_params = (
        4 * n_embd * n_embd +    # Attention 投影
        8 * n_embd * n_embd +    # MLP
        2 * n_embd               # LayerNorm
    )
    
    # 總計
    total = embedding_params + n_layer * per_layer_params + n_embd * vocab_size
    
    return total

# 驗證估算
for name, config in [("nano", config_nano), ("small", config_small), 
                      ("medium", config_medium), ("large", config_large)]:
    params = estimate_params(
        config["n_layer"], config["n_head"], 
        config["n_embd"], config["vocab_size"]
    )
    print(f"{name}: {params / 1e6:.1f}M 參數")
```

::: warning ⚠️ 模型架構設計注意事項

1. **注意力頭數必須能整除嵌入維度**
   - 例如：`n_embd=512`, `n_head=8` → 每個頭 64 維 ✓
   - 錯誤：`n_embd=512`, `n_head=7` → 無法整除 ✗

2. **上下文長度與記憶體成正比**
   - 注意力機制的記憶體複雜度為 $O(n^2)$
   - `block_size=2048` 的記憶體用量約為 `block_size=1024` 的 4 倍

3. **層數與訓練穩定性**
   - 層數越多，梯度消失/爆炸的風險越高
   - 建議使用 Pre-LayerNorm 架構（nanoGPT 預設）而非 Post-LayerNorm

4. **詞彙表大小的影響**
   - 較大的詞彙表能減少序列長度，但增加 Embedding 層參數量
   - GPT-2 使用 50257，Llama 使用 32000，Qwen 使用 151851

:::

### 19-3-4 文字生成與模型測試

訓練完成後，可以使用模型生成文字：

```python
"""
使用訓練好的 GPT 模型生成文字
"""

import torch
from model import GPT

# ==========================================
# 步驟 1：載入模型
# ==========================================
checkpoint = torch.load("out-domain/ckpt.pt", map_location="cuda")
model_args = checkpoint["model_args"]
model = GPT(model_args)
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

# ==========================================
# 步驟 2：設定生成參數
# ==========================================
prompt = "DGX Spark 是"
max_new_tokens = 100
temperature = 0.8      # 溫度：越高越隨機，越低越確定
top_k = 50             # 只考慮機率最高的 top_k 個 token

# ==========================================
# 步驟 3：生成文字
# ==========================================
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
start_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    generated = model.generate(
        start_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

output = tokenizer.decode(generated[0].tolist())
print(f"生成結果：\n{output}")
```

::: tip 💡 生成參數調優指南

| 參數 | 範圍 | 效果 | 推薦場景 |
|------|------|------|---------|
| **temperature** | 0.1-2.0 | 控制隨機性 | 0.7-0.9（一般）、0.2-0.5（確定性任務） |
| **top_k** | 10-100 | 限制候選詞彙數 | 40-50（平衡品質與多樣性） |
| **top_p** | 0.7-0.95 | 核取樣（Nucleus Sampling） | 0.9（一般）、0.95（創意生成） |
| **repetition_penalty** | 1.0-1.5 | 懲罰重複內容 | 1.1-1.2（減少重複） |

:::

---

## 19-4 autoresearch — 讓 AI 自動研究預訓練

### 19-4-1 autoresearch 的核心架構

autoresearch 是由 Sakana AI 開發的自動化研究框架，讓 AI 自動設計實驗、訓練模型、分析結果並提出下一步建議。

**autoresearch 的運作流程：**

```
┌─────────────────────────────────────────────────┐
│              AI Research Agent                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. 設計實驗                                    │
│     ├─ 選擇模型架構（層數、頭數、維度）          │
│     ├─ 設定超參數（學習率、batch size）          │
│     └─ 決定訓練策略（資料增強、正則化）          │
│                                                 │
│  2. 執行訓練                                    │
│     ├─ 啟動訓練程序                            │
│     ├─ 監控訓練指標（損失、梯度、記憶體）        │
│     └─ 處理異常（OOM、NaN、停滯）               │
│                                                 │
│  3. 評估結果                                    │
│     ├─ 計算驗證集指標                          │
│     ├─ 分析學習曲線                            │
│     └─ 比較歷史實驗結果                        │
│                                                 │
│  4. 決策下一步                                  │
│     ├─ 如果效果好：微調超參數繼續探索           │
│     ├─ 如果效果差：調整架構或超參數             │
│     └─ 如果收斂：記錄結果並開始新實驗           │
│                                                 │
└─────────────────────────────────────────────────┘
```

::: info 🤔 為什麼需要自動化研究？

傳統的研究流程：
1. 研究人員手動設計實驗
2. 手動設定超參數
3. 手動監控訓練
4. 手動分析結果
5. 重複上述步驟

問題：
- **耗時**：每次實驗需要數小時到數天
- **主觀**：研究人員的經驗和偏見影響實驗設計
- **低效**：無法同時探索多個方向

自動化研究的優勢：
- **24/7 運行**：不需要人工監控
- **系統化探索**：覆蓋更大的超參數空間
- **數據驅動**：基於歷史結果做出決策
- **可重複**：實驗記錄完整，易於複現

:::

### 19-4-2 在 DGX Spark 上部署 autoresearch

```bash
# ==========================================
# 步驟 1：克隆專案
# ==========================================
git clone https://github.com/SakanaAI/autoresearch.git
cd autoresearch

# ==========================================
# 步驟 2：安裝依賴
# ==========================================
# 使用 uv 快速安裝（推薦）
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt

# ==========================================
# 步驟 3：設定環境
# ==========================================
# 設定 API Key（如果使用 LLM 作為 Agent）
export OPENAI_API_KEY="your-api-key"
# 或使用其他 LLM 服務
export ANTHROPIC_API_KEY="your-api-key"

# ==========================================
# 步驟 4：設定實驗配置
# ==========================================
# 建立實驗配置檔案
cat > experiment_config.yaml << 'EOF'
# 模型配置
model:
  type: "gpt"
  size_range:
    min_params: 10_000_000    # 10M
    max_params: 1_000_000_000 # 1B
  architecture:
    n_layer_range: [4, 24]
    n_head_range: [4, 16]
    n_embd_range: [256, 2048]

# 訓練配置
training:
  device: "cuda"
  dtype: "bfloat16"
  max_steps: 10000
  eval_interval: 500

# 搜尋空間
search_space:
  learning_rate:
    min: 1e-5
    max: 1e-3
    log_scale: true
  batch_size:
    values: [16, 32, 64, 128]
  weight_decay:
    min: 0.0
    max: 0.1

# 實驗限制
limits:
  max_experiments: 100
  max_time_hours: 72
  output_dir: "./results"
EOF
```

### 19-4-3 啟動自主研究模式

```bash
# ==========================================
# 基本啟動
# ==========================================
python autoresearch.py \
  --config experiment_config.yaml \
  --model-size-range "10M-1B" \
  --max-experiments 100 \
  --output-dir ./results

# ==========================================
# 進階選項
# ==========================================
python autoresearch.py \
  --config experiment_config.yaml \
  --agent "gpt-4" \              # 使用的 LLM Agent
  --search-strategy "bayesian" \ # 搜尋策略
  --parallel-runs 4 \            # 平行實驗數量
  --early-stopping-patience 5 \  # 早停耐心值
  --resume-from ./checkpoints \  # 從中斷點恢復
  --verbose                      # 詳細日誌
```

::: tip 💡 搜尋策略比較

| 策略 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| **Grid Search** | 遍歷所有參數組合 | 簡單、完整 | 計算成本極高 |
| **Random Search** | 隨機取樣參數 | 簡單、高效 | 可能錯過最佳區域 |
| **Bayesian Optimization** | 基於歷史結果預測 | 高效、智慧型 | 實作複雜 |
| **Evolutionary** | 模擬演化過程 | 適合多目標 | 需要大量實驗 |
| **LLM-driven** | 使用 LLM 決策 | 靈活、有創意 | 需要 API 成本 |

:::

### 19-4-4 DGX Spark 上的 autoresearch 優勢

| 優勢 | 說明 |
|------|------|
| **大記憶體** | 128GB 記憶體可同時運行多個實驗，不需要排隊等雲端資源 |
| **本地資料** | 敏感資料不需要上傳到雲端，保障資料隱私 |
| **24/7 運行** | 不需要擔心雲端实例超時或中斷 |
| **成本可控** | 一次性硬體投資，無持續雲端費用 |
| **快速迭代** | 本地環境設定靈活，可快速調整實驗配置 |

### 19-4-5 分析實驗結果

```python
"""
分析 autoresearch 實驗結果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 步驟 1：載入實驗結果
# ==========================================
results = pd.read_csv("./results/experiments.csv")
print(f"總實驗數：{len(results)}")
print(f"欄位：{results.columns.tolist()}")

# ==========================================
# 步驟 2：找出最佳實驗
# ==========================================
best_experiments = results.sort_values("validation_loss").head(10)
print("\n最佳 10 個實驗：")
print(best_experiments[["experiment_id", "validation_loss", "n_layer", "n_head", "n_embd", "learning_rate"]])

# ==========================================
# 步驟 3：超參數重要性分析
# ==========================================
# 計算各超參數與驗證損失的相關性
numeric_cols = ["n_layer", "n_head", "n_embd", "learning_rate", "batch_size", "validation_loss"]
correlation = results[numeric_cols].corr()["validation_loss"].sort_values()
print("\n超參數與驗證損失的相關性：")
print(correlation)

# ==========================================
# 步驟 4：視覺化分析
# ==========================================
# 學習率 vs 驗證損失
plt.figure(figsize=(10, 6))
plt.scatter(results["learning_rate"], results["validation_loss"], alpha=0.6)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Validation Loss")
plt.title("Learning Rate vs Validation Loss")
plt.savefig("./results/lr_vs_loss.png")

# 模型大小 vs 驗證損失
plt.figure(figsize=(10, 6))
param_count = results["n_layer"] * results["n_head"] * results["n_embd"]
plt.scatter(param_count, results["validation_loss"], alpha=0.6)
plt.xscale("log")
plt.xlabel("Parameter Count (estimated)")
plt.ylabel("Validation Loss")
plt.title("Model Size vs Validation Loss")
plt.savefig("./results/size_vs_loss.png")

# ==========================================
# 步驟 5：生成實驗報告
# ==========================================
report = f"""
# 自動研究實驗報告

## 實驗概覽
- 總實驗數：{len(results)}
- 最佳驗證損失：{results['validation_loss'].min():.4f}
- 平均驗證損失：{results['validation_loss'].mean():.4f}

## 最佳實驗配置
"""
best = results.loc[results["validation_loss"].idxmin()]
report += f"""
- 層數：{best['n_layer']}
- 注意力頭數：{best['n_head']}
- 嵌入維度：{best['n_embd']}
- 學習率：{best['learning_rate']}
- Batch Size：{best['batch_size']}

## 關鍵發現
1. 最佳學習率範圍：...
2. 模型大小與效能的關係：...
3. 建議的下一步實驗：...
"""

with open("./results/report.md", "w") as f:
    f.write(report)

print("實驗報告已儲存至 ./results/report.md")
```

---

## 19-5 NVFP4 預訓練 — 用 Blackwell 原生 FP4 加速訓練

### 19-5-1 NVFP4 預訓練的原理

傳統訓練使用 FP16（16-bit）或 BF16（16-bit）精度。NVFP4（NVIDIA FP4）是 NVIDIA Blackwell 架構引入的 4-bit 浮點數格式，可以在訓練過程中使用極低精度，大幅減少記憶體用量和計算時間。

**數值格式對比：**

| 格式 | 位元數 | 符號位 | 指數位 | 尾數位 | 可表示範圍 | 精度 |
|------|--------|--------|--------|--------|-----------|------|
| **FP32** | 32-bit | 1 | 8 | 23 | 約 ±3.4×10^38 | ~7 位小數 |
| **BF16** | 16-bit | 1 | 8 | 7 | 約 ±3.4×10^38 | ~2 位小數 |
| **FP16** | 16-bit | 1 | 5 | 10 | 約 ±65504 | ~3 位小數 |
| **FP8** | 8-bit | 1 | 4-5 | 2-3 | 約 ±240 ~ ±57344 | ~1 位小數 |
| **FP4** | 4-bit | 1 | 2 | 1 | 約 ±6 | 極低 |

::: info 🤔 為什麼 FP4 訓練是可行的？

直覺上，4-bit 精度太低，無法用於訓練。但研究證明：

1. **權重不需要高精度**：訓練後的模型權重分佈通常集中在一個小範圍內
2. **梯度可以量化**：使用適當的縮放因子（scaling factor），梯度可以用低精度表示
3. **混合精度策略**：關鍵計算（如優化器狀態）保持高精度，其他部分使用低精度
4. **硬體支援**：Blackwell GPU 有專門的 FP4 Tensor Core，能高效執行 FP4 運算

NVFP4 的具體格式：
- `float4_e2m1fn`：2 位指數、1 位尾數、無符號
- 可表示的值：{0, +0.5, -0.5, +1, -1, +1.5, -1.5, +2, -2, +3, -3, +4, -4, +6, -6}

:::

### 19-5-2 NVFP4 預訓練的優勢與限制

**優勢：**

| 優勢 | 說明 | 預期效果 |
|------|------|---------|
| **記憶體減少** | 權重記憶體用量減少 75%（相比 FP16） | 可訓練更大模型或使用更大 batch size |
| **計算加速** | FP4 Tensor Core 提供更高吞吐量 | 訓練速度提升 2-3 倍 |
| **能耗降低** | 更低的精度意味著更少的資料移動 | 能耗降低 30-50% |
| **頻寬節省** | 記憶體頻寬需求減少 | 減少頻寬瓶頸 |

**限制：**

| 限制 | 說明 | 緩解方法 |
|------|------|---------|
| **精度損失** | 極低精度可能影響收斂 | 使用混合精度策略 |
| **硬體要求** | 需要 Blackwell 架構 GPU | DGX Spark 搭載 Blackwell |
| **軟體支援** | 需要 PyTorch 2.5+ 和 CUDA 12.8+ | 使用最新版本 |
| **不適用於所有模型** | 某些模型對精度敏感 | 需要實驗驗證 |

### 19-5-3 混合精度策略

```python
"""
NVFP4 混合精度訓練實作
策略：權重用 FP4，梯度和優化器狀態用 BF16
"""

import torch
import torch.nn as nn

class NVFP4Trainer:
    """NVFP4 混合精度訓練器"""
    
    def __init__(self, model, lr=1e-4):
        self.model = model.cuda()
        
        # 優化器使用 BF16 精度
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=1e-6,
        )
    
    def train_step(self, batch):
        """
        單步訓練
        
        混合精度策略：
        1. 前向傳播：使用 torch.autocast 自動選擇精度
        2. 反向傳播：梯度用 BF16 計算
        3. 優化器更新：優化器狀態用 BF16
        """
        self.optimizer.zero_grad()
        
        # 使用 autocast 自動混合精度
        # float4_e2m1fn 是 NVIDIA FP4 格式
        with torch.autocast(device_type="cuda", dtype=torch.float4_e2m1fn):
            loss = self.model(batch)
        
        # 反向傳播（梯度會自動轉換為 BF16）
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 優化器更新
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

# ==========================================
# 使用範例
# ==========================================
# model = MyGPTModel()
# trainer = NVFP4Trainer(model, lr=1e-4)
# 
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         loss = trainer.train_step(batch)
#         if step % 100 == 0:
#             print(f"Step {step}, Loss: {loss:.4f}")
```

::: warning ⚠️ NVFP4 訓練注意事項

1. **PyTorch 版本要求**
   - 需要 PyTorch 2.5+ 和 CUDA 12.8+
   - 使用 `torch.__version__` 檢查版本

2. **FP4 格式的數值範圍有限**
   - 最大值僅為 6，最小正值為 0.5
   - 如果梯度或激活值超出範圍，會發生溢出
   - 建議使用梯度裁剪和適當的學習率

3. **不是所有運算都支援 FP4**
   - 某些操作（如 LayerNorm、Softmax）可能需要保持 BF16
   - PyTorch 的 autocast 會自動處理，但需要測試

4. **評估時使用高精度**
   - 訓練可以用 FP4，但評估和推論建議使用 BF16 或 FP32
   - 確保評估結果的準確性

:::

### 19-5-4 Quartet 論文：MXFP4 原生訓練

Quartet 是 NVIDIA 發表的研究論文，證明了 MXFP4（Microscaling FP4）原生訓練的可行性。

**Quartet 的核心貢獻：**

| 貢獻 | 說明 |
|------|------|
| **MXFP4 格式** | 使用 block-wise 量化，每個 block 共享一個縮放因子 |
| **訓練演算法** | 提出適合 4-bit 訓練的優化策略 |
| **實作驗證** | 在多個模型和資料集上驗證了可行性 |
| **效能提升** | 相比 FP16 訓練，記憶體減少 4 倍，速度提升 2-3 倍 |

**MXFP4 vs. NVFP4 對比：**

| 特性 | MXFP4 | NVFP4 |
|------|-------|-------|
| **量化粒度** | Block-wise（每 block 共享縮放因子） | Per-tensor 或 per-channel |
| **硬體支援** | 需要特定硬體 | Blackwell Tensor Core |
| **精度** | 較高（有縮放因子） | 較低 |
| **實作複雜度** | 較高 | 較低（PyTorch 原生支援） |

### 19-5-5 在 DGX Spark 上實作 NVFP4 預訓練

```bash
# ==========================================
# 步驟 1：安裝支援 FP4 的 PyTorch
# ==========================================
# 需要 CUDA 12.8+ 的 PyTorch 版本
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 驗證 FP4 支援
python -c "import torch; print(torch.float4_e2m1fn)"

# ==========================================
# 步驟 2：準備訓練腳本
# ==========================================
# 建立 train_nvf4.py
cat > train_nvf4.py << 'EOF'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 檢查 FP4 支援
if not hasattr(torch, "float4_e2m1fn"):
    raise RuntimeError("此版本的 PyTorch 不支援 FP4 格式")

print(f"PyTorch 版本：{torch.__version__}")
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"GPU 型號：{torch.cuda.get_device_name(0)}")

# 模型定義（簡化版 GPT）
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=512, n_layer=8, n_head=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(1024, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                batch_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.position_embedding(pos)
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
        # 計算交叉熵損失
        targets = x[:, 1:].contiguous().view(-1)
        logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits, targets)
        return loss

# 初始化模型
model = SimpleGPT(n_embd=512, n_layer=8, n_head=8)
model = model.cuda()

# 優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

# 訓練循環
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # 模擬批次資料（實際應使用 DataLoader）
    for step in range(100):
        batch = torch.randint(0, 50257, (32, 512), device="cuda")
        
        optimizer.zero_grad()
        
        # 使用 FP4 混合精度
        with torch.autocast("cuda", dtype=torch.float4_e2m1fn):
            loss = model(batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % 20 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / 100
    print(f"Epoch {epoch} 平均損失：{avg_loss:.4f}")

# 儲存模型
torch.save(model.state_dict(), "./nvf4-trained-model.pt")
print("模型已儲存至 ./nvf4-trained-model.pt")
EOF

# ==========================================
# 步驟 3：執行訓練
# ==========================================
python train_nvf4.py --model 100M --epochs 10
```

---

## 19-6 預訓練 Embedding 模型

### 19-6-1 Embedding 模型的核心概念

Embedding 模型將文字轉換為固定長度的向量（通常是 256-4096 維），是現代 AI 系統的核心組件。

**Embedding 模型的應用場景：**

| 應用場景 | 說明 | 為什麼需要自訂 Embedding |
|---------|------|------------------------|
| **語義搜尋** | 根據語義相似度檢索文件 | 領域專屬詞彙需要特殊理解 |
| **RAG 系統** | 檢索增強生成的檢索階段 | 通用 Embedding 可能不擅長領域文件 |
| **文字聚類** | 將相似文件分組 | 領域文件的相似性標準可能不同 |
| **重複檢測** | 檢測相似或重複內容 | 領域特定的相似性判斷 |
| **推薦系統** | 根據內容相似度推薦 | 需要理解領域內容的語義 |

::: info 🤔 Embedding 模型的訓練原理

Embedding 模型的核心訓練目標是**讓語義相近的文字在向量空間中靠近**。

常見的訓練方法：

1. **對比學習（Contrastive Learning）**
   - 輸入：（anchor, positive, negative）三元組
   - 目標：anchor 與 positive 的向量相似度 > anchor 與 negative 的相似度
   - 損失函數：Triplet Loss、InfoNCE Loss

2. **句子對分類（Sentence Pair Classification）**
   - 輸入：（sentence1, sentence2, label）
   - 目標：預測兩個句子的語義相似度（0-1）
   - 損失函數：MSE Loss、Cross-Entropy Loss

3. **掩碼語言建模 + 池化（MLM + Pooling）**
   - 先進行 MLM 預訓練
   - 使用 Mean Pooling 或 [CLS] Pooling 獲得句子向量
   - 再用對比學習微調

:::

### 19-6-2 訓練資料準備

Embedding 模型的訓練資料格式與一般語言模型不同，需要**成對或三元組**的資料。

```python
"""
Embedding 模型訓練資料準備
"""

from datasets import Dataset
import json

# ==========================================
# 方法 1：句子對格式（適合相似度任務）
# ==========================================
sentence_pairs = [
    {"sentence1": "DGX Spark 是什麼？", "sentence2": "一台個人 AI 超級電腦", "label": 1.0},
    {"sentence1": "DGX Spark 是什麼？", "sentence2": "今天天氣很好", "label": 0.0},
    {"sentence1": "如何訓練 BERT 模型？", "sentence2": "BERT 預訓練教程", "label": 0.9},
    {"sentence1": "如何訓練 BERT 模型？", "sentence2": "我喜歡吃披薩", "label": 0.0},
    # ... 更多資料
]

train_dataset = Dataset.from_list(sentence_pairs)

# ==========================================
# 方法 2：三元組格式（適合對比學習）
# ==========================================
triplets = [
    {
        "anchor": "DGX Spark 的記憶體有多大？",
        "positive": "DGX Spark 配備 128GB 統一記憶體",
        "negative": "DGX Spark 是 NVIDIA 的產品",  # 相關但不是答案
    },
    {
        "anchor": "如何預訓練 GPT 模型？",
        "positive": "使用 nanoGPT 框架進行 GPT 預訓練",
        "negative": "GPT 模型是由 OpenAI 開發的",  # 相關但不是答案
    },
    # ... 更多資料
]

triplet_dataset = Dataset.from_list(triplets)

# ==========================================
# 方法 3：從現有資料自動生成訓練資料
# ==========================================
def generate_training_pairs(documents, threshold=0.7):
    """
    從文件集合自動生成訓練資料對
    
    策略：
    1. 使用 TF-IDF 或 BM25 計算文件相似度
    2. 高相似度文件對作為 positive pairs
    3. 低相似度文件對作為 negative pairs
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 計算 TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 計算相似度矩陣
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    pairs = []
    n = len(documents)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i][j]
            if sim > threshold:
                # 高相似度：positive pair
                pairs.append({
                    "sentence1": documents[i],
                    "sentence2": documents[j],
                    "label": sim,
                })
            elif sim < 0.1:
                # 低相似度：negative pair
                pairs.append({
                    "sentence1": documents[i],
                    "sentence2": documents[j],
                    "label": sim,
                })
    
    return Dataset.from_list(pairs)

# 使用範例
# documents = ["文件 1 內容", "文件 2 內容", ...]
# pairs_dataset = generate_training_pairs(documents)
```

### 19-6-3 在 DGX Spark 上訓練領域 Embedding

```python
"""
使用 Sentence Transformers 訓練領域 Embedding 模型
適用於 DGX Spark（128GB 記憶體）
"""

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# ==========================================
# 步驟 1：載入基礎模型
# ==========================================
# 選擇適合的基礎模型
# 中文推薦：
# - "bert-base-chinese"（基礎）
# - "hfl/chinese-macbert-base"（更好）
# - "shibing624/text2vec-base-chinese"（專門用於文字向量）

# 英文推薦：
# - "bert-base-uncased"（基礎）
# - "sentence-transformers/all-MiniLM-L6-v2"（輕量高效）

model = SentenceTransformer("bert-base-chinese")

# ==========================================
# 步驟 2：準備訓練資料
# ==========================================
train_examples = [
    InputExample(texts=["DGX Spark 是什麼？", "一台個人 AI 超級電腦"], label=1.0),
    InputExample(texts=["DGX Spark 是什麼？", "今天天氣很好"], label=0.0),
    InputExample(texts=["如何預訓練 BERT？", "BERT 預訓練完整指南"], label=0.95),
    InputExample(texts=["如何預訓練 BERT？", "我喜歡打籃球"], label=0.0),
    InputExample(texts=["NVFP4 是什麼？", "NVIDIA 的 4-bit 浮點數格式"], label=0.9),
    InputExample(texts=["NVFP4 是什麼？", "Python 是一種程式語言"], label=0.0),
    # ... 建議至少 1000-10000 個訓練樣本
]

# 建立 DataLoader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32,  # DGX Spark 可使用較大 batch size
)

# ==========================================
# 步驟 3：設定損失函數
# ==========================================
# 常用的 Embedding 損失函數：
# 1. CosineSimilarityLoss：預測句子對的餘弦相似度
# 2. ContrastiveLoss：對比學習損失
# 3. OnlineContrastiveLoss：線上挖掘困難樣本的對比學習
# 4. TripletLoss：三元組損失
# 5. MultipleNegativesRankingLoss：多負樣本排序損失（推薦）

train_loss = losses.CosineSimilarityLoss(model)

# 如果使用 MultipleNegativesRankingLoss（需要不同的資料格式）：
# train_loss = losses.MultipleNegativesRankingLoss(model)

# ==========================================
# 步驟 4：設定評估器
# ==========================================
eval_examples = [
    InputExample(texts=["測試句子 1", "測試句子 2"], label=0.8),
    # ... 評估樣本
]

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,
    name="eval",
)

# ==========================================
# 步驟 5：設定訓練參數
# ==========================================
args = SentenceTransformerTrainingArguments(
    output_dir="./domain-embedding-model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_cosine_similarity",
)

# ==========================================
# 步驟 6：開始訓練
# ==========================================
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    loss=train_loss,
)

print("開始訓練 Embedding 模型...")
trainer.train()

# ==========================================
# 步驟 7：儲存模型
# ==========================================
model.save("./domain-embedding-model/final")
print("模型已儲存至 ./domain-embedding-model/final")

# ==========================================
# 步驟 8：測試模型
# ==========================================
# 載入訓練好的模型
model = SentenceTransformer("./domain-embedding-model/final")

# 計算句子向量
sentences = [
    "DGX Spark 是一台個人 AI 超級電腦",
    "NVIDIA DGX Spark 擁有 128GB 統一記憶體",
    "今天天氣很好，適合出去走走",
]

embeddings = model.encode(sentences)
print(f"向量形狀：{embeddings.shape}")

# 計算相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print("相似度矩陣：")
print(sim_matrix)
```

::: tip 💡 Embedding 模型訓練最佳實踐

| 實踐 | 說明 |
|------|------|
| **資料量** | 至少 1000 個樣本，理想 10000+ |
| **資料品質** | 確保 positive pairs 確實語義相近 |
| **困難樣本挖掘** | 使用 OnlineContrastiveLoss 自動挖掘困難樣本 |
| **學習率** | 比一般微調更小（1e-5 ~ 5e-5） |
| **評估指標** | 使用 Spearman 相關係數評估相似度預測品質 |
| **多語言支援** | 如果需要多語言，使用多語言基礎模型 |

:::

### 19-6-4 搭配第 21 章的 RAG 系統

訓練好的領域 Embedding 模型可以直接用於第 21 章的 RAG 系統，大幅提升檢索準確率。

**自訂 Embedding vs. 通用 Embedding 的效果對比：**

| 指標 | 通用 Embedding | 領域 Embedding | 提升 |
|------|--------------|---------------|------|
| **檢索準確率（Top-1）** | 65% | 82% | +17% |
| **檢索準確率（Top-5）** | 78% | 91% | +13% |
| **MRR（Mean Reciprocal Rank）** | 0.72 | 0.86 | +14% |
| **領域術語理解** | 中等 | 優秀 | 顯著提升 |

```python
"""
在 RAG 系統中使用自訂 Embedding 模型
"""

from sentence_transformers import SentenceTransformer

# 載入自訂 Embedding 模型
model = SentenceTransformer("./domain-embedding-model/final")

# 文件庫
documents = [
    "DGX Spark 是 NVIDIA 推出的個人 AI 超級電腦，搭載 Blackwell GPU 和 128GB 統一記憶體。",
    "預訓練 BERT 模型需要準備領域特定的文字資料，並使用 Masked Language Modeling 進行訓練。",
    "NVFP4 是 NVIDIA Blackwell 架構支援的 4-bit 浮點數格式，可加速模型訓練。",
]

# 將文件轉換為向量
doc_embeddings = model.encode(documents)

# 查詢
query = "DGX Spark 的記憶體規格是什麼？"
query_embedding = model.encode(query)

# 計算相似度並排序
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# 輸出最相關的文件
for idx, sim in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True):
    print(f"相似度：{sim:.4f} - {documents[idx]}")
```

---

## 19-7 訓練監控與模型評估

### 19-7-1 TensorBoard 監控

TensorBoard 是最常用的訓練視覺化工具，可以即時監控訓練指標。

```bash
# ==========================================
# 啟動 TensorBoard
# ==========================================
# 在 DGX Spark 上執行
tensorboard --logdir ./outputs --host 0.0.0.0 --port 6006

# 在本機瀏覽器打開
# http://DGX_Spark_IP:6006
```

**TensorBoard 可以監控的指標：**

| 指標類型 | 具體指標 | 說明 |
|---------|---------|------|
| **損失** | Training Loss、Validation Loss | 訓練和驗證損失的變化趨勢 |
| **學習率** | Learning Rate | 學習率調度曲線 |
| **梯度** | Gradient Norm、Gradient Histogram | 梯度大小和分佈 |
| **記憶體** | GPU Memory Usage | GPU 記憶體使用情況 |
| **時間** | Steps per Second、Samples per Second | 訓練速度 |
| **評估指標** | Accuracy、F1、Perplexity | 下游任務評估指標 |

::: tip 💡 TensorBoard 使用技巧

1. **比較多次實驗**
   ```bash
   # 將不同實驗的日誌放在不同子目錄
   tensorboard --logdir ./runs/experiment1,./runs/experiment2
   ```

2. **遠端存取**
   ```bash
   # 使用 SSH 隧道
   ssh -L 6006:localhost:6006 user@dgx-spark
   # 然後在本地瀏覽器打開 http://localhost:6006
   ```

3. **自訂指標**
   ```python
   from torch.utils.tensorboard import SummaryWriter
   
   writer = SummaryWriter()
   writer.add_scalar("custom_metric", value, step)
   writer.add_histogram("gradient_dist", gradients, step)
   writer.add_text("sample_output", generated_text, step)
   ```

:::

### 19-7-2 Weights & Biases 監控

Weights & Biases（W&B）是雲端版的訓練監控工具，提供更強大的協作和比較功能。

```bash
# ==========================================
# 安裝與設定
# ==========================================
uv pip install wandb

# 登入 W&B（需要帳號）
wandb login

# ==========================================
# 在訓練程式中整合 W&B
# ==========================================
```

```python
"""
在訓練程式中整合 Weights & Biases
"""

import wandb
from transformers import Trainer, TrainingArguments

# 初始化 W&B
wandb.init(
    project="dgx-spark-pretrain",
    name="bert-domain-pretrain-v1",
    config={
        "model": "bert-base-chinese",
        "learning_rate": 5e-5,
        "batch_size": 64,
        "epochs": 3,
        "dtype": "bf16",
    },
)

# Hugging Face Trainer 會自動整合 W&B
# 只需在 TrainingArguments 中設定 report_to=["wandb"]
training_args = TrainingArguments(
    output_dir="./outputs",
    report_to=["wandb"],  # 啟用 W&B 日誌
    logging_steps=10,
    # ... 其他參數
)

# 記錄自訂指標
wandb.log({
    "custom_metric": value,
    "epoch": epoch,
})

# 記錄模型產出物
wandb.save("./domain-bert/final-model/*")

# 結束 W&B 會話
wandb.finish()
```

**W&B 的核心功能：**

| 功能 | 說明 |
|------|------|
| **儀表板** | 即時視覺化訓練指標 |
| **實驗比較** | 並排比較多次實驗的結果 |
| **超參數掃描** | 自動執行超參數搜尋 |
| **模型註冊** | 管理不同版本的模型 |
| **協作** | 團隊共享實驗結果 |
| **報告** | 生成可共享的實驗報告 |

### 19-7-3 模型評估指標

| 指標 | 全名 | 說明 | 計算方式 | 好壞判斷 |
|------|------|------|---------|---------|
| **Perplexity** | 困惑度（Perplexity） | 語言模型預測能力 | $e^{-\frac{1}{N}\sum \log P(w_i)}$ | **越低越好** |
| **BLEU** | 雙語評估替換（Bilingual Evaluation Understudy） | 生成文字與參考文字的 n-gram 重疊度 | 精確度為主的指標 | **越高越好**（0-100） |
| **ROUGE** | 面向記憶的摘要評估（Recall-Oriented Understudy） | 摘要品質評估 | 召回率為主的指標 | **越高越好**（0-100） |
| **Accuracy** | 準確率 | 分類正確的樣本比例 | 正確數 / 總數 | **越高越好**（0-1） |
| **F1 Score** | F1 分數 | 精確率和召回率的調和平均 | $2 \times \frac{P \times R}{P + R}$ | **越高越好**（0-1） |
| **MRR** | 平均倒數排名（Mean Reciprocal Rank） | 檢索系統評估 | 第一個正確結果的倒數排名平均 | **越高越好**（0-1） |
| **Spearman** | 斯皮爾曼相關係數 | 排名相關性 | 等級相關的統計量 | **越高越好**（-1 到 1） |

```python
"""
計算常見評估指標
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# ==========================================
# Perplexity（困惑度）
# ==========================================
def calculate_perplexity(loss):
    """
    根據平均損失計算 Perplexity
    
    Perplexity = exp(loss)
    越低越好，表示模型對資料的不確定性越低
    """
    return np.exp(loss)

# 範例
avg_loss = 3.5
ppl = calculate_perplexity(avg_loss)
print(f"Perplexity: {ppl:.2f}")
# Perplexity < 20：優秀
# Perplexity 20-50：良好
# Perplexity 50-100：可接受
# Perplexity > 100：需要改進

# ==========================================
# BLEU Score
# ==========================================
reference = ["這是一篇關於 DGX Spark 的文章"]
candidate = "這是一篇關於 DGX Spark 的文章"
bleu_score = sentence_bleu([reference], candidate.split())
print(f"BLEU Score: {bleu_score:.4f}")

# ==========================================
# ROUGE Score
# ==========================================
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(
    "這是一篇關於 DGX Spark 的文章",
    "DGX Spark 是一篇介紹文章"
)
print(f"ROUGE Scores: {scores}")
```

::: info 🤔 如何解讀 Perplexity？

Perplexity（困惑度）是語言模型最重要的評估指標之一。

**直觀理解**：
- Perplexity = 10 表示模型在預測下一個 token 時，相當於從 10 個等可能的詞中隨機選擇
- Perplexity = 100 表示相當於從 100 個等可能的詞中隨機選擇

**參考標準**：
| Perplexity | 評價 | 說明 |
|-----------|------|------|
| < 10 | 極佳 | 模型非常確定預測結果 |
| 10-20 | 優秀 | 模型有很好的預測能力 |
| 20-50 | 良好 | 模型有合理的預測能力 |
| 50-100 | 可接受 | 模型有基本的語言理解 |
| 100-500 | 較差 | 模型預測能力有限 |
| > 500 | 很差 | 模型幾乎是隨機預測 |

**注意**：Perplexity 的絕對值受詞彙表大小影響，比較時應使用相同詞彙表的模型。

:::

### 19-7-4 模型匯出與發布

```python
"""
模型匯出與發布完整指南
"""

import os
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 步驟 1：整理最終模型
# ==========================================
# 載入最佳 checkpoint
model = AutoModel.from_pretrained("./domain-bert/checkpoint-best")
tokenizer = AutoTokenizer.from_pretrained("./domain-bert/checkpoint-best")

# 建立發布目錄
output_dir = "./final-model"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 步驟 2：儲存模型和分詞器
# ==========================================
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 建立 model card
model_card = """---
language: zh
tags:
  - bert
  - chinese
  - domain-specific
license: mit
---

# 領域 BERT 模型

## 模型描述
這是一個在領域特定資料上預訓練的 BERT 模型。

## 訓練資料
- 資料來源：領域文字資料
- 資料量：X GB
- 處理方式：清洗、分詞、過濾

## 訓練參數
- 學習率：5e-5
- Batch Size：64
- Epochs：3
- 精度：BF16

## 使用方式
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("your-username/domain-bert")
tokenizer = AutoTokenizer.from_pretrained("your-username/domain-bert")
```

## 評估結果
- Perplexity：XX
- 下游任務準確率：XX%
"""

with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write(model_card)

# ==========================================
# 步驟 3：上傳到 Hugging Face Hub
# ==========================================
from huggingface_hub import HfApi, login

# 登入（需要 API Token）
# login(token="your-hf-token")

api = HfApi()

# 建立模型倉庫
api.create_repo(
    repo_id="your-username/domain-bert",
    repo_type="model",
    exist_ok=True,
)

# 上傳模型檔案
api.upload_folder(
    folder_path=output_dir,
    repo_id="your-username/domain-bert",
    repo_type="model",
)

print(f"模型已上傳至 https://huggingface.co/your-username/domain-bert")

# ==========================================
# 步驟 4：驗證上傳的模型
# ==========================================
# 從 Hugging Face Hub 載入模型
from transformers import pipeline

# 測試模型
classifier = pipeline(
    "fill-mask",
    model="your-username/domain-bert",
    tokenizer="your-username/domain-bert",
)

result = classifier("DGX Spark 是一台 [MASK] 電腦。")
print(f"測試結果：{result}")
```

::: warning ⚠️ 模型發布前檢查清單

- [ ] **模型檔案完整**：包含 `pytorch_model.bin`、`config.json`、`tokenizer` 相關檔案
- [ ] **Model Card 完整**：包含模型描述、訓練細節、使用方式、限制說明
- [ ] **License 明確**：指定模型的使用授權
- [ ] **評估報告**：包含主要評估指標和結果
- [ ] **使用範例**：提供至少一個使用範例
- [ ] **偏見聲明**：說明模型可能存在的偏見和限制
- [ ] **隱私合規**：確保訓練資料不包含個人隱私資訊
- [ ] **安全性測試**：測試模型是否會生成有害內容

:::

---

## 19-8 疑難排解 FAQ

### Q1：訓練時出現 CUDA Out of Memory 怎麼辦？

**原因**：GPU 記憶體不足。

**解決方案**：

| 方法 | 說明 | 記憶體節省 |
|------|------|-----------|
| **減少 Batch Size** | 降低 `per_device_train_batch_size` | 線性減少 |
| **啟用梯度累積** | 設定 `gradient_accumulation_steps=4` | 等效 batch size 不變 |
| **啟用 Gradient Checkpointing** | 用計算換記憶體 | 節省 40-60% |
| **使用 DeepSpeed ZeRO** | 分散優化器狀態 | 節省 50-80% |
| **減少序列長度** | 降低 `max_length` | 線性減少 |
| **使用 8-bit 優化器** | `optim="adamw_bnb_8bit"` | 節省 50% 優化器記憶體 |

```python
# 啟用 Gradient Checkpointing
model.gradient_checkpointing_enable()

# 使用 DeepSpeed
# 建立 ds_config.json
# {
#   "zero_optimization": {
#     "stage": 2
#   },
#   "bf16": {
#     "enabled": true
#   }
# }
```

### Q2：訓練損失不下降怎麼辦？

**可能原因與解決方案：**

| 原因 | 檢查方法 | 解決方案 |
|------|---------|---------|
| **學習率太大** | 損失一開始就很大或 NaN | 降低學習率 10 倍 |
| **學習率太小** | 損失幾乎不變 | 增加學習率 10 倍 |
| **資料有問題** | 檢查資料是否為空或格式錯誤 | 驗證資料載入 |
| **模型初始化錯誤** | 檢查權重是否為 NaN | 重新初始化模型 |
| **梯度消失** | 檢查梯度大小 | 使用 Pre-LayerNorm、增加殘差連接 |
| **梯度爆炸** | 檢查梯度是否為 NaN | 啟用梯度裁剪 `max_grad_norm=1.0` |

### Q3：如何判斷模型是否過擬合？

**過擬合的徵兆：**

| 徵兆 | 說明 |
|------|------|
| **訓練損失持續下降，驗證損失上升** | 最明顯的過擬合信號 |
| **訓練準確率接近 100%，驗證準確率低** | 模型記住了訓練資料 |
| **生成內容重複訓練資料** | 模型過度記憶 |

**解決方案：**

1. **Early Stopping**：驗證損失不再改善時停止訓練
2. **增加資料量**：更多資料能減少過擬合
3. **增加 Dropout**：`dropout=0.2` 或更高
4. **增加 Weight Decay**：`weight_decay=0.1` 或更高
5. **減少模型大小**：使用較小的模型架構

### Q4：訓練速度太慢怎麼辦？

**效能優化建議：**

| 優化方法 | 預期加速 | 實作難度 |
|---------|---------|---------|
| **啟用 BF16** | 1.5-2x | 低 |
| **啟用 PyTorch Compile** | 1.2-1.5x | 低 |
| **增加 DataLoader Workers** | 1.1-1.3x | 低 |
| **使用 Arrow 格式資料** | 1.5-2x | 中 |
| **啟用 DeepSpeed** | 1.2-1.5x | 中 |
| **最佳化資料管線** | 1.2-2x | 高 |

```python
# 啟用 PyTorch 2.0 編譯
model = torch.compile(model)

# 增加 DataLoader workers
training_args = TrainingArguments(
    dataloader_num_workers=8,
    dataloader_prefetch_factor=4,
)
```

### Q5：如何選擇合適的學習率？

**學習率選擇指南：**

| 模型規模 | 預訓練學習率 | 微調學習率 |
|---------|------------|-----------|
| 10M | 1e-3 ~ 5e-4 | 1e-4 ~ 5e-5 |
| 100M | 5e-4 ~ 1e-4 | 5e-5 ~ 1e-5 |
| 1B | 1e-4 ~ 5e-5 | 1e-5 ~ 5e-6 |
| 3B+ | 5e-5 ~ 1e-5 | 5e-6 ~ 1e-6 |

**學習率搜尋技巧：**
```python
# 學習率範圍測試
from transformers import Trainer

# 從小學習率開始，指數增長
lrs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
for lr in lrs:
    # 訓練 100 步，記錄損失
    # 選擇損失開始下降的學習率
```

---

## 19-9 本章小結

::: success ✅ 你現在知道了
- DGX Spark 的 128GB 記憶體適合預訓練 10M-3B 規模的模型
- BERT（Encoder-only）適合分類、搜尋、NER 等理解型任務
- GPT（Decoder-only）適合文字生成、對話、創意寫作等生成型任務
- autoresearch 框架讓 AI 自動設計實驗、訓練模型、分析結果
- NVFP4（4-bit）預訓練可大幅減少記憶體用量並加速訓練
- Embedding 模型是 RAG 系統的核心，領域 Embedding 能大幅提升檢索準確率
- TensorBoard 和 W&B 是訓練監控的必備工具
- Perplexity、BLEU、ROUGE 是評估語言模型的重要指標
- 模型發布前需要完整的檢查清單確保品質和安全性
:::

::: tip 🚀 第五篇完結！
恭喜！你已經完成了「模型微調與訓練」篇。

在這一篇中，你學會了：
- 如何微調大型語言模型（第 16-17 章）
- 如何微調影像生成模型（第 18 章）
- 如何從零預訓練中小型模型（第 19 章）

接下來要進入最酷的部分 — **多模態 AI 與智慧代理**！

你將學會：
- 讓 AI 看懂圖片和影片
- 建立能自主行動的 AI Agent
- 整合多種 AI 能力打造智慧應用

👉 [前往第 20 章：多模態推論與即時視覺 AI →](/guide/chapter20/)
:::

::: info 📝 上一章
← [回到第 18 章：影像模型微調](/guide/chapter18/)
:::
