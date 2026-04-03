# 第 11 章：SGLang 與推測性解碼

::: tip 🎯 本章你將學到什麼
- SGLang 的核心技術：RadixAttention 和結構化生成
- 部署 SGLang 容器
- 推測性解碼：EAGLE-3 vs. Draft-Target
- 疑難排解
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。
:::

---

## 11-1 SGLang 推論框架

### 11-1-1 RadixAttention

::: info 🤔 什麼是 RadixAttention？
RadixAttention 是 SGLang 的獨家技術。它把所有對話歷史組織成一個「前綴樹」（Radix Tree），讓相同前綴的請求可以共享計算結果。

舉例來說，如果 10 個使用者都問了類似開頭的問題，RadixAttention 只需要計算一次前綴，大幅節省資源。
:::

**效果**：
- 重複前綴的請求可以快取
- 多輪對話越來越快
- 適合 RAG 場景（相同的 system prompt）

### 11-1-2 結構化生成

SGLang 支援結構化生成，讓模型輸出特定格式的內容（如 JSON）：

```python
import sglang as sgl

@sgl.function
def json_extraction(s, text):
    s += "請從以下文字中提取資訊，輸出 JSON 格式：\n"
    s += text + "\n"
    s += sgl.gen("json_output", max_tokens=256, regex=r'\{.*\}')

# 輸出一定是合法的 JSON
```

### 11-1-3 SGLang vs. vLLM vs. TRT-LLM

| 特性 | SGLang | vLLM | TRT-LLM |
|------|--------|------|---------|
| 開發者 | UC Berkeley + 社群 | UC Berkeley + 社群 | NVIDIA |
| RadixAttention | ✅ 獨家 | ❌ | ❌ |
| 結構化生成 | ✅ 原生支援 | 有限 | 有限 |
| 推測性解碼 | ✅ EAGLE-3 | ✅ | ✅ |
| 效能 | 高 | 高 | 最高 |
| 靈活性 | **最高** | 高 | 低 |
| 適合場景 | 研究、結構化生成 | 一般生產環境 | 極致效能 |

### 11-1-4 支援的模型

SGLang 支援大部分主流的開源模型：
- Llama 3.x 系列
- Qwen 系列
- Mistral 系列
- Gemma 系列
- DeepSeek 系列

---

## 11-2 部署 SGLang

### 11-2-1 驗證環境

```bash
docker --version
nvidia-smi
```

### 11-2-2 拉取 SGLang 容器

```bash
docker pull lmsysorg/sglang:latest
```

### 11-2-3 啟動推論伺服器

```bash
docker run -d \
  --name sglang \
  --gpus all \
  --network host \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85
```

### 11-2-4 測試 API

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好！",
    "sampling_params": {
      "max_new_tokens": 100,
      "temperature": 0.7
    }
  }'
```

### 11-2-5 換用更大的模型

```bash
docker run -d \
  --name sglang-large \
  --gpus all \
  --network host \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B-A14B-NVFP4 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85
```

### 11-2-6 離線推論

SGLang 也支援離線批次推論：

```python
import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A: " + gen("answer", max_tokens=256)

ret = text_qa.run(question="什麼是 DGX Spark？")
print(ret["answer"])
```

### 11-2-7 清理

```bash
docker stop sglang
docker rm sglang
docker rmi lmsysorg/sglang:latest
```

---

## 11-3 推測性解碼

### 11-3-1 EAGLE-3

EAGLE-3 是 SGLang 支援的一種推測性解碼方法。它訓練一個小型的「草稿模型」來預測大模型的輸出。

```bash
docker run -d \
  --name sglang-eagle \
  --gpus all \
  --network host \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path Qwen/Qwen3-8B-EAGLE-3 \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 64
```

::: tip 💡 EAGLE-3 的效果
在 DGX Spark 上，EAGLE-3 可以帶來：
- 推論速度提升：1.5-2.5 倍
- 額外記憶體用量：~5-10 GB（草稿模型）
:::

### 11-3-2 Draft-Target

Draft-Target 是另一種推測性解碼方法，使用完全不同的模型作為草稿模型。

```bash
docker run -d \
  --name sglang-draft \
  --gpus all \
  --network host \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B \
  --speculative-algorithm DRAFT_TARGET \
  --speculative-draft-model-path Qwen/Qwen3-8B \
  --speculative-num-steps 3
```

這裡用 Qwen3-8B 作為 Qwen3.5-122B 的草稿模型。

### 11-3-3 EAGLE-3 vs. Draft-Target

| 特性 | EAGLE-3 | Draft-Target |
|------|---------|-------------|
| 草稿模型 | 專門訓練的 EAGLE 模型 | 任意小模型 |
| 加速比 | 2-2.5x | 1.5-2x |
| 記憶體用量 | 中等 | 較低 |
| 設定難度 | 高（需要特定模型） | 低（任意小模型） |
| 推薦場景 | 追求極致速度 | 快速設定 |

### 11-3-4 清理

```bash
docker stop sglang-eagle sglang-draft
docker rm sglang-eagle sglang-draft
```

---

## 11-4 疑難排解

### 11-4-1 SGLang 常見問題

**Q：伺服器啟動後無法連線？**

```bash
# 確認 port 正確
curl http://localhost:30000/health

# 查看日誌
docker logs sglang
```

**Q：記憶體不足？**

```bash
# 降低 mem-fraction-static
--mem-fraction-static 0.75
```

### 11-4-2 推測性解碼常見問題

**Q：EAGLE-3 模型下載失敗？**

確認模型名稱正確，並且有足夠的磁碟空間。

**Q：加速效果不明顯？**

推測性解碼的效果取決於：
- 草稿模型和目標模型的匹配度
- 輸出內容的類型（程式碼加速效果較好，創意文字效果較差）

---

## 11-5 本章小結

::: success ✅ 你現在知道了
- SGLang 的 RadixAttention 可以快取重複前綴，加速多輪對話
- 結構化生成讓模型輸出特定格式的內容
- EAGLE-3 和 Draft-Target 是兩種推測性解碼方法
- SGLang 在靈活性和研究用途上表現最佳
:::

::: tip 🚀 下一章預告
我們已經介紹了六種推論引擎，到底該選哪個？下一章我們要做一次全面的比較，並介紹 NVIDIA 的 NIM 推論微服務！

👉 [前往第 12 章：NIM 推論微服務與引擎總比較 →](/guide/chapter12/)
:::

::: info 📝 上一章
← [回到第 10 章：TensorRT-LLM](/guide/chapter10/)
:::
