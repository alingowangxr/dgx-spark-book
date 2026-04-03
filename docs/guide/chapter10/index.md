# 第 10 章：TensorRT-LLM — NVIDIA 原生加速引擎

::: tip 🎯 本章你將學到什麼
- TensorRT-LLM 的架構和支援模型
- 單機部署與 quickstart 驗證
- 整合 Open WebUI
- 疑難排解
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。
:::

---

## 10-1 TRT-LLM 架構與支援模型

### 10-1-1 支援的模型

TensorRT-LLM（TRT-LLM）是 NVIDIA 官方開發的 LLM 推論加速引擎。

目前支援的模型家族：

| 模型家族 | 代表模型 | 備註 |
|---------|---------|------|
| Llama | Llama 3.1 8B/70B | 廣泛支援 |
| Qwen | Qwen3-8B | 中文最佳 |
| Mistral | Mistral 7B | 輕量高效 |
| Gemma | Gemma 2 9B | Google 開源 |
| Falcon | Falcon 40B | |
| Baichuan | Baichuan 2 13B | 中文 |

::: info 🤔 TRT-LLM vs. vLLM 的核心差異
- **vLLM**：Python 為主，靈活性高，社群活躍
- **TRT-LLM**：C++/CUDA 為主，極致效能，NVIDIA 官方最佳化

簡單說：vLLM 像是改裝車（好調校），TRT-LLM 像是 F1 賽車（極致效能但較難改動）。
:::

### 10-1-2 TRT-LLM vs. vLLM

| 特性 | TRT-LLM | vLLM |
|------|---------|------|
| 開發者 | NVIDIA 官方 | 社群（UC Berkeley） |
| 語言 | C++/CUDA | Python |
| 效能 | **極致** | 高 |
| 靈活性 | 低（需重新編譯） | 高 |
| 模型支援 | 較少（但持續增加） | 較多 |
| 部署難度 | 較高 | 較低 |
| 適合場景 | 生產環境、效能優先 | 開發、測試、一般生產 |

---

## 10-2 單機部署

### 10-2-1 確認 Docker 權限與 GPU

```bash
docker ps
nvidia-smi
```

### 10-2-2 設定環境變數並拉取容器

```bash
# 設定 NGC 認證（從 NGC 網站取得 API Key）
export NGC_API_KEY="你的API_KEY"

# 登入 NGC Docker Registry
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 拉取 TRT-LLM 容器
docker pull nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3
```

### 10-2-3 quickstart 驗證：LLM

```bash
# 執行 quickstart 腳本
docker run --rm \
  --gpus all \
  --network host \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  python3 /opt/tritonserver/tensorrtllm/backends/tensorrtllm/scripts/quickstart.py
```

這個腳本會：
1. 下載一個小型測試模型
2. 編譯為 TensorRT engine
3. 執行推論測試
4. 顯示效能數據

### 10-2-4 quickstart 驗證：多模態（VLM）

```bash
# 多模態 quickstart
docker run --rm \
  --gpus all \
  --network host \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  python3 /opt/tritonserver/tensorrtllm/backends/tensorrtllm/scripts/quickstart_vlm.py
```

### 10-2-5 啟動 OpenAI 相容 API 伺服器

```bash
docker run -d \
  --name trt-llm \
  --gpus all \
  --network host \
  --shm-size=8g \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
  tritonserver \
  --model-repository=/models \
  --allow-http=true
```

測試 API：

```bash
curl http://localhost:8000/v2/models/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "你好！",
    "max_tokens": 100
  }'
```

---

## 10-3 整合 Open WebUI

### 10-3-1 啟動 Open WebUI

如果 Open WebUI 已經在執行（第 6 章），只需要添加 TRT-LLM 作為新的模型提供者：

1. 打開 Open WebUI **Admin Panel**
2. 進入 **Settings → Connections**
3. 添加新的 OpenAI 相容端點：
   - URL：`http://localhost:8000/v1`
   - API Key：任意填寫（TRT-LLM 不需要認證）
4. 點擊 **Save**

現在你可以在模型選擇器中看到 TRT-LLM 的模型。

### 10-3-2 清理 Open WebUI

如果不需要了：

```bash
docker stop open-webui
docker rm open-webui
```

---

## 10-4 單機清理

```bash
# 停止並移除容器
docker stop trt-llm
docker rm trt-llm

# 移除映像檔
docker rmi nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# 清理模型
rm -rf /models
```

---

## 10-5 疑難排解

### Q：Docker pull 失敗？

```bash
# 確認 NGC API Key 正確
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

# 如果還是失敗，檢查網路
ping nvcr.io
```

### Q：TRT-LLM 編譯模型失敗？

TRT-LLM 需要先把模型編譯為 TensorRT engine。如果編譯失敗：

1. 確認模型在支援列表中
2. 確認有足夠的記憶體
3. 查看編譯日誌：
   ```bash
   docker logs trt-llm
   ```

### Q：GPU 記憶體不足？

```bash
# 降低 batch size
# 在啟動指令中加入
--max-batch-size 1

# 或減少上下文長度
--max-input-len 4096
```

---

## 10-6 本章小結

::: success ✅ 你現在知道了
- TensorRT-LLM 是 NVIDIA 官方的極致效能推論引擎
- 需要從 NGC 拉取容器映像
- quickstart 腳本可以快速驗證環境
- 可以整合到 Open WebUI 中使用
:::

::: tip 🚀 下一章預告
接下來我們要介紹另一個強大的推論框架 — SGLang，它帶來了獨特的 RadixAttention 和推測性解碼技術！

👉 [前往第 11 章：SGLang 與推測性解碼 →](/guide/chapter11/)
:::

::: info 📝 上一章
← [回到第 9 章：vLLM](/guide/chapter9/)
:::
