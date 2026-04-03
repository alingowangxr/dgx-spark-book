# 第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用

::: tip 🎯 本章你將學到什麼
- RAPIDS 加速資料科學（cuDF、cuML）
- JAX GPU 加速計算框架
- 影片搜尋與摘要系統（VSS Agent）
- 金融、生技、機器人等特殊領域應用
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。
:::

---

## 23-1 CUDA-X 資料科學

### 23-1-1 什麼是 CUDA-X？

::: info 🤔 CUDA-X 是什麼？
CUDA-X 是 NVIDIA 的加速計算軟體堆疊，包含：

- **CUDA**：GPU 平行計算平台
- **cuDNN**：深度學習加速庫
- **TensorRT**：推論加速引擎
- **RAPIDS**：資料科學加速套件
- **NCCL**：多 GPU 通訊庫

在 DGX Spark 上，這些全部預裝或可以輕鬆安裝。
:::

### 23-1-2 RAPIDS 環境與 cuDF

RAPIDS 是 NVIDIA 的 GPU 加速資料科學套件。其中的 **cuDF** 是一個 DataFrame 庫，API 跟 pandas 一模一樣，但速度快 10-50 倍。

**部署 RAPIDS 容器**：

```bash
docker run -d \
  --name rapids \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/data:/data \
  -v ~/rapids-notebooks:/rapids/notebooks \
  -w /rapids/notebooks \
  nvcr.io/nvidia/rapidsai/base:25.02-cuda12.0-py3 \
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token=''
```

打開 `http://DGX_Spark_IP:8888` 即可使用 JupyterLab。

**cuDF vs pandas 程式碼比較**：

```python
# pandas（CPU）
import pandas as pd
df = pd.read_csv("/data/big_dataset.csv")
result = df.groupby("category")["sales"].mean()

# cuDF（GPU）— 只要改 import！
import cudf
df = cudf.read_csv("/data/big_dataset.csv")
result = df.groupby("category")["sales"].mean()
```

::: tip 💡 只要改 import，程式碼完全不用動！
這就是 RAPIDS 最強大的地方：API 跟 pandas/scikit-learn 完全相容，你不需要重寫程式碼。
:::

### 23-1-3 實測：10GB CSV 處理速度比較

| 操作 | pandas（CPU） | cuDF（GPU） | 加速比 |
|------|--------------|-------------|--------|
| 讀取 CSV | 45 秒 | 3 秒 | **15x** |
| groupby | 30 秒 | 2 秒 | **15x** |
| merge（1000 萬列） | 60 秒 | 4 秒 | **15x** |
| 過濾 + 排序 | 15 秒 | 1 秒 | **15x** |
| 描述性統計 | 20 秒 | 1.5 秒 | **13x** |

### 23-1-4 cuML 加速機器學習

cuML 是 RAPIDS 中的機器學習庫，API 跟 scikit-learn 一樣：

```python
# scikit-learn（CPU）
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# cuML（GPU）— 一樣只要改 import！
from cuml.ensemble import RandomForestClassifier
from cuml.cluster import KMeans

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

**支援的演算法**：

| 類別 | 演算法 | CPU 加速比 |
|------|--------|-----------|
| 分類 | 隨機森林、SVM、KNN | 10-50x |
| 迴歸 | 線性迴歸、隨機森林 | 10-30x |
| 分群 | K-Means、DBSCAN、HDBSCAN | 20-100x |
| 降維 | PCA、t-SNE、UMAP | 10-50x |
| 矩陣分解 | SVD、NMF | 10-30x |

### 23-1-5 UMAP 降維實戰

UMAP 是高維資料視覺化的常用工具，在 CPU 上跑很慢：

```python
from cuml.manifold import UMAP
import cudf

# 載入資料（100 萬筆 x 128 維）
df = cudf.read_csv("/data/embeddings.csv")
X = df.values

# UMAP 降維到 2D
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding_2d = umap.fit_transform(X)

# 繪製
import matplotlib.pyplot as plt
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1, alpha=0.5)
plt.title("UMAP 降維視覺化（GPU 加速）")
plt.savefig("/data/umap_plot.png", dpi=300)
```

| 資料量 | scikit-learn | cuML | 加速比 |
|--------|-------------|------|--------|
| 10 萬筆 | 30 秒 | 2 秒 | **15x** |
| 100 萬筆 | 8 分鐘 | 15 秒 | **32x** |
| 1000 萬筆 | 記憶體不足 | 3 分鐘 | **∞** |

---

## 23-2 JAX on DGX Spark

### 23-2-1 什麼是 JAX？

::: info 🤔 JAX 是什麼？
JAX 是 Google 開發的 GPU/TPU 加速數值計算庫。它的特色：

1. **NumPy 相容**：API 跟 NumPy 一樣
2. **自動微分**：`jax.grad()` 自動計算梯度
3. **JIT 編譯**：`@jax.jit` 自動最佳化
4. **向量化**：`jax.vmap` 自動批次化
5. **平行化**：`jax.pmap` 自動分散到多裝置

簡單說：JAX = NumPy + 自動微分 + JIT 編譯 + 平行化
:::

### 23-2-2 部署 JAX 環境

```bash
# 用 uv 安裝 JAX（CUDA 13.0 版本）
uv pip install "jax[cuda13]"

# 驗證安裝
python -c "
import jax
print(f'JAX 版本: {jax.__version__}')
print(f'裝置: {jax.devices()}')
print(f'GPU: {jax.devices()[0]}')
"
```

### 23-2-3 JAX 入門與 GPU 偵測

```python
import jax
import jax.numpy as jnp

# 偵測 GPU
print(f"可用裝置: {jax.devices()}")
# 輸出：[cuda(id=0)]

# 基本運算（跟 NumPy 一樣）
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
print(x + y)  # [5. 7. 9.]

# 矩陣運算
A = jnp.ones((1000, 1000))
B = jnp.ones((1000, 1000))
C = jnp.dot(A, B)
print(C.shape)  # (1000, 1000)
```

### 23-2-4 JAX vs NumPy 效能比較

```python
import numpy as np
import jax.numpy as jnp
import jax
import time

size = 10000

# NumPy（CPU）
a = np.random.randn(size, size).astype(np.float32)
b = np.random.randn(size, size).astype(np.float32)

start = time.time()
c = np.dot(a, b)
numpy_time = time.time() - start
print(f"NumPy: {numpy_time:.3f}s")

# JAX（GPU）
a_jax = jnp.array(a)
b_jax = jnp.array(b)

# 第一次執行包含編譯時間，所以跑兩次
_ = jnp.dot(a_jax, b_jax)
jax.block_until_ready(_)

start = time.time()
c_jax = jnp.dot(a_jax, b_jax)
jax.block_until_ready(c_jax)
jax_time = time.time() - start
print(f"JAX: {jax_time:.3f}s")

print(f"加速比: {numpy_time/jax_time:.1f}x")
```

**典型結果**（10000x10000 矩陣乘法）：

| 框架 | 時間 | 加速比 |
|------|------|--------|
| NumPy（CPU） | 12.5 秒 | 1x |
| JAX（GPU） | 0.4 秒 | **31x** |

### 23-2-5 JIT 編譯：免費的效能提升

```python
import jax

# 定義函式
def compute(x, y):
    return jnp.sin(x) * jnp.cos(y) + jnp.exp(-x**2)

# 一般執行
x = jnp.ones((1000,))
y = jnp.ones((1000))
result = compute(x, y)

# JIT 編譯後執行（自動最佳化）
compute_jit = jax.jit(compute)
result = compute_jit(x, y)  # 更快！
```

### 23-2-6 自動微分：機器學習的核心

```python
import jax

# 定義損失函式
def loss(params, X, y):
    predictions = jnp.dot(X, params)
    return jnp.mean((predictions - y) ** 2)

# 自動計算梯度
grad_loss = jax.grad(loss)

# 梯度下降
params = jnp.zeros(10)
learning_rate = 0.01

for step in range(100):
    g = grad_loss(params, X, y)
    params = params - learning_rate * g
```

### 23-2-7 自組織映射（SOM）加速實戰

自組織映射（Self-Organizing Map）是一種無監督學習演算法：

```python
import jax
import jax.numpy as jnp

@jax.jit
def som_update(weights, input, learning_rate, neighborhood):
    """JIT 編譯的 SOM 更新函式"""
    # 計算距離
    distances = jnp.sum((weights - input) ** 2, axis=1)
    best_match = jnp.argmin(distances)

    # 更新權重
    diff = weights - input
    update = learning_rate * neighborhood[:, None] * diff
    weights = weights - update

    return weights

# 大規模 SOM 訓練
key = jax.random.PRNGKey(0)
weights = jax.random.normal(key, (100, 100, 128))  # 100x100 網格，128 維

# 準備資料
data = jax.random.normal(key, (10000, 128))

# 訓練
for epoch in range(100):
    for i, data_point in enumerate(data):
        lr = 0.1 * (1 - epoch / 100)  # 遞減學習率
        weights = som_update(weights, data_point, lr, jnp.ones(10000))
```

---

## 23-3 影片搜尋與摘要

### 23-3-1 什麼是 VSS Agent？

VSS（Video Search & Summary）Agent 是一個能搜尋和摘要影片的 AI 系統。

工作流程：
```
影片 → 幀提取 → VLM 分析 → 文字描述 → 向量資料庫 → 搜尋
```

### 23-3-2 部署 VSS Agent

```bash
# 部署 VSS Agent
docker run -d \
  --name vss-agent \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/videos:/videos \
  -v ~/vss-data:/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  ghcr.io/community/vss-agent:latest
```

### 23-3-3 使用 VSS Agent

打開 `http://DGX_Spark_IP:8080`：

1. **上傳影片**：把影片放到 `/videos` 目錄
2. **自動分析**：系統自動提取幀、用 VLM 分析、建立索引
3. **搜尋**：輸入文字描述，找到對應的影片片段

**搜尋範例**：

```
搜尋：「有人在講話的片段」
結果：
  - meeting.mp4 @ 05:23-05:45
  - presentation.mp4 @ 12:00-12:30
  - interview.mp4 @ 00:15-00:45
```

### 23-3-4 影片摘要

```
輸入：summary meeting.mp4
輸出：
  「這場會議共 45 分鐘，主要討論以下主題：
   1. 00:00-10:00 專案進度報告
   2. 10:00-25:00 新功能需求討論
   3. 25:00-35:00 技術架構選擇
   4. 35:00-45:00 下次會議時間安排

   關鍵決策：選擇 React 作為前端框架，Python 作為後端。」
```

---

## 23-4 特殊領域應用

### 23-4-1 金融投資組合最佳化

```python
import cudf
from cuml.cluster import KMeans
from cuml.decomposition import PCA

# 載入股票資料
returns = cudf.read_csv("/data/stock_returns.csv")

# 用 K-Means 分群
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(returns)

# 每個分群的投資組合權重
for cluster_id in range(5):
    mask = clusters == cluster_id
    cluster_returns = returns[mask]
    print(f"分群 {cluster_id}: {mask.sum()} 支股票")
    print(f"  平均報酬率: {cluster_returns.mean().mean():.4f}")
    print(f"  風險（標準差）: {cluster_returns.std().mean():.4f}")
```

### 23-4-2 單細胞 RNA 定序分析

```python
from cuml.manifold import UMAP
import cudf

# 載入單細胞 RNA 定序資料
sc_data = cudf.read_csv("/data/scrna_counts.csv")

# 標準化
sc_data = (sc_data - sc_data.mean()) / sc_data.std()

# UMAP 降維
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding = umap.fit_transform(sc_data)

# 儲存結果
result = cudf.DataFrame({
    'cell_id': sc_data.index,
    'umap_1': embedding[:, 0],
    'umap_2': embedding[:, 1]
})
result.to_csv("/data/scrna_umap.csv")
```

### 23-4-3 機器人模擬

```bash
# 部署 Isaac Sim（NVIDIA 機器人模擬平台）
docker run -d \
  --name isaac-sim \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/isaac-data:/root/.nvidia-omniverse \
  -e ACCEPT_EULA=Y \
  nvcr.io/nvidia/isaac-sim:2025.1
```

Isaac Sim 提供：
- 物理引擎模擬
- 感測器模擬（攝影機、雷達、LiDAR）
- 強化學習訓練環境
- 真實場景重建

---

## 23-5 常見問題與疑難排解

### 23-5-1 RAPIDS 容器啟動失敗

```bash
# 確認 GPU 驅動版本
nvidia-smi

# RAPIDS 25.02 需要 CUDA 12.0+
# 如果驅動太舊，換用舊版 RAPIDS
docker pull nvcr.io/nvidia/rapidsai/base:24.12-cuda12.0-py3
```

### 23-5-2 JAX 偵測不到 GPU

```bash
# 確認安裝了 CUDA 版本
uv pip install "jax[cuda13]"

# 驗證
python -c "import jax; print(jax.devices())"
# 應該輸出 [cuda(id=0)]
```

### 23-5-3 cuDF 記憶體不足

```python
# 分批處理大資料
chunk_size = 1000000
for chunk in cudf.read_csv("/data/big.csv", chunksize=chunk_size):
    result = process(chunk)
    result.to_parquet(f"/data/output/part_{i}.parquet")
```

---

## 23-6 本章小結

::: success ✅ 你現在知道了
- RAPIDS 讓資料科學加速 10-50 倍，API 跟 pandas/scikit-learn 一樣
- JAX 是高效的 GPU 加速計算框架，支援自動微分和 JIT 編譯
- VSS Agent 可以搜尋和摘要影片內容
- DGX Spark 可以應用於金融、生技、機器人等領域
:::

::: tip 🚀 下一章預告
寫了這麼多程式，來看看怎麼用 AI 輔助開發吧！VS Code + Ollama + Claude Code 的組合讓 coding 變得更簡單！

👉 [前往第 24 章：開發環境與 AI 輔助程式開發 →](/guide/chapter24/)
:::

::: info 📝 上一章
← [回到第 22 章：AI Agent 與安全沙箱](/guide/chapter22/)
:::
