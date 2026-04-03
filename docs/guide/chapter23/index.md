# 第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用

::: tip 🎯 本章你將學到什麼
- RAPIDS 加速資料科學
- JAX GPU 加速計算
- 影片搜尋與摘要
- 特殊領域應用
:::

---

## 23-1 CUDA-X 資料科學

### 23-1-1 RAPIDS 環境與 cuDF

RAPIDS 是 NVIDIA 的 GPU 加速資料科學套件。cuDF 是其中的 DataFrame 庫，API 跟 pandas 一樣。

```bash
docker run -d \
  --name rapids \
  --gpus all \
  --network host \
  --shm-size=16g \
  -v ~/data:/data \
  -w /data \
  nvcr.io/nvidia/rapidsai/base:25.02-cuda12.0-py3
```

```python
import cudf

# 跟 pandas 一樣的 API
df = cudf.read_csv("/data/big_dataset.csv")
print(df.head())
print(df.describe())

# 過濾
filtered = df[df["column"] > 100]

# 分組
grouped = df.groupby("category").mean()
```

::: tip 💡 速度比較
| 操作 | pandas | cuDF | 加速比 |
|------|--------|------|--------|
| 讀取 10GB CSV | 45 秒 | 3 秒 | **15x** |
| groupby | 30 秒 | 2 秒 | **15x** |
| merge | 60 秒 | 4 秒 | **15x** |
:::

### 23-1-2 cuML 加速機器學習

```python
from cuml.ensemble import RandomForestClassifier
from cuml.cluster import HDBSCAN
from cuml.decomposition import PCA

# 隨機森林（比 sklearn 快 10-50x）
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 分群
cluster = HDBSCAN(min_cluster_size=10)
labels = cluster.fit_predict(X)

# 降維
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
```

### 23-1-3 GPU 監控

```bash
# 在 RAPIDS 容器中
nvidia-smi
```

---

## 23-2 JAX on DGX Spark

### 23-2-1 部署 JAX 環境

```bash
uv pip install "jax[cuda13]"
```

### 23-2-2 JAX 入門與 GPU 偵測

```python
import jax
import jax.numpy as jnp

print(f"JAX 裝置: {jax.devices()}")
# 輸出：[cuda(id=0)]

# 基本運算
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
print(x + y)  # [5. 7. 9.]
```

### 23-2-3 JAX vs NumPy 效能比較

```python
import numpy as np
import time

# NumPy
size = 10000
a = np.random.randn(size, size)
b = np.random.randn(size, size)

start = time.time()
c = np.dot(a, b)
numpy_time = time.time() - start

# JAX
a_jax = jnp.array(a)
b_jax = jnp.array(b)

start = time.time()
c_jax = jnp.dot(a_jax, b_jax)
jax.block_until_ready(c_jax)
jax_time = time.time() - start

print(f"NumPy: {numpy_time:.3f}s, JAX: {jax_time:.3f}s, 加速: {numpy_time/jax_time:.1f}x")
```

### 23-2-4 自組織映射（SOM）加速實戰

```python
from jax import jit, grad, vmap

@jit
def som_update(weights, input, learning_rate):
    """JIT 編譯的 SOM 更新函式"""
    diff = weights - input
    weights = weights - learning_rate * diff
    return weights

# 大規模 SOM 訓練
weights = jnp.zeros((100, 100, 128))
for epoch in range(100):
    for data_point in data:
        weights = som_update(weights, data_point, 0.01)
```

---

## 23-3 影片搜尋與摘要

### 23-3-1 部署架構

```
影片 → 幀提取 → VLM 分析 → 向量資料庫 → 搜尋
```

### 23-3-2 VSS Agent 介面

```bash
docker run -d \
  --name vss-agent \
  --gpus all \
  --network host \
  -v ~/videos:/videos \
  ghcr.io/community/vss-agent:latest
```

---

## 23-4 特殊領域應用

### 23-4-1 金融投資組合最佳化

```python
import cudf
from cuml.cluster import KMeans

# 用 GPU 加速投資組合分析
returns = cudf.read_csv("/data/stock_returns.csv")
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(returns)
```

### 23-4-2 單細胞 RNA 定序分析

RAPIDS 可以加速單細胞 RNA 定序資料的處理：

```python
from cuml.manifold import UMAP

# UMAP 降維（比 CPU 快 20x）
umap = UMAP(n_components=2, n_neighbors=15)
embedding = umap.fit_transform(scrna_data)
```

### 23-4-3 機器人模擬

```bash
# Isaac Sim 機器人模擬
docker run -d \
  --name isaac-sim \
  --gpus all \
  --network host \
  nvcr.io/nvidia/isaac-sim:2025.1
```

---

## 23-5 本章小結

::: success ✅ 你現在知道了
- RAPIDS 讓資料科學加速 10-50 倍
- JAX 是高效的 GPU 加速計算框架
- DGX Spark 可以應用於金融、生技、機器人等領域
:::

::: tip 🚀 下一章預告
寫了這麼多程式，來看看怎麼用 AI 輔助開發吧！VS Code + Ollama + Claude Code 的組合讓 coding 變得更簡單！

👉 [前往第 24 章：開發環境與 AI 輔助程式開發 →](/guide/chapter24/)
:::

::: info 📝 上一章
← [回到第 22 章：AI Agent 與安全沙箱](/guide/chapter22/)
:::
