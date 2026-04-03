# 第 25 章：多機互連與分散式運算

::: tip 🎯 本章你將學到什麼
- 雙機直連與網路設定
- NCCL 分散式通訊
- 三機環狀互連
- 分散式推論實戰：雙機跑 235B 模型
:::

---

## 25-1 硬體準備

要進行多機互連，你需要：
- 2 台或以上的 DGX Spark
- DAC 線（Direct Attach Cable）或 200GbE 交換器
- 網路線

## 25-2 雙機直連

### 25-2-1 連接與網路設定

```bash
# 在兩台機器上分別設定
sudo nano /etc/netplan/02-direct-connect.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses:
        - 10.0.0.1/24  # 第一台用 .1
      dhcp4: no
```

第二台用 `10.0.0.2/24`。

### 25-2-2 無密碼 SSH 設定

```bash
# 在第一台產生金鑰
ssh-keygen -t ed25519

# 複製到第二台
ssh-copy-id 使用者名稱@10.0.0.2

# 反向也做一遍
```

## 25-3 NCCL 分散式通訊

### 25-3-1 編譯安裝

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j $(nproc)
sudo make install
```

### 25-3-2 效能測試

```bash
cd nccl-tests
make MPI=1
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
```

## 25-4 三機環狀互連

### 25-4-1 環形拓撲

```
Spark 1 ←→ Spark 2 ←→ Spark 3 ←→ Spark 1
```

### 25-4-2 網路設定

每台機器設定兩個網路介面，分別連接相鄰兩台。

## 25-5 多機交換器互連

### 25-5-1 交換器設定

使用 200GbE 交換器，所有 DGX Spark 連到同一台交換器。

### 25-5-2 網路與測試

所有機器在同一個子網路中，可以直接通訊。

## 25-6 分散式推論實戰：雙機跑 235B 模型

### 25-6-1 建立 Ray 叢集

```bash
# 在主節點
ray start --head --port=6379

# 在工作節點
ray start --address=主節點IP:6379
```

### 25-6-2 啟動 vLLM 分散式推論

```python
import ray
from vllm import LLM

ray.init(address="auto")

llm = LLM(
    model="Qwen/Qwen3.5-235B",
    tensor_parallel_size=2,  # 兩台機器
    distributed_executor_backend="ray",
)
```

### 25-6-3 測試推論

```python
output = llm.generate("你好，請介紹 DGX Spark 多機叢集")
print(output[0].outputs[0].text)
```

## 25-7 三種拓撲比較

| 拓撲 | 頻寬 | 延遲 | 適合場景 |
|------|------|------|---------|
| 雙機直連 | 最高 | 最低 | 兩機分散式推論 |
| 三機環狀 | 中等 | 中等 | 三機訓練 |
| 交換器 | 高 | 低 | 多機叢集 |

## 25-8 回復設定

```bash
# 停用多機設定
sudo netplan revert

# 停止 Ray
ray stop
```

## 25-9 本章小結

::: success ✅ 你現在知道了
- 多台 DGX Spark 可以互連，跑更大的模型
- NCCL 是分散式通訊的標準
- Ray + vLLM 可以實現分散式推論
- 雙機可以跑 235B 等級的超大模型
:::

::: tip 🎉 恭喜！
你已經完成了整本書的所有章節！從硬體認識到多機叢集，你現在是 DGX Spark 的專家了！

👉 [回到首頁](/) | [查看附錄](/guide/appendix-a/)
:::

::: info 📝 上一章
← [回到第 24 章：開發環境與 AI 輔助程式開發](/guide/chapter24/)
:::
