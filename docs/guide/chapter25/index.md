# 第 25 章：多機互連與分散式運算

::: tip 🎯 本章你將學到什麼
- 雙機直連與網路設定
- NCCL 分散式通訊
- 三機環狀互連
- 多機交換器互連
- 分散式推論實戰：雙機跑 235B 模型
- 三種拓撲比較
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。需要 2 台以上的 DGX Spark。
:::

---

## 25-1 硬體準備

### 25-1-1 需要什麼？

| 設備 | 數量 | 用途 |
|------|------|------|
| DGX Spark | 2+ 台 | 計算節點 |
| DAC 線（Direct Attach Cable） | 1+ 條 | 雙機直連 |
| 200GbE 交換器 | 1 台（選配） | 三機以上互連 |
| 網路線（Cat 6a） | 若干 | 一般網路連線 |

### 25-1-2 為什麼要多機互連？

::: info 🤔 單機 128GB 不夠嗎？
單機 128GB 已經能跑很多模型了，但有些場景需要多機：

1. **跑超大模型**：235B、405B 模型需要 200GB+ 記憶體
2. **加速訓練**：多台機器並行訓練，時間縮短
3. **高可用性**：一台掛了，另一台繼續服務
4. **分散式推論**：同時服務更多使用者
:::

---

## 25-2 雙機直連

### 25-2-1 物理連接

用 DAC 線連接兩台 DGX Spark 的 200GbE 埠：

```
DGX Spark #1 (200GbE) ←── DAC 線 ──→ DGX Spark #2 (200GbE)
```

::: tip 💡 DAC 線是什麼？
DAC（Direct Attach Cable）是一種高速銅纜線，兩端已經接好連接器。比光纖便宜，適合短距離（5 公尺以內）連接。

推薦規格：
- 200GbE DAC（SFP56）
- 長度：1-3 公尺
- 價格：約 $20-50 美元
:::

### 25-2-2 網路設定

**第一台 DGX Spark（10.0.0.1）**：

```bash
sudo nano /etc/netplan/02-direct-connect.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses:
        - 10.0.0.1/24
      dhcp4: no
      mtu: 9000
```

**第二台 DGX Spark（10.0.0.2）**：

```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses:
        - 10.0.0.2/24
      dhcp4: no
      mtu: 9000
```

套用設定：

```bash
sudo netplan apply
```

**測試連線**：

```bash
# 在第一台上
ping 10.0.0.2

# 在第二台上
ping 10.0.0.1
```

::: tip 💡 MTU 9000 是什麼？
MTU（Maximum Transmission Unit）是網路封包的最大大小。預設是 1500，設為 9000（Jumbo Frame）可以減少封包數量，提升大資料傳輸的效能。
:::

### 25-2-3 無密碼 SSH 設定

**在第一台 DGX Spark 上**：

```bash
# 產生 SSH 金鑰（如果還沒有的話）
ssh-keygen -t ed25519 -C "dgx-spark-1"
# 一路按 Enter

# 把公鑰複製到第二台
ssh-copy-id user@10.0.0.2

# 測試
ssh user@10.0.0.2 "hostname"
# 應該輸出 dgx-spark-2，不需要輸入密碼
```

**在第二台 DGX Spark 上**（反向也做一遍）：

```bash
ssh-keygen -t ed25519 -C "dgx-spark-2"
ssh-copy-id user@10.0.0.1
ssh user@10.0.0.1 "hostname"
```

---

## 25-3 NCCL 分散式通訊

### 25-3-1 什麼是 NCCL？

::: info 🤔 NCCL 是什麼？
NCCL（NVIDIA Collective Communications Library）是 NVIDIA 開發的多 GPU 通訊庫。它負責：

- 在多個 GPU 之間高效地傳輸資料
- 支援 All-Reduce、All-Gather、Broadcast 等操作
- 自動選擇最快的通訊路徑

簡單說：NCCL 是多機 GPU 通訊的「高速公路」。
:::

### 25-3-2 編譯安裝 NCCL Tests

```bash
# 在兩台機器上都執行
sudo apt install -y build-essential devscripts debhelper cmake

# 複製 NCCL
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j $(nproc)
sudo make install

# 複製 NCCL Tests
cd ..
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 -j $(nproc)
```

### 25-3-3 NCCL 效能測試

**雙機 All-Reduce 測試**：

```bash
# 在第一台上
mpirun -np 2 \
  -host 10.0.0.1,10.0.0.2 \
  -x NCCL_SOCKET_IFNAME=eth1 \
  ./build/all_reduce_perf \
  -b 8 -e 128M -f 2 \
  -g 1
```

**解讀結果**：

```
# Out-of-place                       In-place
# size         count    type   redop    root     time   algbw   busbw
8              2       float   sum      -1    0.00    0.00    0.00
...
134217728    33554432  float   sum      -1    2.15   62.34  117.82
```

重點看 `busbw`（匯流排頻寬），在 200GbE 直連下應該達到 150-180 Gbps。

### 25-3-4 效能基準數據

| 連接方式 | 頻寬 | 延遲 | 適合場景 |
|---------|------|------|---------|
| **200GbE DAC 直連** | **150-180 Gbps** | **最低** | 雙機分散式推論/訓練 |
| 200GbE 交換器 | 120-150 Gbps | 低 | 三機以上叢集 |
| 10GbE 網路 | 8-9 Gbps | 中 | 資料同步 |
| 1GbE 網路 | 0.8-0.9 Gbps | 高 | 管理網路 |

---

## 25-4 三機環狀互連

### 25-4-1 環形拓撲

```
        DGX Spark #1
       /             \
  DAC /               \ DAC
     /                 \
DGX Spark #2 ─── DAC ─── DGX Spark #3
```

每台機器用兩條 DAC 線連接相鄰兩台。

### 25-4-2 網路設定

每台機器需要設定兩個網路介面：

**DGX Spark #1**：
```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses: [10.0.1.1/24]
      dhcp4: no
    eth2:
      addresses: [10.0.2.1/24]
      dhcp4: no
```

**DGX Spark #2**：
```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses: [10.0.1.2/24]
      dhcp4: no
    eth2:
      addresses: [10.0.3.2/24]
      dhcp4: no
```

**DGX Spark #3**：
```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses: [10.0.2.3/24]
      dhcp4: no
    eth2:
      addresses: [10.0.3.3/24]
      dhcp4: no
```

### 25-4-3 環形 All-Reduce

環形拓撲的 All-Reduce 操作：

```
資料從 #1 → #2 → #3 → #1
```

每台機器只跟相鄰的機器通訊，適合環形拓撲。

```bash
mpirun -np 3 \
  -host 10.0.1.1,10.0.1.2,10.0.2.3 \
  -x NCCL_SOCKET_IFNAME=eth1,eth2 \
  -x NCCL_ALGO=Ring \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

---

## 25-5 多機交換器互連

### 25-5-1 交換器設定

使用 200GbE 交換器（如 NVIDIA Quantum-2），所有 DGX Spark 連到同一台交換器：

```
DGX Spark #1 ─┐
DGX Spark #2 ─┤── 200GbE 交換器
DGX Spark #3 ─┤
DGX Spark #4 ─┘
```

### 25-5-2 網路設定

所有機器在同一個子網路中：

```yaml
network:
  version: 2
  ethernets:
    eth1:
      addresses:
        - 10.0.0.1/24  # 每台機器用不同的 IP
      dhcp4: no
      mtu: 9000
```

### 25-5-3 測試

```bash
# 測試所有機器之間的連線
for i in 1 2 3 4; do
  ping -c 3 10.0.0.$i
done
```

---

## 25-6 分散式推論實戰：雙機跑 235B 模型

### 25-6-1 為什麼需要雙機跑 235B？

| 模型 | 參數 | NVFP4 大小 | 單機 128GB |
|------|------|-----------|-----------|
| Qwen3.5-122B | 122B | ~61 GB | ✅ 可以 |
| **Qwen3.5-235B** | **235B** | **~118 GB** | ⚠️ 勉強 |
| Qwen3.5-405B | 405B | ~203 GB | ❌ 不行 |

235B 模型用 NVFP4 量化後約 118GB，單機雖然裝得下，但幾乎沒有空間給 KV cache 和其他程式。用雙機分散式推論，每台只負責一半，輕鬆又穩定。

### 25-6-2 建立 Ray 叢集

Ray 是一個分散式計算框架，讓多台機器像一台一樣工作。

**在主節點（DGX Spark #1）上**：

```bash
# 安裝 Ray
uv pip install ray

# 啟動 Ray 主節點
ray start --head \
  --port=6379 \
  --node-ip-address=10.0.0.1 \
  --resources='{"gpu": 1}'
```

**在工作節點（DGX Spark #2）上**：

```bash
# 安裝 Ray
uv pip install ray

# 加入叢集
ray start \
  --address=10.0.0.1:6379 \
  --node-ip-address=10.0.0.2 \
  --resources='{"gpu": 1}'
```

**確認叢集狀態**：

```bash
ray status
```

應該看到：
```
Node status
---------------------------------------------------------------
Active:
  10.0.0.1: 1 GPU
  10.0.0.2: 1 GPU
```

### 25-6-3 啟動 vLLM 分散式推論

在主節點上執行：

```python
from vllm import LLM, SamplingParams

# 初始化分散式 LLM
llm = LLM(
    model="Qwen/Qwen3.5-235B-A22B-NVFP4",
    tensor_parallel_size=2,  # 兩台機器
    distributed_executor_backend="ray",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)

# 測試推論
prompts = [
    "請介紹 DGX Spark 多機叢集的優勢",
    "解釋什麼是分散式計算",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"提示詞: {output.prompt}")
    print(f"生成: {output.outputs[0].text}")
    print("---")
```

### 25-6-4 效能比較

| 配置 | 模型 | 記憶體/機 | 輸出速度 | 首次回應 |
|------|------|----------|---------|---------|
| 單機 | Qwen3.5-122B | 61 GB | ~12 t/s | ~5 秒 |
| **雙機** | **Qwen3.5-235B** | **59 GB** | **~8 t/s** | **~8 秒** |
| 四機 | Qwen3.5-405B | ~51 GB | ~5 t/s | ~12 秒 |

::: tip 💡 分散式推論的速度
分散式推論的速度會受到機器之間通訊頻寬的限制。在 200GbE 直連下，雙機的效率約為單機的 60-70%。

雖然速度沒有線性增長，但**能跑起來**才是最重要的。
:::

### 25-6-5 用 Open WebUI 存取分散式模型

分散式 vLLM 依然提供 OpenAI 相容 API，所以 Open WebUI 可以直接使用：

1. 在 Open WebUI 中添加新的 OpenAI 端點
2. URL：`http://10.0.0.1:8000/v1`
3. 模型名稱：`Qwen/Qwen3.5-235B-A22B-NVFP4`
4. 開始對話

---

## 25-7 分散式訓練實戰：雙機微調

### 25-7-1 用 DeepSpeed 分散式訓練

DeepSpeed 是 Microsoft 開發的分散式訓練框架：

```bash
# 在兩台機器上都安裝
uv pip install deepspeed
```

**訓練設定檔**：

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10
}
```

**啟動分散式訓練**：

```bash
deepspeed --hostfile=hostfile.txt train.py --deepspeed_config ds_config.json
```

`hostfile.txt`：
```
10.0.0.1 slots=1
10.0.0.2 slots=1
```

---

## 25-8 三種拓撲比較

| 拓撲 | 示意圖 | 頻寬 | 延遲 | 成本 | 適合場景 |
|------|--------|------|------|------|---------|
| **雙機直連** | A ←→ B | 最高 | 最低 | 最低（1 條線） | 兩機分散式推論 |
| **三機環狀** | A ←→ B ←→ C ←→ A | 中等 | 中等 | 中等（3 條線） | 三機訓練 |
| **交換器** | A,B,C,D ←→ Switch | 高 | 低 | 最高（交換器 + 線） | 多機叢集 |

### 選擇建議

| 需求 | 推薦拓撲 |
|------|---------|
| 兩台機器，跑 235B 模型 | 雙機直連 |
| 三台機器，分散式訓練 | 三機環狀 |
| 四台以上，企業叢集 | 交換器 |

---

## 25-9 回復設定

### 25-9-1 移除多機網路設定

```bash
# 刪除直連網路設定
sudo rm /etc/netplan/02-direct-connect.yaml

# 套用變更
sudo netplan apply

# 確認恢復為 DHCP
ip addr show eth1
```

### 25-9-2 停止 Ray 叢集

```bash
# 在所有節點上
ray stop
```

### 25-9-3 清理 NCCL

```bash
# 移除 NCCL
cd nccl
sudo make uninstall
cd ..
rm -rf nccl nccl-tests
```

---

## 25-10 常見問題與疑難排解

### 25-10-1 NCCL 測試失敗

**問題**：`NCCL connection timeout`

**解決方案**：
```bash
# 確認防火牆允許 NCCL 使用的 port
sudo ufw allow from 10.0.0.0/24

# 確認網路介面名稱正確
NCCL_SOCKET_IFNAME=eth1

# 測試網路連線
iperf3 -c 10.0.0.2
```

### 25-10-2 Ray 工作節點無法加入

**問題**：`Connection refused`

**解決方案**：
```bash
# 確認主節點的 Ray 正在執行
ray status

# 確認 port 6379 沒有被防火牆阻擋
sudo ufw allow 6379

# 確認網路連線
ping 10.0.0.1
telnet 10.0.0.1 6379
```

### 25-10-3 分散式推論速度很慢

**問題**：雙機推論速度比預期慢很多。

**解決方案**：
1. 確認使用 200GbE 而不是 1GbE
2. 確認 MTU 設為 9000（Jumbo Frame）
3. 確認 NCCL 使用正確的網路介面
4. 減少 `tensor_parallel_size` 的通訊量

---

## 25-11 本章小結

::: success ✅ 你現在知道了
- 多台 DGX Spark 可以互連，跑更大的模型
- 200GbE DAC 直連是最簡單的雙機互連方式
- NCCL 是多 GPU 通訊的標準庫
- Ray + vLLM 可以實現分散式推論
- 雙機能跑 235B 模型，四機能跑 405B 模型
- 分散式訓練用 DeepSpeed 最方便
:::

::: tip 🎉 恭喜！
你已經完成了整本書的所有章節！從開箱到多機叢集，你現在是 DGX Spark 的專家了！

👉 [回到首頁](/) | [查看附錄](/guide/appendix-a/)
:::

::: info 📝 上一章
← [回到第 23 章：CUDA-X 資料科學、JAX 與特殊領域應用](/guide/chapter23/)
:::
