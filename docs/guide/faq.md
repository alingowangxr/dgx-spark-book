# 常見問題與故障排除（FAQ）

本 FAQ 涵蓋 DGX Spark 使用過程中最常遇到的問題，包含系統建置、GPU 驅動、Docker、LLM 推論、微調、多機互連等主題。如果以下 FAQ 沒有解決你的問題，建議：

1. 回到相關章節重新閱讀
2. 用 Claude Code 協助除錯（告訴它錯誤訊息）
3. 查看 NVIDIA 官方論壇或 GitHub Issues

---

## 系統建置與開機

### Q1：DGX Spark 開機後沒有畫面輸出？

1. 確認 HDMI 線材已正確連接至 HDMI 2.1 埠
2. 確認顯示器已開啟且輸入源設定正確
3. 嘗試使用 USB-C 轉 HDMI 轉接器
4. 檢查電源指示燈是否正常亮起
5. 如果仍無畫面，嘗試連接至其他顯示器或電視
6. 透過 SSH 遠端登入確認系統是否正常啟動

### Q2：如何取得 DGX Spark 的 IP 位址？

```bash
# 方法一：在 DGX Spark 本機執行
ip addr show | grep inet

# 方法二：查看路由器 DHCP 客戶端列表
# 登入路由器管理介面，尋找 "DGX-Spark" 或類似名稱

# 方法三：使用 nmap 掃描
nmap -sn 192.168.1.0/24 | grep -B1 "NVIDIA"
```

### Q3：DGX OS 安裝失敗怎麼辦？

1. 確認 USB 安裝碟製作正確（建議使用 Rufus 或 balenaEtcher）
2. 檢查 USB 隨身碟是否有損壞
3. 重新下載 DGX OS 映像檔並驗證 SHA256
4. 在 BIOS/UEFI 中確認啟動順序正確
5. 確認硬碟空間足夠（建議至少 256GB SSD）

### Q4：如何更新 DGX OS 系統？

```bash
# 更新套件清單
sudo apt update

# 升級所有套件
sudo apt upgrade -y

# 更新 NVIDIA 驅動與 CUDA
sudo apt install -y nvidia-driver-570 cuda-toolkit-13-0

# 清理舊套件
sudo apt autoremove -y

# 重開機套用更新
sudo reboot
```

---

## GPU 與驅動相關

### Q5：nvidia-smi 顯示找不到 GPU？

```bash
# 步驟 1：檢查驅動是否載入
lsmod | grep nvidia

# 步驟 2：如果沒有輸出，重新載入驅動
sudo modprobe nvidia

# 步驟 3：檢查 dmesg 是否有錯誤
dmesg | grep -i nvidia

# 步驟 4：重新啟動 NVIDIA 服務
sudo systemctl restart nvidia-persistenced

# 步驟 5：重試
nvidia-smi
```

如果以上步驟都無效，嘗試重新開機。如果重開機後仍找不到，檢查硬體連接。

### Q6：GPU 驅動版本與 CUDA 版本不匹配？

```bash
# 查看目前驅動版本
nvidia-smi

# 查看 CUDA 版本
nvcc --version

# 安裝匹配的驅動與 CUDA
sudo apt install -y nvidia-driver-570 cuda-13-0

# 更新環境變數
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Q7：GPU 溫度過高怎麼辦？

正常運作溫度是 30-70°C。如果超過 80°C：

```bash
# 監控 GPU 溫度
watch -n 1 nvidia-smi

# 查看風扇轉速
nvidia-smi --query-fan=speed --format=csv
```

**降溫措施：**
1. 確認 DGX Spark 周圍有至少 10cm 的通風空間
2. 清理散熱孔的灰塵（使用壓縮空氣）
3. 降低同時運行的模型數量
4. 避免在高溫環境（>30°C）下長時間滿載運行
5. 考慮使用外部散熱墊或小型風扇輔助散熱
6. 限制 GPU 功耗：`sudo nvidia-smi -pl 100`（限制為 100W）

### Q8：如何限制 GPU 功耗？

```bash
# 查看目前功耗限制
nvidia-smi -q -d POWER

# 設定功耗限制（單位：瓦特）
sudo nvidia-smi -pl 100

# 設定為最低功耗模式
sudo nvidia-smi -pm 1

# 恢復預設
sudo nvidia-smi -pl 150
```

### Q9：GPU 記憶體使用量異常高？

```bash
# 查看 GPU 記憶體使用狀況
nvidia-smi

# 查看哪些行程占用 GPU
fuser -v /dev/nvidia*

# 找出占用 GPU 記憶體的行程
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# 終止不需要的行程
kill -9 <PID>

# 清除 GPU 記憶體快取（Python）
import torch
torch.cuda.empty_cache()
```

---

## Docker 相關

### Q10：Docker 指令需要 sudo 才能執行？

```bash
# 把使用者加入 docker 群組
sudo usermod -aG docker $USER

# 重新登入使設定生效（或執行）
newgrp docker

# 驗證（不應該需要 sudo）
docker ps
```

### Q11：Docker 容器佔用太多磁碟空間？

```bash
# 查看磁碟使用狀況
docker system df

# 查看詳細使用量
docker system df -v

# 清理未使用的映像檔、容器、網路和快取
docker system prune -a --volumes

# 只清理停止的容器
docker container prune

# 只清理懸掛的映像檔
docker image prune

# 只清理未使用的 volumes
docker volume prune

# 限制 Docker 日誌大小
sudo nano /etc/docker/daemon.json
# 加入：
# {
#   "log-driver": "json-file",
#   "log-opts": {
#     "max-size": "10m",
#     "max-file": "3"
#   }
# }
sudo systemctl restart docker
```

### Q12：Docker 容器無法使用 GPU？

```bash
# 確認 NVIDIA Container Toolkit 已安裝
nvidia-container-cli info

# 如果未安裝
sudo apt install -y nvidia-container-toolkit

# 設定 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 測試 GPU 支援
docker run --rm --gpus all nvidia/cuda:13.0-base nvidia-smi
```

### Q13：Docker 映像檔下載很慢？

```bash
# 設定 Docker 鏡像站（以中國為例）
sudo nano /etc/docker/daemon.json
# 加入：
# {
#   "registry-mirrors": [
#     "https://mirror.ccs.tencentyun.com",
#     "https://registry.docker-cn.com"
#   ]
# }
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Q14：如何備份 Docker 容器資料？

```bash
# 備份 Docker volume
docker run --rm -v <volume_name>:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /data .

# 還原 Docker volume
docker run --rm -v <volume_name>:/data -v $(pwd):/backup alpine tar xzf /backup/backup.tar.gz -C /data

# 匯出容器為映像檔
docker commit <container_name> backup_image
docker save backup_image > backup_image.tar

# 匯入映像檔
docker load < backup_image.tar
```

---

## Ollama 相關

### Q15：Ollama 服務無法啟動？

```bash
# 檢查服務狀態
systemctl status ollama

# 查看日誌
journalctl -u ollama -f

# 重新啟動
sudo systemctl restart ollama

# 如果服務不存在，手動安裝
curl -fsSL https://ollama.com/install.sh | sh

# 手動啟動（除錯用）
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Q16：模型下載很慢或中斷？

```bash
# 方法一：設定代理
sudo systemctl edit ollama
# 加入：
# [Service]
# Environment="HTTPS_PROXY=http://你的代理伺服器:port"
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 方法二：手動下載模型（如果有 GGUF 檔案）
# 將 GGUF 檔案放到 ~/.ollama/models/blobs/ 目錄
# 然後建立 Manifest 檔案

# 方法三：使用鏡像站
export OLLAMA_ORIGINS="https://your-mirror-server.com"
```

### Q17：Ollama 模型載入很慢？

1. 首次載入需要將模型從磁碟讀入記憶體，屬正常現象
2. 使用 SSD 可以加快載入速度
3. 模型載入後會保留在記憶體中，後續呼叫會很快
4. 可以調整模型保留時間：
```bash
# 設定模型保留時間（設為 -1 表示永久保留）
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "keep_alive": -1
}'
```

### Q18：如何管理 Ollama 模型？

```bash
# 列出已下載的模型
ollama list

# 下載模型
ollama pull llama3.1:8b

# 刪除模型
ollama rm llama3.1:8b

# 複製模型
ollama cp llama3.1 my-model

# 查看模型資訊
ollama show llama3.1

# 建立自訂 Modelfile
cat > Modelfile << EOF
FROM llama3.1
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM 你是一個專業的 AI 助手。
EOF

# 建立自訂模型
ollama create my-assistant -f Modelfile
```

---

## 網路與遠端存取

### Q19：SSH 無法連線？

```bash
# 在 DGX Spark 上檢查：
# 1. SSH 服務是否啟動
sudo systemctl status ssh

# 2. 如果未安裝
sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh

# 3. 檢查防火牆
sudo ufw status
sudo ufw allow 22/tcp

# 4. 檢查 SSH 設定
sudo nano /etc/ssh/sshd_config
# 確認：
# Port 22
# PermitRootLogin no
# PasswordAuthentication yes

# 5. 重啟 SSH 服務
sudo systemctl restart ssh
```

**在客戶端檢查：**
```bash
# 測試連線
ssh -v user@dgx-spark-ip

# 使用金鑰登入（推薦）
ssh-keygen -t ed25519
ssh-copy-id user@dgx-spark-ip
```

### Q20：Tailscale 連線中斷？

```bash
# 檢查 Tailscale 狀態
tailscale status

# 查看詳細狀態
tailscale ip

# 重新連線
sudo tailscale down
sudo tailscale up

# 重新認證
sudo tailscale up --reset

# 查看日誌
sudo journalctl -u tailscaled -f

# 設定開機自動啟動
sudo systemctl enable tailscaled
sudo systemctl start tailscaled

# 設定 Exit Node（讓其他裝置透過 DGX Spark 上網）
sudo tailscale up --advertise-exit-node
```

### Q21：如何設定靜態 IP？

```bash
# 使用 netplan 設定（Ubuntu 24.04）
sudo nano /etc/netplan/01-network-manager-all.yaml

# 加入以下內容（根據實際網路介面調整）：
# network:
#   version: 2
#   ethernets:
#     eth0:
#       dhcp4: no
#       addresses:
#         - 192.168.1.100/24
#       routes:
#         - to: default
#           via: 192.168.1.1
#       nameservers:
#         addresses: [8.8.8.8, 8.8.4.4]

# 套用設定
sudo netplan apply

# 驗證
ip addr show eth0
```

---

## 模型推論相關

### Q22：模型推論速度很慢？

**效能檢查清單：**

| 檢查項目 | 指令/方法 | 正常值 |
|---------|----------|--------|
| GPU 是否被使用 | `nvidia-smi` | GPU 使用率 >50% |
| 記憶體是否足夠 | `free -h` | 可用記憶體 > 模型大小 |
| 量化格式是否正確 | 檢查模型檔案 | INT4/INT8 比 FP16 快 |
| 批次大小設定 | 檢查框架設定 | 單人推論 batch=1 |
| 上下文長度 | 檢查 num_ctx | 過長會影響速度 |

**最佳化建議：**
```bash
# 使用量化模型（INT4）
ollama pull llama3.1:8b-instruct-q4_0

# 在 vLLM 中啟用 PagedAttention
vllm serve llama3.1 --max-model-len 4096

# 使用 TensorRT-LLM 最佳化
trtllm-build --checkpoint_dir ./model --output_dir ./engine
```

### Q23：模型輸出亂碼或產生無意義內容？

1. 確認模型檔案完整（重新下載）
2. 檢查溫度設定（temperature 建議 0.1-0.7）
3. 確認提示詞（prompt）格式正確
4. 嘗試不同的量化格式（Q4_K_M 通常品質最佳）
5. 檢查上下文長度是否足夠
6. 確認模型與任務匹配（對話模型 vs 程式碼模型）

### Q24：如何同時運行多個模型？

```bash
# 方法一：使用 vLLM 多模型服務
vllm serve model1 --port 8001
vllm serve model2 --port 8002

# 方法二：使用 Ollama（自動管理）
# Ollama 會自動將不活躍模型卸載，需要時重新載入

# 方法三：使用 Docker 容器隔離
docker run -d --gpus all -p 8001:8000 vllm/vllm-openai --model model1
docker run -d --gpus all -p 8002:8000 vllm/vllm-openai --model model2

# 監控記憶體使用
watch -n 1 nvidia-smi
```

---

## 微調相關

### Q25：訓練時出現 CUDA Out of Memory？

**解決方案（依推薦順序）：**

| 方法 | 效果 | 品質影響 |
|------|------|---------|
| 降低 batch size | 大幅減少記憶體 | 無 |
| 使用梯度累積 | 模擬大 batch | 無 |
| 使用 QLoRA（4-bit） | 減少 75% 記憶體 | 極小 |
| 減少 max_seq_length | 線性減少 | 可能影響長文 |
| 啟用梯度檢查點 | 減少 60% 記憶體 | 速度變慢 20% |
| 使用 DeepSpeed ZeRO | 分散記憶體 | 設定複雜 |

```python
# Unsloth QLoRA 設定範例
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B",
    max_seq_length=2048,
    load_in_4bit=True,  # 使用 4-bit 量化
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing=True,  # 啟用梯度檢查點
)
```

### Q26：訓練 Loss 不下降？

**診斷步驟：**

| 可能原因 | 檢查方法 | 解決方案 |
|---------|---------|---------|
| 學習率太高 | 檢查學習率曲線 | 降低學習率（1e-4 → 1e-5） |
| 資料格式錯誤 | 檢查訓練資料 | 確認 JSON/對話格式正確 |
| 模型未正確載入 | 檢查模型權重 | 重新下載或檢查路徑 |
| 資料量不足 | 檢查資料集大小 | 增加訓練資料 |
| 過擬合 | 檢查驗證 Loss | 增加 dropout 或 early stopping |
| 梯度消失 | 檢查梯度範數 | 使用梯度裁剪 |

```python
# 梯度裁剪設定
training_args = TrainingArguments(
    learning_rate=1e-5,
    max_grad_norm=1.0,  # 梯度裁剪
    warmup_ratio=0.05,  # 學習率預熱
    fp16=True,
)
```

### Q27：微調後的模型品質不好？

1. 檢查訓練資料品質（是否有噪音、錯誤標籤）
2. 增加訓練步數（epochs）
3. 調整 LoRA 參數（rank、alpha）
4. 使用更高品質的基礎模型
5. 加入驗證集監控過擬合
6. 嘗試不同的量化格式
7. 使用 DPO/ORPO 等對齊方法

---

## ComfyUI 與影像生成

### Q28：ComfyUI 啟動失敗？

```bash
# 檢查 Python 版本（需要 3.10+）
python3 --version

# 安裝依賴
pip install torch torchvision torchaudio
pip install -r requirements.txt

# 檢查 GPU 是否可用
python3 -c "import torch; print(torch.cuda.is_available())"

# 啟動 ComfyUI
python3 main.py --listen 0.0.0.0

# 如果記憶體不足
python3 main.py --lowvram
```

### Q29：影像生成速度很慢？

1. 確認使用 GPU 而非 CPU（檢查 `torch.cuda.is_available()`）
2. 降低生成解析度
3. 減少取樣步數（20-30 步通常足夠）
4. 使用更高效的模型（SDXL Turbo / LCM）
5. 啟用 xFormers 或 SDPA 加速
6. 使用 TensorRT 最佳化

```bash
# 啟用 xFormers
pip install xformers

# 使用 TensorRT 加速
pip install tensorrt
```

---

## 多機互連

### Q30：NCCL 測試失敗？

**診斷步驟：**

```bash
# 步驟 1：確認網路連線
ping <另一台 DGX Spark IP>

# 步驟 2：檢查 NCCL 版本
python3 -c "import torch; print(torch.cuda.nccl.version())"

# 步驟 3：檢查防火牆
sudo ufw status
sudo ufw allow 1024:65535/tcp  # NCCL 使用動態 port

# 步驟 4：設定 NCCL 網路介面
export NCCL_SOCKET_IFNAME=eth0

# 步驟 5：啟用 NCCL 除錯日誌
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 步驟 6：執行 NCCL 測試
mpirun -np 2 -H node1,node1 --allow-run-as-root nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

**常見錯誤與解決方案：**

| 錯誤訊息 | 原因 | 解決方案 |
|---------|------|---------|
| `NCCL: Connection refused` | 防火牆阻擋 | 開放 NCCL 使用的 port |
| `NCCL: Network error` | 網路不通 | 檢查網路連線與路由 |
| `NCCL: Invalid usage` | 版本不一致 | 統一 NCCL 版本 |
| `NCCL: System error` | 權限問題 | 使用 `--allow-run-as-root` |
| `NCCL: Timeout` | 延遲過高 | 增加 `NCCL_TIMEOUT` |

### Q31：如何設定多機 SSH 免密登入？

```bash
# 在每台機器上產生金鑰
ssh-keygen -t ed25519

# 將公鑰複製到其他機器
ssh-copy-id user@node1
ssh-copy-id user@node2

# 測試免密登入
ssh user@node1 "hostname"

# 設定 SSH config（簡化連線）
nano ~/.ssh/config
# 加入：
# Host node1
#   HostName 192.168.1.101
#   User user
#   IdentityFile ~/.ssh/id_ed25519
# Host node2
#   HostName 192.168.1.102
#   User user
#   IdentityFile ~/.ssh/id_ed25519
```

---

## 效能監控與調校

### Q32：如何監控系統效能？

```bash
# GPU 監控
watch -n 1 nvidia-smi

# 詳細 GPU 資訊
nvidia-smi -q

# CPU 與記憶體監控
htop

# 磁碟 I/O 監控
iostat -x 1

# 網路監控
iftop

# 綜合監控（推薦）
sudo apt install glances
glances

# 自訂監控指令碼
watch -n 1 'echo "=== GPU ===" && nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader && echo "=== Memory ===" && free -h'
```

### Q33：如何設定開機自動啟動服務？

```bash
# 設定 Ollama 開機啟動
sudo systemctl enable ollama

# 設定 Tailscale 開機啟動
sudo systemctl enable tailscaled

# 設定 Docker 開機啟動
sudo systemctl enable docker

# 建立自訂服務
sudo nano /etc/systemd/system/my-ai-service.service
# 加入：
# [Unit]
# Description=My AI Service
# After=network.target
#
# [Service]
# Type=simple
# User=user
# WorkingDirectory=/home/user/my-service
# ExecStart=/usr/bin/python3 main.py
# Restart=always
#
# [Install]
# WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable my-ai-service
sudo systemctl start my-ai-service
```

---

## 資料備份與還原

### Q34：如何備份 DGX Spark 的資料？

```bash
# 方法一：備份重要目錄
tar czf backup-$(date +%Y%m%d).tar.gz ~/projects ~/models ~/.ollama

# 方法二：使用 rsync 同步到外部硬碟
rsync -avz --progress ~/projects/ /mnt/external-drive/backup/

# 方法三：備份到遠端伺服器
rsync -avz --progress ~/projects/ user@remote-server:/backup/

# 方法四：使用 Docker volume 備份
docker run --rm -v ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/ollama-backup.tar.gz -C /data .

# 方法五：建立系統快照（如果有 Btrfs/ZFS）
sudo btrfs subvolume snapshot / /snapshots/backup-$(date +%Y%m%d)
```

---

## 其他常見問題

### Q35：如何安裝 Python 套件時遇到編譯錯誤？

```bash
# 安裝編譯工具
sudo apt install -y build-essential python3-dev

# 安裝特定版本的套件（避免相容性問題）
pip install package==1.2.3

# 使用預編譯的二進位檔
pip install --only-binary :all: package

# 如果 CUDA 相關套件編譯失敗
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### Q36：Jupyter Notebook 無法連線？

```bash
# 啟動 Jupyter 並允許外部連線
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 如果 port 被占用
lsof -i :8888
kill -9 <PID>

# 設定密碼
jupyter lab password

# 建立 systemd 服務
sudo nano /etc/systemd/system/jupyter.service
# 加入：
# [Unit]
# Description=Jupyter Lab
# After=network.target
#
# [Service]
# Type=simple
# User=user
# ExecStart=/home/user/.local/bin/jupyter lab --ip=0.0.0.0 --port=8888
# Restart=always
#
# [Install]
# WantedBy=multi-user.target
```

### Q37：如何重置 DGX Spark 到出廠設定？

```bash
# 警告：此操作會刪除所有資料！

# 方法一：重新安裝 DGX OS
# 1. 製作 USB 安裝碟
# 2. 從 USB 開機
# 3. 選擇重新安裝

# 方法二：手動清理
# 備份重要資料
rsync -avz ~/important-data/ /backup/

# 清理使用者資料
rm -rf ~/.ollama ~/.cache/pip ~/.local/share/Trash/*

# 清理 Docker
docker system prune -a --volumes

# 重設網路設定
sudo netplan apply
```

### Q38：DGX Spark 適合跑哪些模型？

| 模型類型 | 推薦模型 | 記憶體需求 | 效能 |
|---------|---------|-----------|------|
| 對話 LLM | Llama 3.1 8B | 4-16 GB | ⭐⭐⭐⭐⭐ |
| 對話 LLM | Qwen 2.5 7B | 4-16 GB | ⭐⭐⭐⭐⭐ |
| 對話 LLM | Llama 3.1 70B (INT4) | 35-40 GB | ⭐⭐⭐⭐ |
| 程式碼 LLM | DeepSeek-Coder 6.7B | 4-14 GB | ⭐⭐⭐⭐⭐ |
| 視覺語言模型 | LLaVA 7B | 14-16 GB | ⭐⭐⭐⭐ |
| 影像生成 | FLUX.1-schnell | 12-24 GB | ⭐⭐⭐⭐ |
| 影像生成 | SDXL | 6-8 GB | ⭐⭐⭐⭐⭐ |
| 語音辨識 | Whisper Large v3 | 3-10 GB | ⭐⭐⭐⭐⭐ |
| 語音合成 | CosyVoice | 2-4 GB | ⭐⭐⭐⭐⭐ |
| 嵌入模型 | nomic-embed-text | 0.3 GB | ⭐⭐⭐⭐⭐ |

---

## 快速診斷流程

如果遇到問題，按照以下流程快速診斷：

```
問題發生
  │
  ├─ 系統問題？ → 檢查 dmesg、journalctl、系統日誌
  │
  ├─ GPU 問題？ → nvidia-smi → lsmod → dmesg | grep nvidia
  │
  ├─ 網路問題？ → ping → traceroute → 防火牆檢查
  │
  ├─ Docker 問題？ → docker logs → docker system df
  │
  ├─ 模型問題？ → 檢查記憶體 → 檢查量化格式 → 重新下載
  │
  └─ 效能問題？ → nvidia-smi → htop → iostat → 網路監控
```

---

::: tip 🚀 回到導覽
- [首頁](/)
- [第 1 章：DGX Spark 硬體總覽](/guide/chapter1/)
- [推薦模型清單](/guide/models)
- [附錄 A：縮寫與術語表](/guide/appendix-a/)
- [附錄 B：Playbook 對照表](/guide/appendix-b/)
- [附錄 E：硬體規格速查表](/guide/appendix-e/)
:::
