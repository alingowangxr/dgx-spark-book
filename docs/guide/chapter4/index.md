# 第 4 章：遠端桌面與網路存取

::: tip 🎯 本章你將學到什麼
- 安裝 XFCE 桌面環境和 VNC 遠端桌面
- 設定固定 IP 和 mDNS（用名字代替 IP）
- 用 Tailscale VPN 從世界任何地方連回家
- 五種 GPU 監控工具的比較與使用
- 用 Timeshift 備份整個環境
:::

::: warning ⏱️ 預計閱讀時間
約 25 分鐘。實際設定約需 30-40 分鐘。
:::

---

## 4-1 桌面環境安裝

### 4-1-1 安裝 LightDM 顯示管理器

::: info 🤔 什麼是顯示管理器？
顯示管理器（Display Manager）是 Linux 的「登入畫面」。DGX OS 預設可能沒有圖形化登入介面，我們需要安裝一個。

LightDM 是輕量級的選擇，適合 DGX Spark。
:::

```bash
# 安裝 LightDM
sudo apt install -y lightdm

# 安裝過程中會問你要用哪個顯示管理器
# 選擇 lightdm
```

### 4-1-2 安裝 XFCE 桌面環境

```bash
# 安裝 XFCE 桌面（輕量、快速）
sudo apt install -y xfce4 xfce4-goodies

# 如果問你顯示管理器，選 lightdm
```

::: info 🤔 為什麼選 XFCE 而不是 GNOME？
| 桌面環境 | 記憶體用量 | 適合場景 |
|----------|-----------|---------|
| **XFCE** | ~200 MB | 遠端桌面、伺服器 |
| GNOME | ~800 MB | 本機使用、美觀 |
| KDE | ~500 MB | 本機使用、功能豐富 |

DGX Spark 的 128GB 記憶體主要要給 AI 模型用，所以桌面環境越輕量越好。
:::

### 4-1-3 設定自動登入

為了方便 VNC 連線，我們設定開機後自動登入：

```bash
# 編輯 LightDM 設定檔
sudo nano /etc/lightdm/lightdm.conf
```

加入或修改以下內容：

```ini
[Seat:*]
autologin-user=你的使用者名稱
autologin-user-timeout=0
user-session=xfce
```

重新啟動 LightDM 讓設定生效：

```bash
sudo systemctl restart lightdm
```

---

## 4-2 VNC 遠端桌面

### 4-2-1 安裝 x11vnc

x11vnc 可以讓你遠端看到 DGX Spark 的實際桌面（不是新建一個虛擬桌面）。

```bash
# 安裝 x11vnc
sudo apt install -y x11vnc

# 設定 VNC 密碼
x11vnc -storepasswd
# 會問你輸入密碼，這個密碼等一下遠端連線時要用
```

### 4-2-2 設定 x11vnc 為系統服務

為了讓 x11vnc 開機自動啟動，我們建立一個 systemd 服務：

```bash
sudo nano /etc/systemd/system/x11vnc.service
```

貼入以下內容：

```ini
[Unit]
Description=Start x11vnc at startup
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -auth guess -forever -loop -noxdamage -repeat -rfbauth /home/你的使用者名稱/.vnc/passwd -rfbport 5900 -shared

[Install]
WantedBy=multi-user.target
```

::: danger 🚨 重要
把上面設定中的 `你的使用者名稱` 換成你實際的使用者名稱！
:::

啟動服務：

```bash
# 重新載入 systemd
sudo systemctl daemon-reload

# 啟用並啟動 x11vnc
sudo systemctl enable x11vnc
sudo systemctl start x11vnc

# 確認服務狀態
sudo systemctl status x11vnc
```

### 4-2-3 從本機連線 VNC

在你的個人電腦上：

1. 下載 VNC 客戶端：
   - **macOS**：內建「螢幕共享」App（Spotlight 搜尋「螢幕共享」）
   - **Windows**：下載 [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)（免費）
   - **Linux**：`sudo apt install -y remmina`

2. 連線到 `DGX_Spark_IP:5900`
3. 輸入你剛才設定的 VNC 密碼

::: tip 💡 macOS 快捷方式
在 macOS 的 Finder 中，按 `Cmd + K`，輸入：
```
vnc://DGX_Spark_IP:5900
```
就可以直接連線。
:::

### 4-2-4 安全性考量：SSH Tunnel + VNC

::: warning ⚠️ VNC 本身不安全
VNC 的連線是**沒有加密**的。如果你的 DGX Spark 在同一个可信的區域網路中（例如家裡），問題不大。但如果要從外部網路連線，一定要用 SSH Tunnel。
:::

**SSH Tunnel 做法**：

```bash
# 在你的個人電腦上執行
ssh -L 5900:localhost:5900 你的使用者名稱@DGX_Spark_IP
```

這個指令的意思是：把你個人電腦的 `localhost:5900` 透過加密的 SSH 通道，轉發到 DGX Spark 的 `5900` port。

然後用 VNC 客戶端連線到 `localhost:5900` 即可。

::: tip 💡 一條指令搞定
有些 VNC 客戶端支援直接透過 SSH 連線。以 macOS 的「螢幕共享」為例：

1. 先開 SSH Tunnel（上面的指令）
2. 連線到 `vnc://localhost:5900`
3. 所有流量都會經過加密的 SSH 通道
:::

---

## 4-3 區域網路設定

### 4-3-1 固定 IP 設定

DHCP 每次給的 IP 可能不一樣，這對於遠端連線很不方便。我們來設定固定 IP。

**方法 1：透過 Netplan（推薦）**

```bash
# 查看網路介面名稱
ip addr show
# 找到你的有線網路介面，通常是 eth0 或 enpXsX

# 編輯 Netplan 設定
sudo nano /etc/netplan/01-netcfg.yaml
```

貼入以下內容：

```yaml
network:
  version: 2
  ethernets:
    eth0:  # 換成你的網路介面名稱
      dhcp4: no
      addresses:
        - 192.168.1.100/24  # 你想要的固定 IP
      routes:
        - to: default
          via: 192.168.1.1  # 你的路由器 IP（閘道器）
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

套用設定：

```bash
sudo netplan apply
```

**方法 2：透過路由器設定（更簡單）**

很多路由器支援「DHCP 保留」（DHCP Reservation），讓特定裝置永遠拿到同一個 IP：

1. 登入路由器管理介面（通常是 `192.168.1.1`）
2. 找到「DHCP 保留」或「靜態 DHCP」
3. 把 DGX Spark 的 MAC 位址綁定到一個固定 IP

這種方法的好處是不需要改 DGX Spark 的設定。

### 4-3-2 mDNS / Avahi 設定

mDNS 讓你不用記 IP，直接用名字連線。

```bash
# 安裝 Avahi
sudo apt install -y avahi-daemon

# 啟用並啟動
sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon
```

設定完成後，你就可以用以下名字代替 IP：

```bash
# 從同一個區域網路中的其他電腦
ssh 你的使用者名稱@dgx-spark.local
```

::: tip 💡 超實用！
設定好 mDNS 後，你再也不需要記 IP 了。不管是 SSH、VNC、還是瀏覽器，都可以用 `dgx-spark.local` 來連線。
:::

### 4-3-3 防火牆基本設定

```bash
# 安裝 UFW（簡易防火牆）
sudo apt install -y ufw

# 預設拒絕所有外部連線
sudo ufw default deny incoming

# 允許 SSH
sudo ufw allow 22/tcp

# 允許 VNC（只有需要時才開）
sudo ufw allow 5900/tcp

# 允許 Tailscale（第 4-4 節會用到）
sudo ufw allow from 100.64.0.0/10

# 啟用防火牆
sudo ufw enable

# 查看規則
sudo ufw status
```

::: warning ⚠️ 小心！
啟用防火牆前，**一定要先允許 SSH（port 22）**，否則你可能把自己鎖在外面！

如果不小心鎖住了，你需要接上顯示器和鍵盤，在本機關閉防火牆：
```bash
sudo ufw disable
```
:::

---

## 4-4 外部存取：Tailscale VPN

### 4-4-1 什麼是 Tailscale

::: info 🤔 為什麼需要 Tailscale？
如果你想從公司、咖啡廳、甚至國外連回家裡的 DGX Spark，傳統做法是：
1. 在路由器設定 Port Forwarding（把外部請求轉到 DGX Spark）
2. 設定 DDNS（因為家裡的對外 IP 會變）
3. 處理各種防火牆和安全問題

這很麻煩，而且有安全風險。

**Tailscale** 讓這件事變得超簡單：
- 不需要設定路由器
- 不需要開 Port
- 自動加密連線
- 免費方案就夠用
:::

Tailscale 建立了一個虛擬的私人網路，所有加入這個網路的裝置都可以互相連線，就像在同一個區域網路中一樣。

### 4-4-2 安裝 Tailscale

```bash
# 安裝 Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 啟動並登入
sudo tailscale up
```

執行 `sudo tailscale up` 後，終端機會顯示一個網址。用瀏覽器打開該網址，用 Google 或 GitHub 帳號登入，完成授權。

::: tip 💡 免費方案限制
Tailscale 免費方案：
- 最多 100 個裝置
- 最多 3 個使用者
- 100 GB/月 的 P2P 傳輸量

對個人使用來說完全夠用。
:::

查看你的 Tailscale IP：

```bash
tailscale ip
# 輸出範例：100.x.x.x
```

### 4-4-3 從外面連回家中的 DGX Spark

在你的個人電腦上也安裝 Tailscale：

```bash
# macOS
brew install tailscale
sudo tailscale up

# Windows
# 從 tailscale.com 下載安裝程式

# Linux
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

登入同一個帳號後，你就可以用 Tailscale IP 連線到 DGX Spark：

```bash
# SSH 連線
ssh 你的使用者名稱@100.x.x.x

# VNC 連線（透過 SSH Tunnel）
ssh -L 5900:localhost:5900 你的使用者名稱@100.x.x.x
```

### 4-4-4 Tailscale SSH：不用開 Port 的遠端存取

Tailscale 內建了 SSH 功能，甚至不需要 DGX Spark 開啟 SSH 服務：

```bash
# 用 Tailscale SSH（需要先在 Tailscale 管理頁面啟用）
ssh 你的使用者名稱@dgx-spark
```

Tailscale SSH 的好處：
- 使用 Tailscale 的身分驗證，不需要管理 SSH 金鑰
- 可以設定存取政策（誰可以連、什麼時候可以連）
- 所有連線都有紀錄

---

## 4-5 系統監控與 DGX Dashboard

### 4-5-1 DGX Dashboard Web UI 導覽

DGX Dashboard 預設網址：`http://DGX_Spark_IP:8080`

主要功能：
- **System**：CPU、GPU、記憶體、儲存空間使用狀況
- **Containers**：Docker 容器管理
- **Updates**：系統更新
- **Settings**：系統設定

### 4-5-2 nvidia-smi：命令列 GPU 監控

```bash
# 基本用法
nvidia-smi

# 持續監控（每 1 秒更新）
nvidia-smi -l 1

# 顯示更多資訊
nvidia-smi -q
```

**nvidia-smi 輸出解讀**：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.xx.xx    Driver Version: 570.xx.xx    CUDA Version: 13.0                 |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Orin                     Off        |   00000000:00:00.0 Off |                  N/A |
| N/A   45C    P0             15W /  150W |   2048MiB / 131072MiB |      5%      Default |
+-----------------------------------------+------------------------+----------------------+
```

重點看：
- **Temp**：溫度（正常範圍 30-70°C）
- **Pwr:Usage/Cap**：功耗 / 最大功耗
- **Memory-Usage**：記憶體使用量（總共 128GB = 131072MiB）
- **GPU-Util**：GPU 使用率

### 4-5-3 nvtop：互動式 GPU 監控

nvtop 像是 `htop` 的 GPU 版本，有漂亮的圖形介面。

```bash
# 安裝
sudo apt install -y nvtop

# 執行
nvtop
```

nvtop 會顯示：
- GPU 使用率的即時圖表
- 記憶體使用量
- 每個程序的 GPU 佔用
- 溫度、功耗、風扇轉速

### 4-5-4 nvitop：GPU 監控的 Python 方案

```bash
# 安裝
uv pip install nvitop

# 執行
nvitop
```

nvitop 的特色：
- 比 nvtop 更多資訊
- 支援自訂顏色主題
- 可以顯示每個程序的詳細 GPU 使用情況

### 4-5-5 dgxtop：DGX Spark 專用的系統監控

```bash
# 安裝（從 GitHub）
pip install dgxtop

# 執行
dgxtop
```

dgxtop 是專門為 DGX Spark 設計的監控工具，會顯示：
- GPU、CPU、記憶體使用狀況
- 網路流量
- 儲存空間
- 溫度

### 4-5-6 dgx-spark-status：Web 監控儀表板

```bash
# 用 Docker 執行
docker run -d \
  --name dgx-status \
  --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/community/dgx-spark-status:latest
```

然後用瀏覽器打開 `http://DGX_Spark_IP:8501`。

### 4-5-7 五套監控工具的比較與使用建議

| 工具 | 介面 | 安裝難度 | 資訊量 | 推薦場景 |
|------|------|---------|--------|---------|
| **nvidia-smi** | 命令列 | 免安裝（內建） | 基本 | 快速檢查 |
| **nvtop** | 命令列 | 簡單 | 中等 | 日常監控 |
| **nvitop** | 命令列 | 中等 | 豐富 | 詳細分析 |
| **dgxtop** | 命令列 | 中等 | 豐富 | DGX Spark 專用 |
| **dgx-spark-status** | Web | 簡單 | 豐富 | 遠端監控 |

::: tip 💡 推薦組合
- **日常使用**：`nvidia-smi`（快速檢查）+ `nvtop`（詳細監控）
- **遠端監控**：`dgx-spark-status`（Web 介面）
- **除錯時**：`nvitop`（最詳細的程序資訊）
:::

### 4-5-8 JupyterLab 整合使用

DGX OS 預裝了 JupyterLab。啟動方法：

```bash
# 啟動 JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

然後用瀏覽器打開終端機顯示的網址（包含 token）。

JupyterLab 的好處是可以在瀏覽器中直接寫 Python 程式、執行、看結果，非常適合 AI 開發。

### 4-5-9 系統更新管理介面

DGX Dashboard 的「Updates」頁面可以：
- 檢查是否有新版本
- 一鍵更新
- 查看更新歷史

建議每週檢查一次更新。

---

## 4-6 多節點叢集監控：DGX Spark Dashboard

### 4-6-1 架構概覽

如果你有多台 DGX Spark，可以用 DGX Spark Dashboard 同時監控所有節點。

```
你的瀏覽器
  │
  └─ DGX Spark Dashboard（跑在其中一台 DGX Spark 上）
        │
        ├─ DGX Spark #1（透過 SSH）
        ├─ DGX Spark #2（透過 SSH）
        └─ DGX Spark #3（透過 SSH）
```

### 4-6-2 監控指標說明

Dashboard 會顯示每台節點的：
- GPU 使用率、記憶體使用量
- CPU 使用率、溫度
- 網路流量
- 儲存空間
- Docker 容器狀態

### 4-6-3 安裝前置準備：無密碼 SSH

Dashboard 需要透過 SSH 連線到其他節點，所以需要先設定無密碼 SSH。

**在主節點上執行**：

```bash
# 產生 SSH 金鑰（如果還沒有的話）
ssh-keygen -t ed25519
# 一路按 Enter 即可

# 把公鑰複製到其他節點
ssh-copy-id 使用者名稱@DGX_Spark_2_IP
ssh-copy-id 使用者名稱@DGX_Spark_3_IP
```

### 4-6-4 安裝與設定

```bash
# 複製專案
git clone https://github.com/community/dgx-spark-dashboard.git
cd dgx-spark-dashboard

# 編輯設定檔
nano config.yaml
```

在 `config.yaml` 中加入所有節點的資訊：

```yaml
nodes:
  - name: "spark-1"
    host: "192.168.1.100"
    user: "你的使用者名稱"
  - name: "spark-2"
    host: "192.168.1.101"
    user: "你的使用者名稱"
  - name: "spark-3"
    host: "192.168.1.102"
    user: "你的使用者名稱"
```

### 4-6-5 啟動與使用

```bash
# 啟動 Dashboard
python main.py
```

然後用瀏覽器打開 `http://主節點IP:8501`。

### 4-6-6 與其他監控工具的比較

| 工具 | 監控節點數 | 適合場景 |
|------|-----------|---------|
| nvtop / nvidia-smi | 1 | 單機日常監控 |
| dgx-spark-status | 1 | 單機 Web 監控 |
| **DGX Spark Dashboard** | **多台** | **叢集監控** |

---

## 4-7 備份整個環境

### 4-7-1 安裝 Timeshift

Timeshift 是一個系統快照工具，可以讓你一鍵還原到之前的狀態。

```bash
# 安裝 Timeshift
sudo apt install -y timeshift
```

### 4-7-2 使用 Timeshift 建立快照

```bash
# 建立第一個快照
sudo timeshift --create --comments "初始設定完成"

# 查看現有快照
sudo timeshift --list
```

::: tip 💡 建議的快照時機
- ✅ 完成基本設定後
- ✅ 安裝大型軟體前
- ✅ 系統更新前
- ✅ 開始微調實驗前
:::

**自動化快照**：

```bash
# 編輯 Timeshift 設定
sudo timeshift --setup
```

建議設定：
- **快照類型**：RSYNC
- **排程**：每天
- **保留數量**：
  - 每小時：2
  - 每天：5
  - 每週：3
  - 每月：2

### 4-7-3 還原快照

```bash
# 列出所有快照
sudo timeshift --list

# 還原到指定快照
sudo timeshift --restore --snapshot '2025-01-15_12-00-00'
```

::: warning ⚠️ 注意
還原會把系統回到快照時的狀態。快照之後新增的檔案會消失（你的個人檔案通常不受影響，但最好還是另外備份重要資料）。
:::

---

## 4-8 本章小結

::: success ✅ 你現在知道了
- XFCE + VNC 讓你擁有輕量級的遠端桌面
- 固定 IP + mDNS 讓區域網路連線更方便
- Tailscale 讓你從世界任何地方安全地連回家
- 五種監控工具各有用途，推薦 nvidia-smi + nvtop 日常使用
- Timeshift 是你的安全網，出問題時可以一鍵還原
:::

::: tip 🚀 第一篇完結！
恭喜！你已經完成了「硬體與系統建置」篇。現在你的 DGX Spark 已經：
- ✅ 開機設定完成
- ✅ 工作環境建置完成
- ✅ 遠端存取設定完成
- ✅ 監控和備份設定完成

接下來我們要進入最有趣的部分 — 開始跑 AI 模型！

👉 [前往第 5 章：Ollama — 在 128 GB 上跑超大模型 →](/guide/chapter5/)
:::

::: info 📝 上一章
← [回到第 3 章：Linux 環境建置與 Claude Code 安裝](/guide/chapter3/)
:::
