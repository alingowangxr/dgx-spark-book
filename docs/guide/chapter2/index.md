# 第 2 章：DGX OS 安裝與首次開機

::: tip 🎯 本章你將學到什麼
- 正確的接線順序與首次開機流程
- DGX OS 是什麼、裡面預裝了什麼
- 如何啟用 SSH 遠端連線
- 如何設定 Headless 模式（拔掉顯示器也能用）
- 系統更新與還原的方法
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。首次開機設定約需 30 分鐘。
:::

---

## 2-1 首次開機設定

::: danger 🚨 重要提醒
請按照以下順序操作，不要跳過任何步驟。接線順序錯誤可能導致顯示器沒有畫面。
:::

### 2-1-1 接線順序

::: tip 💡 正確接線順序（照著做就對了）

1. **先接網路線** → 把 DGX Spark 連到路由器
2. **再接 HDMI 顯示器** → 接到 HDMI 2.1 孔
3. **接鍵盤、滑鼠** → 插到 USB-A 孔
4. **最後接電源** → 插上電源線，按下電源鍵
:::

為什麼順序很重要？因為 DGX Spark 在開機時會偵測顯示器。如果先開機才接顯示器，可能偵測不到。

### 2-1-2 UEFI / BIOS 設定要點

開機後，如果看到 NVIDIA Logo，**不要急著按任何鍵**。DGX Spark 的開機流程如下：

```
按下電源鍵
  → NVIDIA Logo（約 5 秒）
  → UEFI 自我檢測（約 10 秒）
  → GRUB 開機選單（約 3 秒）
  → DGX OS 載入（約 30 秒）
  → 首次設定畫面
```

::: info 🤔 需要進 BIOS 嗎？
一般情況下**不需要**進 BIOS。DGX Spark 出廠設定已經最佳化。

如果你真的需要進 BIOS（例如要改開機順序），在 UEFI 畫面按 `F2` 或 `Del`。
:::

### 2-1-3 First-Time Setup Wizard

第一次開機會看到設定精靈，跟設定新手機很像：

| 步驟 | 內容 | 建議 |
|------|------|------|
| 1. 選擇語言 | 選你熟悉的語言 | 建議選英文（後續工具多為英文介面） |
| 2. 鍵盤配置 | 選你的鍵盤類型 | 一般選 US English |
| 3. 建立使用者 | 設定使用者名稱和密碼 | **密碼一定要記好！** |
| 4. 時區設定 | 選你的時區 | Asia/Taipei |
| 5. 同意條款 | 閱讀並同意授權條款 | 按同意繼續 |

::: warning ⚠️ 密碼安全
這個密碼是你登入系統的唯一憑證。建議：
- 至少 12 個字元
- 包含大小寫字母、數字、符號
- 用密碼管理器記住
:::

### 2-1-4 網路設定：DHCP 或固定 IP

設定精靈會問你要怎麼連線到網路：

**DHCP（自動取得 IP）**：
- 適合：一般家用環境
- 好處：不用設定，插上網線就能用
- 壞處：每次開機 IP 可能不一樣

**固定 IP（Static IP）**：
- 適合：需要穩定遠端連線
- 好處：IP 永遠不變，方便 SSH
- 壞處：需要手動設定

::: tip 💡 建議做法
首次設定先用 DHCP，確定系統正常運作後，再到第 4 章改為固定 IP。
:::

### 2-1-5 啟用 SSH 遠端連線

SSH 是什麼？簡單來說，就是讓你可以從另一台電腦的「終端機」遠端操控 DGX Spark。

在 DGX OS 中，SSH 預設可能沒有開啟。開啟方法：

```bash
# 在 DGX Spark 的終端機中執行
sudo systemctl enable ssh
sudo systemctl start ssh
```

執行後，你就可以從另一台電腦連線了：

```bash
# 在你的個人電腦上執行（把 ip 換成 DGX Spark 的 IP）
ssh 你的使用者名稱@DGX_Spark_IP
```

::: tip 💡 怎麼知道 DGX Spark 的 IP？
在 DGX Spark 的終端機中執行：
```bash
ip addr show
```
找到 `eth0` 或 `wlan0` 那一段，`inet` 後面就是 IP 位址（例如 `192.168.1.100`）。
:::

### 2-1-6 插上 HDMI Dummy Plug，轉為 Headless 模式

::: info 🤔 什麼是 Headless 模式？
Headless = 無頭。意思是這台電腦不接顯示器、鍵盤、滑鼠，完全靠遠端操控。

這是 DGX Spark 的正常使用模式 — 設定完成後，它就安靜地待在角落，你用自己的筆電遠端控制它。
:::

步驟：

1. **確認 SSH 可以正常連線**（上一步）
2. **關機**：
   ```bash
   sudo shutdown now
   ```
3. **拔掉顯示器、鍵盤、滑鼠**
4. **插上 HDMI Dummy Plug**
5. **重新開機**

完成！現在你的 DGX Spark 就是一台 Headless 伺服器了。

---

## 2-2 DGX OS 概觀

### 2-2-1 DGX OS 版本：基於 Ubuntu 24.04

DGX OS 是 NVIDIA 基於 **Ubuntu 24.04 LTS** 客製化的作業系統。

為什麼選 Ubuntu？因為：
- 最多人使用的 Linux 發行版
- 軟體支援最完整
- 社群資源最豐富

::: info 🤔 我完全沒用過 Linux 怎麼辦？
不用擔心！本書會從最基礎的命令教起。你只需要知道：
- Linux 的「終端機」就像 Windows 的「命令提示字元」
- 你在終端機中輸入指令，系統就會執行
- 本書每個指令都會解釋在做什麼
:::

### 2-2-2 預裝軟體堆疊

DGX OS 出廠就幫你裝好了以下軟體：

| 軟體 | 用途 | 版本 |
|------|------|------|
| NVIDIA Driver | GPU 驅動 | 最新 |
| CUDA Toolkit | GPU 運算平台 | 13.0 |
| cuDNN | 深度學習加速庫 | 最新版 |
| Docker | 容器執行環境 | 最新版 |
| NGC CLI | NVIDIA GPU Cloud 命令列工具 | 最新版 |
| JupyterLab | 互動式程式開發環境 | 最新版 |

你**不需要**自己安裝這些，開箱就能用。

### 2-2-3 核心版本：Canonical Kernel 6.17

DGX OS 使用 Ubuntu 官方核心 **6.17**，這個版本針對 ARM 架構和 NVIDIA GPU 做了最佳化。

你可以用以下指令確認核心版本：

```bash
uname -r
# 輸出範例：6.17.0-nvidia-dgx
```

### 2-2-4 Container Runtime for Docker

DGX OS 使用 Docker 作為容器執行環境。

::: info 🤔 什麼是容器（Container）？
想像容器是一個「打包好的軟體包裹」，裡面包含了程式本身和所有需要的環境設定。

好處是：
- 不用擔心「在我的電腦上可以跑，在你的電腦上不行」
- 每個容器互相隔離，不會互相影響
- 可以輕鬆安裝、移除、更新
:::

本書中，幾乎所有的 AI 工具（Ollama、Open WebUI、vLLM 等）都會用 Docker 來部署。

確認 Docker 正常運作：

```bash
docker --version
# 輸出範例：Docker version 27.x.x

docker info
# 會顯示很多系統資訊，確認沒有錯誤訊息
```

### 2-2-5 NGC（NVIDIA GPU Cloud）存取

NGC 是 NVIDIA 的雲端服務，提供：
- 預先最佳化的 Docker 映像檔
- 預訓練模型
- AI 工具和 SDK

在 DGX OS 中，NGC CLI 已經預裝。你只需要註冊帳號並設定 API Key：

```bash
# 登入 NGC
ngc config set
```

系統會問你 API Key，你可以從 [NGC 網站](https://ngc.nvidia.com) 取得。

### 2-2-6 DGX Dashboard：Web 管理介面

DGX Dashboard 是一個網頁介面，讓你可以用瀏覽器管理 DGX Spark。

預設網址：`http://DGX_Spark_IP:8080`（實際 port 請參考系統說明）

Dashboard 可以：
- 查看系統狀態（CPU、GPU、記憶體使用量）
- 執行系統更新
- 管理 Docker 容器
- 查看日誌

---

## 2-3 系統更新

### 2-3-1 透過 DGX Dashboard 進行 OTA 更新

OTA = Over-The-Air，也就是無線更新。

1. 打開 DGX Dashboard
2. 點選「System Update」
3. 如果有新版本，會顯示「Update Available」
4. 點擊「Update」並等待完成

::: tip 💡 建議
收到 DGX Spark 後，**第一件事就是執行系統更新**。這能確保你拿到最新的驅動和軟體。
:::

### 2-3-2 手動更新 DGX OS 元件（apt）

除了 Dashboard 的 OTA 更新，你也可以用命令列更新：

```bash
# 1. 更新軟體列表
sudo apt update

# 2. 升級已安裝的軟體
sudo apt upgrade -y

# 3. 清理舊版本的快取
sudo apt autoremove -y
```

::: warning ⚠️ 注意
`apt upgrade` 可能會更新核心或驅動程式。更新後建議重新開機：
```bash
sudo reboot
```
:::

### 2-3-3 更新最佳實務

| 做法 | 說明 |
|------|------|
| ✅ 定期更新 | 每週檢查一次更新 |
| ✅ 更新前備份 | 用 Timeshift 建立快照（第 4 章會教） |
| ✅ 看更新日誌 | 了解更新了什麼 |
| ❌ 不要跳過太多版本 | 一次更新一個版本比較安全 |
| ❌ 不要在訓練中途更新 | 等訓練完成再更新 |

---

## 2-4 系統還原與救援模式

### 2-4-1 建立 Recovery USB

Recovery USB 是一個可以讓你還原系統到出廠狀態的 USB 隨身碟。

::: warning ⚠️ 建議盡快建立
收到 DGX Spark 並完成基本設定後，建議立刻建立 Recovery USB。萬一系統出問題，可以一鍵還原。
:::

建立方法：

```bash
# 1. 插入一個至少 16GB 的 USB 隨身碟

# 2. 執行建立指令（DGX OS 內建工具）
sudo dgx-recovery-usb create /dev/sdX
# 把 sdX 換成你的 USB 裝置名稱
```

::: danger 🚨 小心！
`/dev/sdX` 一定要確認是正確的 USB 裝置。如果選錯，可能會把硬碟的資料洗掉！

用 `lsblk` 指令確認：
```bash
lsblk
```
找到容量跟你 USB 隨身碟相符的那個裝置。
:::

### 2-4-2 使用 Recovery USB 還原系統

如果系統嚴重損壞，可以用 Recovery USB 還原：

1. 插入 Recovery USB
2. 重新開機，在 UEFI 畫面按 `F12` 選擇從 USB 開機
3. 選擇「Recover System」
4. 等待還原完成（約 15-20 分鐘）

::: warning ⚠️ 還原會清除所有資料
Recovery 會把系統回到出廠狀態，所有你自己安裝的軟體和資料都會消失。

所以平時要做好備份（第 4 章會教 Timeshift）。
:::

### 2-4-3 救援模式（Rescue Mode）

如果系統還能開機但有些問題，可以進入救援模式：

1. 重新開機
2. 在 GRUB 選單選擇「Advanced options」
3. 選擇帶有「(recovery mode)」的選項
4. 進入救援選單後，可以選擇：
   - `fsck`：檢查檔案系統
   - `clean`：清理空間
   - `dpkg`：修復損壞的套件
   - `root`：進入 root 命令列

---

## 2-5 NVIDIA Sync 客戶端

### 2-5-1 什麼是 NVIDIA Sync

NVIDIA Sync 是一個桌面應用程式，讓你可以：
- 從個人電腦管理 DGX Spark
- 一鍵 SSH 連線
- 自動配置 SSH 金鑰（不用每次打密碼）
- 查看系統狀態

它就像是 DGX Spark 的「遙控器」。

### 2-5-2 安裝 NVIDIA Sync

NVIDIA Sync 可以安裝在你的**個人電腦**上（不是 DGX Spark）：

```bash
# macOS
brew install --cask nvidia-sync

# Windows
# 從 NVIDIA 官網下載安裝程式
```

### 2-5-3 新增裝置

打開 NVIDIA Sync 後：

1. 點擊「Add Device」
2. 輸入 DGX Spark 的 IP 位址
3. 輸入使用者名稱和密碼
4. 點擊「Connect」

連線成功後，你會在列表中看到 DGX Spark。

### 2-5-4 自動 SSH Key 配置

NVIDIA Sync 可以自動幫你設定 SSH 金鑰認證：

1. 在 NVIDIA Sync 中右鍵點擊 DGX Spark
2. 選擇「Configure SSH Keys」
3. 按照指示完成

完成後，你從個人電腦 SSH 到 DGX Spark 就不需要打密碼了。

### 2-5-5 支援的應用程式

NVIDIA Sync 可以與以下應用程式整合：
- **VS Code**：遠端開發
- **Terminal**：直接開啟終端機視窗
- **File Browser**：瀏覽 DGX Spark 上的檔案

---

## 2-6 本章小結

::: success ✅ 你現在知道了
- 正確的接線順序：網路線 → HDMI → 鍵盤滑鼠 → 電源
- DGX OS 基於 Ubuntu 24.04，預裝了 CUDA、Docker、NGC 等工具
- SSH 是遠端操控的關鍵，一定要啟用
- Headless 模式需要 HDMI Dummy Plug
- 系統更新可以透過 Dashboard 或 apt 進行
- Recovery USB 是系統的最後防線
- NVIDIA Sync 讓遠端管理更方便
:::

::: tip 🚀 下一章預告
接下來我們要打造一個舒適的 Linux 工作環境，安裝 Zsh、Homebrew，還有本書的核心工具 — Claude Code。有了 Claude Code，你之後只需用自然語言就能完成各種複雜的部署工作！

👉 [前往第 3 章：Linux 環境建置與 Claude Code 安裝 →](/guide/chapter3/)
:::

::: info 📝 上一章
← [回到第 1 章：DGX Spark 硬體總覽](/guide/chapter1/)
:::
