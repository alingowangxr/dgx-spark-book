# 第 3 章：Linux 環境建置與 Claude Code 安裝

::: tip 🎯 本章你將學到什麼
- 把預設的 Bash 換成更強大的 Zsh + Oh My Zsh
- 安裝 Homebrew、uv、nvm 三大套件管理工具
- 安裝 ffmpeg 等多媒體工具
- 安裝並設定 Claude Code — 本書的核心 AI 輔助工具
- 了解為什麼之後的章節都用 Claude Code 來操作
:::

::: warning ⏱️ 預計閱讀時間
約 15 分鐘。實際安裝約需 10-15 分鐘。
:::

---

## 3-1 Shell 環境

### 3-1-1 安裝 Zsh

::: info 🤔 什麼是 Shell？為什麼要換？
Shell 是你跟 Linux 系統對話的介面。DGX OS 預設用的是 Bash，但 Zsh 更好用：

| 功能 | Bash | Zsh |
|------|------|-----|
| 自動補全 | 基本 | 智慧（會根據上下文提示） |
| 語法高亮 | 無 | 有（指令打錯會變紅色） |
| 外掛支援 | 有限 | 豐富 |
| 主題 | 無 | 超多 |

簡單說：Zsh 就是 Bash 的加強版。
:::

```bash
# 安裝 Zsh
sudo apt install -y zsh

# 確認安裝成功
zsh --version
# 輸出範例：zsh 5.9 (aarch64-ubuntu-linux-gnu)
```

### 3-1-2 安裝 Oh My Zsh

Oh My Zsh 是一個 Zsh 的管理框架，讓你輕鬆安裝主題和外掛。

```bash
# 一鍵安裝 Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

執行後，你的終端機會自動切換到 Zsh，並問你是否要設為預設 Shell。輸入 `Y` 確認。

::: tip 💡 如果安裝失敗
有些網路環境可能無法直接連線到 GitHub。如果上面的指令失敗，改用：
```bash
# 先安裝 git
sudo apt install -y git

# 手動安裝
git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
```
:::

### 3-1-3 安裝主題與外掛

Oh My Zsh 裝好後，預設主題是 `robbyrussell`。我們來換一個更好看的，並安裝實用的外掛。

**更換主題**：

```bash
# 編輯 Zsh 設定檔
nano ~/.zshrc
```

找到 `ZSH_THEME="robbyrussell"` 這行，改成：

```bash
ZSH_THEME="agnoster"
```

::: info 🤔 agnoster 主題長什麼樣？
它會顯示：
- 目前的使用者和主機名稱
- 目前的工作目錄
- Git 分支狀態
- 指令執行成功/失敗的提示

非常適合開發者使用。
:::

**安裝實用外掛**：

```bash
# 1. zsh-autosuggestions（自動提示歷史指令）
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# 2. zsh-syntax-highlighting（語法高亮）
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

然後編輯 `~/.zshrc`，找到 `plugins=(git)` 這行，改成：

```bash
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
```

最後重新載入設定：

```bash
source ~/.zshrc
```

### 3-1-4 設定 Zsh 為預設 Shell

如果安裝 Oh My Zsh 時沒有自動設定，可以手動設定：

```bash
# 設定 Zsh 為預設 Shell
chsh -s $(which zsh)

# 確認設定成功
echo $SHELL
# 輸出應該是：/usr/bin/zsh
```

::: warning ⚠️ 注意
`chsh` 設定後需要**登出再登入**才會生效。如果是 SSH 連線，斷線重連即可。
:::

---

## 3-2 套件管理工具

### 3-2-1 安裝 Homebrew（Linuxbrew）

::: info 🤔 什麼是 Homebrew？
Homebrew 原本是 macOS 的套件管理工具，現在也支援 Linux。它的好處是：
- 可以安裝很多 `apt` 沒有的工具
- 不需要 sudo 權限就能安裝
- 版本更新快
:::

```bash
# 安裝 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安裝完成後，把 Homebrew 加入 PATH
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# 確認安裝成功
brew --version
```

::: tip 💡 常用 Brew 指令
```bash
brew install <套件名>    # 安裝套件
brew search <關鍵字>     # 搜尋套件
brew update              # 更新 Homebrew 本身
brew upgrade             # 更新所有已安裝套件
brew list                # 列出已安裝的套件
```
:::

### 3-2-2 安裝 uv：Python 套件管理

::: info 🤔 什麼是 uv？為什麼不用 pip？
uv 是一個超快速的 Python 套件管理工具，由 Rust 撰寫。它比 pip 快 10-100 倍，而且功能更強大。

在 AI 領域，我們經常需要：
- 建立獨立的 Python 環境（避免套件衝突）
- 安裝特定版本的 PyTorch、TensorFlow
- 管理多個專案的不同依賴

uv 讓這些事情變得超簡單。
:::

```bash
# 用 Homebrew 安裝 uv
brew install uv

# 確認安裝成功
uv --version
```

**uv 基本用法**：

```bash
# 建立一個新的 Python 專案環境
uv venv .venv

# 啟動虛擬環境
source .venv/bin/activate

# 安裝套件
uv pip install torch numpy

# 從 requirements.txt 安裝
uv pip install -r requirements.txt
```

### 3-2-3 安裝 nvm：Node.js 版本管理

```bash
# 安裝 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# 重新載入設定
source ~/.zshrc

# 安裝 Node.js 最新版
nvm install node

# 確認安裝成功
node --version
npm --version
```

為什麼需要 nvm？因為有些 AI 工具（如 Open WebUI 的開發版）需要特定版本的 Node.js。

### 3-2-4 uv vs. pip vs. conda：何時用哪個

| 工具 | 適合場景 | 優點 | 缺點 |
|------|---------|------|------|
| **uv** | 一般 Python 專案 | 超快、現代化、輕量 | 比較新，社群還在成長 |
| **pip** | 簡單的一次性安裝 | 內建在 Python 中 | 慢、依賴管理較弱 |
| **conda** | 科學計算、需要非 Python 依賴 | 可以安裝非 Python 套件 | 體積大、速度慢 |

::: tip 💡 本書的建議
- **優先使用 Docker 容器**（本書大部分工具都用 Docker 部署）
- 需要 Python 環境時用 **uv**
- 只有在 Docker 不適用時才考慮 **conda**
:::

---

## 3-3 多媒體與系統工具

### 3-3-1 安裝 ffmpeg

ffmpeg 是一個強大的影音處理工具，在第 14 章（音訊處理）會大量使用。

```bash
# 用 apt 安裝
sudo apt install -y ffmpeg

# 確認安裝成功
ffmpeg -version
```

**ffmpeg 常用操作**：

```bash
# 轉換音訊格式
ffmpeg -i input.wav -codec:a libmp3lame -qscale:a 2 output.mp3

# 從影片提取音訊
ffmpeg -i video.mp4 -vn -acodec libmp3lame audio.mp3

# 合併音訊和影片
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac output.mp4

# 調整音量
ffmpeg -i input.mp3 -filter:a "volume=1.5" output.mp3
```

### 3-3-2 安裝常用系統工具

```bash
# 安裝實用工具套件
sudo apt install -y \
  htop \
  tmux \
  tree \
  jq \
  wget \
  curl \
  git \
  unzip \
  p7zip-full \
  neofetch
```

| 工具 | 用途 |
|------|------|
| `htop` | 互動式程序查看器（比 `top` 好用） |
| `tmux` | 終端機多工器（斷線後程式繼續跑） |
| `tree` | 以樹狀圖顯示目錄結構 |
| `jq` | JSON 處理工具 |
| `neofetch` | 顯示系統資訊（裝帥用 😄） |

### 3-3-3 確認 Docker Compose

DGX OS 已經預裝 Docker，但我們需要確認 Docker Compose 也正常：

```bash
# 確認 Docker Compose 版本
docker compose version
# 輸出範例：Docker Compose version v2.x.x
```

::: info 🤔 `docker-compose` vs `docker compose`？
- `docker-compose`（有連字號）是舊版的獨立工具
- `docker compose`（沒連字號）是 Docker 內建的外掛

本書統一使用 `docker compose`（新版語法）。
:::

如果 Docker Compose 沒有安裝：

```bash
sudo apt install -y docker-compose-plugin
```

---

## 3-4 安裝 Claude Code

### 3-4-1 什麼是 Claude Code

::: tip 💡 本書的核心工具
Claude Code 是 Anthropic 推出的命令列 AI 助手。它能理解你的自然語言指令，自動完成：
- 安裝和設定軟體
- 編寫和修改程式碼
- 除錯和修復問題
- 閱讀和分析檔案

從第 5 章開始，本書幾乎每個章節都會用 Claude Code 來完成操作。你只需要告訴它「幫我裝 Ollama」，它就會自動執行所有需要的指令。
:::

### 3-4-2 原生安裝（推薦）

```bash
# 用 npm 全域安裝 Claude Code
npm install -g @anthropic-ai/claude-code
```

::: warning ⚠️ 權限問題
如果出現權限錯誤，不要加 `sudo`！改用以下方法：

```bash
# 方法 1：用 npx（不需要安裝）
npx @anthropic-ai/claude-code

# 方法 2：設定 npm 全域安裝目錄
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
npm install -g @anthropic-ai/claude-code
```
:::

### 3-4-3 其他安裝方式

**方法 2：用 Homebrew**

```bash
brew install anthropic/claude-code/claude-code
```

**方法 3：直接用 npx（免安裝）**

```bash
npx @anthropic-ai/claude-code
```

### 3-4-4 驗證安裝

```bash
claude --version
# 輸出範例：Claude Code 1.x.x
```

如果看到版本號，代表安裝成功。

### 3-4-5 認證與登入

第一次執行 Claude Code 時，它會引導你完成登入：

```bash
claude
```

你會看到：
1. 一個登入網址和驗證碼
2. 用瀏覽器打開該網址，貼上驗證碼
3. 用你的 Anthropic 帳號登入

::: info 🤔 需要什麼帳號？
你需要一個 **Anthropic API 帳號**。如果還沒註冊：
1. 前往 [console.anthropic.com](https://console.anthropic.com)
2. 用 Google 或 GitHub 帳號註冊
3. 取得 API Key
:::

### 3-4-6 自動更新機制

Claude Code 會自動檢查更新。如果你想手動更新：

```bash
npm update -g @anthropic-ai/claude-code
```

---

## 3-5 本書的 Claude Code 操作模式

### 3-5-1 從第 5 章起的操作原則

從第 5 章開始，本書的操作模式是：

```
你（用自然語言告訴 Claude Code 你想做什麼）
  → Claude Code（自動執行需要的指令）
    → 你（確認結果）
```

舉例來說，第 5 章要裝 Ollama，你不需要自己查安裝指令，只需要對 Claude Code 說：

> 「幫我安裝 Ollama，並確認服務正常運作」

Claude Code 就會自動完成所有步驟。

### 3-5-2 操作流程

```
1. 在終端機中輸入 claude 啟動
2. 用自然語言描述你想做的事
3. Claude Code 會提出執行計劃
4. 你確認後，它會自動執行
5. 檢查結果，如果有問題就告訴 Claude Code
```

### 3-5-3 為什麼用 Claude Code

| 傳統方式 | 用 Claude Code |
|----------|---------------|
| 查文件找安裝指令 | 直接告訴 AI 你想做什麼 |
| 手動複製貼上指令 | AI 自動執行指令 |
| 出錯時自己除錯 | AI 自動分析錯誤並修復 |
| 需要記住大量指令 | 用自然語言溝通 |

::: tip 💡 但你還是需要理解原理
雖然 Claude Code 能自動完成很多事，但本書還是會詳細解釋每個步驟的原理。因為：
- 了解原理才能除錯
- 了解原理才能举一反三
- AI 也會出錯，你需要能判斷
:::

### 3-5-4 Claude Code 常用操作技巧

**1. 給明確的指令**

```
❌ 不好的指令：「裝那個 AI 工具」
✅ 好的指令：「幫我安裝 Ollama，下載 Nemotron-3-Super 120B 模型，並確認可以正常對話」
```

**2. 分步驟進行**

```
✅ 好的做法：
  第一步：「幫我安裝 Ollama」
  第二步：「下載 Nemotron-3-Super 120B 模型」
  第三步：「測試對話功能」
```

**3. 提供上下文**

```
✅ 好的指令：
  「我在 DGX Spark 上，DGX OS 基於 Ubuntu 24.04 ARM64，
   請幫我安裝適合這個平台的 Ollama」
```

**4. 善用 CLAUDE.md 設定檔**

在專案目錄中建立 `CLAUDE.md` 檔案，可以告訴 Claude Code 你的偏好：

```markdown
# CLAUDE.md

## 環境
- 系統：DGX OS (Ubuntu 24.04 ARM64)
- GPU：NVIDIA Blackwell, 128GB 統一記憶體
- Docker：已安裝

## 偏好
- 優先使用 Docker 部署
- Python 環境使用 uv
- 所有指令都要解釋在做什麼
```

---

## 3-6 本章小結

::: success ✅ 你現在知道了
- Zsh + Oh My Zsh 讓終端機更好用
- Homebrew、uv、nvm 是三大套件管理工具
- ffmpeg 是影音處理的瑞士刀
- Claude Code 是本書的核心 AI 輔助工具
- 用自然語言就能讓 Claude Code 幫你完成各種操作
:::

::: tip 🚀 下一章預告
現在你的 DGX Spark 已經有了舒適的工作環境。下一章我們要設定遠端桌面和網路存取，讓你可以從任何地方控制 DGX Spark，包括從公司、咖啡廳、甚至國外！

👉 [前往第 4 章：遠端桌面與網路存取 →](/guide/chapter4/)
:::

::: info 📝 上一章
← [回到第 2 章：DGX OS 安裝與首次開機](/guide/chapter2/)
:::
