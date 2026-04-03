# 第 14 章：音訊、語音與音樂 AI

::: tip 🎯 本章你將學到什麼
- 部署 Qwen3-TTS 中文語音合成
- 部署 faster-whisper 語音辨識
- 部署 ACE-Step 1.5 AI 音樂生成
- ffmpeg 音訊處理工具鏈
- Demucs 人聲分離
- 完整工作流程串接實戰
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 14-1 語音合成（Text-to-Speech）

語音合成（TTS, Text-to-Speech）是將文字轉換為自然語音的技術。在 DGX Spark 上，你可以部署高品質的中文 TTS 模型，甚至複製自己的聲音。

### 14-1-1 TTS 模型比較

選擇合適的 TTS 模型取決於你的需求：語言支援、音質要求、記憶體限制和生成速度。

| 模型 | 語言支援 | 記憶體需求 | 音質 | 速度 | 聲音複製 | 適合場景 |
|------|---------|-----------|------|------|---------|---------|
| **Qwen3-TTS** | 中文 + 英文 | ~2 GB | 高 | 快 | ✅ | 通用、產品配音 |
| CosyVoice | 中文為主 | ~4 GB | 高 | 中等 | ✅ | 中文廣播、有聲書 |
| XTTS v2 | 16+ 語言 | ~4 GB | 中高 | 中等 | ✅ | 多語言內容 |
| Piper | 20+ 語言 | ~0.5 GB | 中 | 超快 | ❌ | 嵌入式、即時應用 |
| VITS | 多語言 | ~1 GB | 中高 | 快 | ❌ | 研究、客製化 |

::: info 🤔 什麼是聲音複製（Voice Cloning）？
聲音複製是指讓 TTS 模型用特定人物的聲音來說話。你只需要提供 10-30 秒的參考音訊，模型就能學習這個人的聲音特徵（音調、口音、語速），然後用這個聲音說出任何文字。

應用場景：
- 為自己的 YouTube 頻道生成配音
- 用已故親人的聲音朗讀文字
- 為遊戲角色建立一致的語音
- 建立個人語音助手
:::

### 14-1-2 Qwen3-TTS 的架構

Qwen3-TTS 基於端到端的神聲網路架構，與傳統的 TTS 系統有顯著不同：

**傳統 TTS 流程**：
```
文字 → 文字正規化 → 音素轉換 → 聲學模型 → 聲碼器 → 音訊
```
每一步都可能引入誤差，而且中文的音素轉換特別容易出錯（多音字問題）。

**Qwen3-TTS 流程**：
```
文字 → 端到端模型 → 音訊波形
```
直接從文字生成音訊，跳過了中間步驟，好處是：
- 中文發音更自然（不會有多音字錯誤）
- 語調和停頓更符合人類習慣
- 支援多情感（開心、悲傷、正式、輕鬆）
- 推理速度更快（步驟更少）

### 14-1-3 部署 Qwen3-TTS

```bash
# 用 Docker 部署 Qwen3-TTS
docker run -d \
  --name qwen3-tts \
  --gpus all \
  --network host \
  -v ~/tts-output:/app/output \
  -v ~/tts-reference:/app/reference \
  ghcr.io/community/qwen3-tts:latest
```

各參數說明：

| 參數 | 說明 |
|------|------|
| `--gpus all` | 啟用 GPU 加速 |
| `--network host` | 使用主機網路，方便存取 |
| `-v ~/tts-output:/app/output` | 映射輸出目錄，生成的音訊會出現在這裡 |
| `-v ~/tts-reference:/app/reference` | 映射參考音訊目錄，用於聲音複製 |

等待容器啟動完成後，測試 API 是否正常運作：

```bash
# 測試基本語音合成
curl http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，歡迎使用 DGX Spark 語音合成系統。",
    "output": "/app/output/test.wav"
  }'
```

成功後，你會在 `~/tts-output/` 目錄中找到 `test.wav` 檔案。

### 14-1-4 中文語音合成實測

以下是更完整的 Python 呼叫範例，包含所有可用參數：

```python
import requests
import os

# API 端點
TTS_API = "http://localhost:8000/tts"

# 基本語音合成
response = requests.post(
    TTS_API,
    json={
        "text": "塞爆 128G GPU 記憶體，DGX Spark 全面玩透！",
        "speaker": "zh-CN-YunxiNeural",  # 說話人
        "speed": 1.0,                     # 語速（0.5-2.0）
        "pitch": 0,                       # 音調（-100 到 +100）
        "volume": 1.0,                    # 音量（0.0-1.0）
        "output": "/app/output/dgx-spark.wav"
    }
)

print(f"狀態碼: {response.status_code}")
print(f"回應: {response.json()}")

# 確認檔案已生成
if os.path.exists("/home/user/tts-output/dgx-spark.wav"):
    print("✅ 音訊檔案生成成功！")
```

**可用說話人列表**：

| 說話人 ID | 性別 | 風格 | 適合場景 |
|----------|------|------|---------|
| `zh-CN-YunxiNeural` | 男 | 自然、親切 | 一般對話、Podcast |
| `zh-CN-YunyangNeural` | 男 | 專業、穩重 | 新聞播報、教學 |
| `zh-CN-XiaoxiaoNeural` | 女 | 溫柔、親切 | 有聲書、客服 |
| `zh-CN-XiaoyiNeural` | 女 | 活潑、年輕 | 兒童內容、廣告 |
| `zh-CN-YunjianNeural` | 男 | 體育解說 | 運動、熱血內容 |
| `zh-CN-XiaochenNeural` | 女 | 兒童 | 兒童故事、教育 |

### 14-1-5 語音複製（Voice Cloning）

Qwen3-TTS 支援零樣本聲音複製（Zero-shot Voice Cloning），只需要一段參考音訊即可。

**準備參考音訊的要求**：
- 時長：10-30 秒（越長效果越好）
- 格式：WAV 或 MP3
- 品質：清晰、無背景雜訊
- 內容：自然說話即可，不需要特定文字

```bash
# 聲音複製 API 呼叫
curl http://localhost:8000/tts-clone \
  -H "Content-Type: application/json" \
  -d '{
    "text": "這段話是用你的聲音說的。歡迎使用 DGX Spark 語音複製功能。",
    "reference_audio": "/app/reference/my-voice.wav",
    "output": "/app/output/cloned.wav"
  }'
```

**Python 批次聲音複製**：

```python
import requests

CLONE_API = "http://localhost:8000/tts-clone"

# 用同一個聲音複製多段文字
texts = [
    "第一段：歡迎來到 DGX Spark 教學課程。",
    "第二段：今天我們要學習如何部署 AI 模型。",
    "第三段：感謝您的收看，我們下次見。"
]

for i, text in enumerate(texts):
    response = requests.post(
        CLONE_API,
        json={
            "text": text,
            "reference_audio": "/app/reference/my-voice.wav",
            "output": f"/app/output/cloned_part{i+1}.wav"
        }
    )
    print(f"第 {i+1} 段: {response.status_code}")
```

::: tip 💡 聲音複製品質提升技巧
1. **參考音訊品質**：使用高品質錄音，避免背景雜訊
2. **參考音訊長度**：至少 15 秒，30 秒以上效果更佳
3. **參考音訊內容**：涵蓋不同的音調和語速
4. **文字長度匹配**：生成的文字長度不要遠超過參考音訊
5. **後製處理**：用 ffmpeg 添加輕微的混響可以讓聲音更自然
:::

### 14-1-6 進階：多情感語音合成

Qwen3-TTS 支援不同的情感風格：

```python
# 不同情感的語音合成
emotions = [
    {"emotion": "happy", "text": "太棒了！我們成功完成了！"},
    {"emotion": "sad", "text": "很遺憾地告訴大家這個消息。"},
    {"emotion": "neutral", "text": "今天的會議將在下午三點舉行。"},
    {"emotion": "excited", "text": "不可思議！這簡直是革命性的突破！"}
]

for e in emotions:
    requests.post(TTS_API, json={
        "text": e["text"],
        "emotion": e["emotion"],
        "output": f"/app/output/emotion_{e['emotion']}.wav"
    })
```

---

## 14-2 語音辨識（Speech-to-Text）

語音辨識（STT, Speech-to-Text）是將語音轉換為文字的技術。faster-whisper 是目前開源領域最高效的語音辨識引擎。

### 14-2-1 Whisper 模型比較

OpenAI 的 Whisper 模型有多種尺寸，選擇哪一種取決於你的準確率需求和硬體條件：

| 模型 | 參數量 | 記憶體需求 | 中文辨識率 | 速度（RTF） | 適合場景 |
|------|--------|-----------|-----------|------------|---------|
| tiny | 39M | ~1 GB | 低（~60%） | 0.05 | 快速測試、資源受限 |
| base | 74M | ~1 GB | 中低（~70%） | 0.08 | 簡單指令辨識 |
| small | 244M | ~2 GB | 中（~78%） | 0.15 | 一般會議記錄 |
| medium | 769M | ~3 GB | 高（~85%） | 0.25 | 正式文件轉寫 |
| **large-v3** | 1550M | ~5 GB | **最高（~90%）** | 0.40 | 高精度需求 |

::: info 🤔 什麼是 RTF？
RTF（Real-Time Factor）是語音辨識速度的指標。RTF = 0.40 表示處理 1 分鐘的音訊需要 0.40 分鐘（24 秒）。RTF 越低越快。DGX Spark 的 GPU 加速可以讓 RTF 更低。
:::

### 14-2-2 部署 faster-whisper

faster-whisper 是 Whisper 的加速版本，使用 CTranslate2 引擎，速度比原版快 4 倍，記憶體用量減少一半。

```bash
# 用 Docker 部署 faster-whisper
docker run -d \
  --name faster-whisper \
  --gpus all \
  --network host \
  -v ~/audio-files:/app/audio \
  ghcr.io/community/faster-whisper:latest
```

### 14-2-3 語音辨識實測

**基本轉寫**：

```bash
# 轉寫中文音訊
curl http://localhost:8000/transcribe \
  -F "audio=@/app/audio/meeting.wav" \
  -F "language=zh" \
  -F "model=large-v3" \
  -F "task=transcribe"
```

**Python 完整範例**：

```python
import requests
import json

WHISPER_API = "http://localhost:8000/transcribe"

# 基本轉寫
response = requests.post(
    WHISPER_API,
    files={
        "audio": open("/home/user/audio-files/meeting.wav", "rb")
    },
    data={
        "language": "zh",
        "model": "large-v3",
        "task": "transcribe",
        "word_timestamps": "true",      # 回傳每個字的時間戳
        "vad_filter": "true",           # 啟用語音活動檢測
        "initial_prompt": "以下是中文會議記錄"  # 提示詞幫助辨識
    }
)

result = response.json()
print(f"轉寫結果: {result['text']}")
print(f"耗時: {result['processing_time']:.2f} 秒")

# 儲存為 JSON
with open("/home/user/audio-files/transcript.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
```

**進階參數說明**：

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `language` | 語言代碼 | `zh`（中文）、`en`（英文） |
| `model` | 模型大小 | `large-v3`（最佳品質） |
| `task` | 任務類型 | `transcribe`（轉寫）、`translate`（翻譯為英文） |
| `word_timestamps` | 字級時間戳 | `true`（用於字幕生成） |
| `vad_filter` | 語音活動檢測 | `true`（過濾靜音段） |
| `temperature` | 取樣溫度 | `0`（最確定）或 `[0.0, 0.2, 0.4]`（自動回退） |
| `beam_size` | 束搜尋大小 | `5`（品質與速度平衡） |

### 14-2-4 字幕生成

自動為影片或 Podcast 生成 SRT 字幕檔：

```bash
# 生成 SRT 字幕
curl http://localhost:8000/subtitle \
  -F "audio=@/app/audio/podcast.mp3" \
  -F "language=zh" \
  -F "format=srt" \
  -F "model=large-v3" \
  -o /app/audio/podcast.srt
```

生成的 SRT 檔案格式：
```
1
00:00:01,000 --> 00:00:04,500
歡迎收聽 DGX Spark 教學 Podcast

2
00:00:05,000 --> 00:00:09,200
今天我們要介紹如何在本地部署 AI 模型
```

**Python 批次字幕生成**：

```python
import requests
import glob

SUBTITLE_API = "http://localhost:8000/subtitle"

# 批次處理所有 MP3 檔案
for audio_file in glob.glob("/home/user/audio-files/*.mp3"):
    filename = audio_file.split("/")[-1].replace(".mp3", "")
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            SUBTITLE_API,
            files={"audio": f},
            data={
                "language": "zh",
                "format": "srt",
                "model": "large-v3"
            }
        )
    
    with open(f"/home/user/audio-files/{filename}.srt", "wb") as out:
        out.write(response.content)
    
    print(f"✅ {filename}.srt 已生成")
```

### 14-2-5 即時語音轉文字

從麥克風即時轉寫語音，適合會議記錄、即時翻譯等場景：

```bash
# 從麥克風即時轉寫
docker run -it \
  --gpus all \
  --network host \
  --device /dev/snd \
  ghcr.io/community/faster-whisper:latest \
  python live_transcribe.py --language zh --model large-v3
```

這會啟動一個即時監聽模式，你說的話會即時轉換為文字顯示在終端機上。

::: tip 💡 即時轉寫優化
- 使用 USB 麥克風獲得更好的音質
- 在安靜的環境中使用
- 說話時保持適當距離（15-30 公分）
- 如果辨識率不佳，嘗試切換到 `medium` 模型（速度更快，適合即時）
:::

---

## 14-3 AI 音樂生成

AI 音樂生成是近年來發展最快的領域之一。在 DGX Spark 上，你可以用 ACE-Step 1.5 生成高品質的純音樂和歌曲。

### 14-3-1 音樂生成模型比較

| 模型 | 記憶體需求 | 生成時長 | 音質 | 支援語言 | 特點 |
|------|-----------|---------|------|---------|------|
| **ACE-Step 1.5** | ~8 GB | 30-120 秒 | 高 | 多語言 | 支援歌曲生成、中文優化 |
| MusicGen (Meta) | ~6 GB | 10-30 秒 | 中 | 英文為主 | 開源、社群資源多 |
| Riffusion | ~4 GB | 10-15 秒 | 中低 | 英文為主 | 最輕量、速度快 |
| AudioCraft | ~8 GB | 10-30 秒 | 中高 | 多語言 | Meta 出品、功能豐富 |

### 14-3-2 部署 ACE-Step 1.5

```bash
docker run -d \
  --name ace-step \
  --gpus all \
  --network host \
  -v ~/music-output:/app/output \
  ghcr.io/community/ace-step:1.5
```

等待容器啟動後，訪問 `http://DGX_Spark_IP:7860` 可以看到 Web 介面。

### 14-3-3 生成純音樂

```bash
curl http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "輕鬆的鋼琴曲，適合早晨聆聽，柔和的旋律，帶有希望的感覺",
    "duration": 60,
    "guidance_scale": 7.5,
    "steps": 50,
    "seed": 42,
    "output": "/app/output/morning-piano.wav"
  }'
```

**Python 批次生成**：

```python
import requests

MUSIC_API = "http://localhost:7860/generate"

# 生成一系列背景音樂
prompts = [
    {
        "prompt": "輕鬆的鋼琴曲，適合早晨聆聽",
        "filename": "morning-piano.wav"
    },
    {
        "prompt": "深沉的大提琴獨奏，適合夜晚閱讀",
        "filename": "night-cello.wav"
    },
    {
        "prompt": "活潑的吉他旋律，適合咖啡廳背景音樂",
        "filename": "cafe-guitar.wav"
    },
    {
        "prompt": "電子環境音樂，適合專注工作",
        "filename": "focus-ambient.wav"
    }
]

for p in prompts:
    response = requests.post(
        MUSIC_API,
        json={
            "prompt": p["prompt"],
            "duration": 60,
            "guidance_scale": 7.5,
            "steps": 50,
            "output": f"/app/output/{p['filename']}"
        }
    )
    print(f"✅ {p['filename']}: {response.status_code}")
```

### 14-3-4 生成中文歌曲

ACE-Step 1.5 支援歌詞生成，可以根據歌詞和風格描述生成完整的歌曲：

```bash
curl http://localhost:7860/generate-song \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "今天天氣真好，適合出去走走\n陽光灑在臉上，心情特別輕鬆\n讓我們一起出發，探索這個美麗的世界",
    "style": "流行音樂，輕快，吉他伴奏",
    "language": "zh",
    "duration": 90,
    "guidance_scale": 8.0,
    "steps": 75,
    "output": "/app/output/song.wav"
  }'
```

**歌詞格式注意事項**：
- 用換行符 `\n` 分隔每一句
- 可以用 `[Verse]`、`[Chorus]`、`[Bridge]` 標記段落
- 歌詞越具體，生成效果越好

```python
# 進階歌詞格式
lyrics = """[Verse]
清晨的陽光灑在窗台
新的一天充滿期待
背上行囊出發去遠方
未知的世界等著我探索

[Chorus]
走吧走吧不要猶豫
世界那么大我想去看看
每一步都是新的發現
每一次都是新的成長

[Verse 2]
穿過森林越過山丘
看見了最美的日落
"""

requests.post("http://localhost:7860/generate-song", json={
    "lyrics": lyrics,
    "style": "民謠，木吉他，溫暖的聲音",
    "language": "zh",
    "duration": 120,
    "output": "/app/output/folk-song.wav"
})
```

### 14-3-5 進階設定詳解

| 參數 | 說明 | 建議值 | 影響 |
|------|------|--------|------|
| `guidance_scale` | 提示詞遵循度 | 7-9 | 越高越忠於提示詞，但可能不自然 |
| `steps` | 生成步數 | 50-100 | 越高品質越好，但越慢 |
| `seed` | 隨機種子 | 固定值可重現 | 相同 seed + 相同參數 = 相同結果 |
| `cfg_rescale` | CFG 重縮放 | 0.7 | 防止過度擬合提示詞 |
| `duration` | 生成時長（秒） | 30-120 | 越長需要越多記憶體和時間 |
| `sample_rate` | 取樣率 | 44100 | 音訊品質，44100 為 CD 品質 |

::: tip 💡 音樂生成提示詞技巧
好的提示詞格式：
```
[風格/流派] + [樂器] + [情緒/氛圍] + [節奏] + [參考（可選）]

例如：
「爵士樂，鋼琴和薩克斯風，慵懶的夜晚氛圍，中慢板，類似 Bill Evans 的風格」
「電子舞曲，強烈的 bassline，充滿能量，128 BPM，適合派對」
```
:::

---

## 14-4 音訊處理工具鏈

### 14-4-1 ffmpeg 基本操作

ffmpeg 是音訊/影片處理的瑞士刀，第 3 章已安裝。這裡整理最常用的操作：

**格式轉換**：
```bash
# WAV 轉 MP3（壓縮）
ffmpeg -i input.wav -codec:a libmp3lame -qscale:a 2 output.mp3

# MP3 轉 WAV（用於 TTS 輸入）
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# 轉為 OGG（網頁友善）
ffmpeg -i input.wav -c:a libvorbis -q:a 5 output.ogg
```

**剪輯音訊**：
```bash
# 剪輯：從 1:00 到 2:00
ffmpeg -i input.mp3 -ss 00:01:00 -to 00:02:00 -c copy output.mp3

# 裁剪前 30 秒
ffmpeg -i input.mp3 -t 30 -c copy output.mp3
```

**合併音訊**：
```bash
# 先建立檔案列表
echo "file 'part1.mp3'" > filelist.txt
echo "file 'part2.mp3'" >> filelist.txt
echo "file 'part3.mp3'" >> filelist.txt

# 合併
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp3
```

**音量調整**：
```bash
# 放大 1.5 倍
ffmpeg -i input.mp3 -filter:a "volume=1.5" output.mp3

# 標準化音量（推薦）
ffmpeg -i input.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" output.mp3
```

**淡入淡出**：
```bash
# 3 秒淡入 + 3 秒淡出（總長 60 秒）
ffmpeg -i input.mp3 \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=57:d=3" \
  output.mp3
```

**音訊資訊查看**：
```bash
# 查看音訊檔詳細資訊
ffprobe -v quiet -print_format json -show_format -show_streams input.wav
```

### 14-4-2 Demucs 人聲分離

Demucs 是 Meta 開源的人聲分離模型，基於深度學習，可以將歌曲分離為多個音軌。

```bash
# 用 uv 安裝 Demucs
uv pip install demucs

# 基本用法：分離人聲和伴奏
demucs --two-stems=vocals song.mp3 -o ~/separated/
```

輸出結構：
```
~/separated/
└── song/
    ├── vocals.wav      # 人聲
    └── no_vocals.wav   # 伴奏（無人聲）
```

### 14-4-3 人聲分離實測

**四軌分離**（人聲、鼓、貝斯、其他）：

```bash
# 分離所有音軌
demucs --all song.mp3 -o ~/separated/
```

輸出：
```
~/separated/
└── song/
    ├── vocals.wav    # 人聲
    ├── drums.wav     # 鼓
    ├── bass.wav      # 貝斯
    └── other.wav     # 其他樂器
```

**進階參數**：

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--two-stems` | 只分離人聲和伴奏 | `vocals` |
| `--all` | 分離所有音軌 | 需要更多處理時間 |
| `-n` | 使用的模型 | `htdemucs`（預設，最佳） |
| `--device` | 計算裝置 | `cuda`（GPU 加速） |
| `--jobs` | 平行處理數量 | `2`（DGX Spark 建議） |

```bash
# 使用 GPU 加速，平行處理 2 首歌曲
demucs --all --device cuda --jobs 2 \
  song1.mp3 song2.mp3 \
  -o ~/separated/
```

::: tip 💡 Demucs 使用技巧
1. **品質優先**：使用 `htdemucs` 模型（預設），品質最好
2. **速度優先**：使用 `htdemucs_fast` 模型，速度快但品質稍差
3. **卡拉 OK 製作**：`--two-stems=vocals` 分離後，用 `no_vocals.wav` 當伴奏
4. **Remix 製作**：分離所有音軌後，可以重新混音
:::

### 14-4-4 完整工作流程串接實戰

讓我們把前面學到的所有工具串接成一個完整的工作流程：

```
1. 錄音/下載歌曲
   ↓
2. Demucs 分離人聲
   ↓
3. faster-whisper 轉寫歌詞
   ↓
4. ACE-Step 用同樣風格生成新歌曲
   ↓
5. ffmpeg 合併和後製
   ↓
6. Qwen3-TTS 生成旁白/介紹
```

**Python 自動化腳本**：

```python
import requests
import subprocess
import os
import json

class AudioWorkflow:
    """完整的音訊處理工作流程"""
    
    def __init__(self):
        self.tts_api = "http://localhost:8000/tts"
        self.whisper_api = "http://localhost:8000/transcribe"
        self.music_api = "http://localhost:7860/generate"
        self.output_dir = "/app/output"
    
    def separate_vocals(self, input_file):
        """步驟 1：分離人聲"""
        print("🎵 正在分離人聲...")
        result = subprocess.run([
            "demucs", "--two-stems=vocals",
            "--device", "cuda",
            input_file,
            "-o", f"{self.output_dir}/separated"
        ], capture_output=True, text=True)
        print(f"✅ 人聲分離完成")
        return f"{self.output_dir}/separated"
    
    def transcribe_audio(self, audio_file):
        """步驟 2：轉寫歌詞"""
        print("📝 正在轉寫歌詞...")
        with open(audio_file, "rb") as f:
            response = requests.post(
                self.whisper_api,
                files={"audio": f},
                data={"language": "zh", "model": "large-v3"}
            )
        lyrics = response.json()["text"]
        print(f"✅ 轉寫完成: {lyrics[:50]}...")
        return lyrics
    
    def generate_music(self, style_prompt, duration=60):
        """步驟 3：生成背景音樂"""
        print("🎶 正在生成背景音樂...")
        response = requests.post(
            self.music_api,
            json={
                "prompt": style_prompt,
                "duration": duration,
                "output": f"{self.output_dir}/background.wav"
            }
        )
        print("✅ 背景音樂生成完成")
    
    def generate_narration(self, text):
        """步驟 4：生成旁白"""
        print("🗣️ 正在生成旁白...")
        response = requests.post(
            self.tts_api,
            json={
                "text": text,
                "speaker": "zh-CN-YunyangNeural",
                "output": f"{self.output_dir}/narration.wav"
            }
        )
        print("✅ 旁白生成完成")
    
    def mix_audio(self, background, narration, output):
        """步驟 5：混音"""
        print("🎧 正在混音...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", background,
            "-i", narration,
            "-filter_complex",
            "[0:a]volume=0.3[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first",
            output
        ])
        print(f"✅ 混音完成: {output}")

# 使用範例
workflow = AudioWorkflow()

# 1. 分離人聲
workflow.separate_vocals("/app/audio/original-song.mp3")

# 2. 生成新的背景音樂
workflow.generate_music(
    "溫暖的鋼琴曲，帶有希望的氛圍",
    duration=90
)

# 3. 生成旁白
workflow.generate_narration(
    "這是一首關於希望和夢想的歌曲，"
    "讓我們一起聆聽這段美妙的旋律。"
)

# 4. 混音
workflow.mix_audio(
    f"{workflow.output_dir}/background.wav",
    f"{workflow.output_dir}/narration.wav",
    f"{workflow.output_dir}/final-mix.wav"
)
```

---

## 14-5 常見問題與疑難排解

### 14-5-1 音訊格式不相容

**問題**：TTS 或 STT 模型不接受你的音訊格式。

**解決方案**：轉換為標準 WAV 格式：

```bash
# 轉換為 16kHz 單聲道 PCM WAV（STT 標準格式）
ffmpeg -i input.xxx \
  -ar 16000 \
  -ac 1 \
  -c:a pcm_s16le \
  output.wav

# 轉換為 44.1kHz 立體聲 WAV（TTS 輸出格式）
ffmpeg -i input.xxx \
  -ar 44100 \
  -ac 2 \
  -c:a pcm_s16le \
  output.wav
```

### 14-5-2 Qwen3-TTS 中文語音品質不佳

**可能原因和解決方案**：

| 問題 | 原因 | 解決方案 |
|------|------|---------|
| 發音不自然 | 語速太快 | 降低 `speed` 到 0.8-0.9 |
| 語調奇怪 | 說話人不適合 | 嘗試不同的 `speaker` |
| 有雜訊 | 模型載入問題 | 重新啟動 Docker 容器 |
| 中文斷句錯誤 | 標點符號問題 | 在適當位置添加句號或逗號 |

```bash
# 調整語速和說話人
curl http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "請慢慢說，這樣比較清楚。",
    "speaker": "zh-CN-YunyangNeural",
    "speed": 0.85,
    "output": "/app/output/slow.wav"
  }'
```

### 14-5-3 faster-whisper 辨識結果有錯字

**排查步驟**：

1. **確認語言設定**：
```bash
# 明確指定中文
-F "language=zh"
```

2. **嘗試更大的模型**：
```bash
# 從 medium 升級到 large-v3
-F "model=large-v3"
```

3. **檢查音訊品質**：
```bash
# 查看音訊資訊
ffprobe input.wav

# 確保：
# - 取樣率 >= 16000 Hz
# - 無明顯背景雜訊
# - 音量適中（不過小也不爆音）
```

4. **使用 VAD 過濾靜音**：
```bash
-F "vad_filter=true"
```

5. **添加初始提示詞**：
```bash
-F "initial_prompt=以下是中文會議記錄，包含專業術語"
```

### 14-5-4 ACE-Step 生成的歌曲品質不穩定

| 問題 | 解決方案 |
|------|---------|
| 音樂不連貫 | 增加 `steps`（50 → 100） |
| 不遵循提示詞 | 提高 `guidance_scale`（7 → 9） |
| 每次結果差異太大 | 固定 `seed` 值 |
| 人聲不清楚 | 使用更具體的歌詞格式，添加 `[Verse]`、`[Chorus]` 標記 |
| 背景雜訊 | 增加 `cfg_rescale`（0.5 → 0.7） |
| 生成時間太長 | 減少 `duration` 或降低 `steps` |

### 14-5-5 Docker 容器佔用太多磁碟空間

```bash
# 查看磁碟使用情況
docker system df

# 查看各容器的詳細使用
docker ps -s

# 清理未使用的資源
docker system prune -a

# 只清理停止的容器（保留映像檔）
docker container prune

# 只清理懸空的映像檔
docker image prune
```

::: warning ⚠️ 注意
`docker system prune -a` 會刪除所有未使用的映像檔，下次啟動容器時需要重新下載。建議定期清理，但不要在高頻率使用期間執行。
:::

### 14-5-6 Demucs 分離效果不佳

| 問題 | 解決方案 |
|------|---------|
| 人聲和伴奏分離不乾淨 | 使用 `htdemucs` 模型（預設） |
| 處理速度太慢 | 添加 `--device cuda` 使用 GPU |
| 輸出有爆音 | 確保輸入音訊沒有 clipping |
| 某些樂器分類錯誤 | 嘗試 `--all` 四軌分離 |

---

## 14-6 本章小結

::: success ✅ 你現在知道了
- Qwen3-TTS 可以做高品質的中文語音合成和聲音複製，支援多情感
- faster-whisper 是高效的語音辨識工具，large-v3 模型中文辨識率達 90%
- ACE-Step 1.5 能生成純音樂和中文歌曲，支援歌詞段落標記
- Demucs 可以分離人聲和伴奏，支援四軌分離
- ffmpeg 是串接所有工具的核心，掌握格式轉換、剪輯、混音
- 完整的工作流程：分離 → 轉寫 → 生成 → 混音，可以完全自動化
:::

::: tip 🚀 第四篇完結！
恭喜！你已經完成了「多媒體 AI 生成」篇。現在你可以用 AI 生成圖片、影片、語音和音樂了！

接下來要進入最硬核的部分 — 模型微調與訓練！

👉 [前往第 15 章：LoRA / QLoRA 微調實戰 →](/guide/chapter15/)
:::

::: info 📝 上一章
← [回到第 13 章：圖片與影片生成](/guide/chapter13/)
:::
