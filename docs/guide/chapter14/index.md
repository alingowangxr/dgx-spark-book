# 第 14 章：音訊、語音與音樂 AI

::: tip 🎯 本章你將學到什麼
- 部署 Qwen3-TTS 中文語音合成
- 部署 faster-whisper 語音辨識
- 部署 ACE-Step 1.5 AI 音樂生成
- ffmpeg 音訊處理工具鏈
- Demucs 人聲分離
:::

::: warning ⏱️ 預計閱讀時間
約 20 分鐘。
:::

---

## 14-1 語音合成（Text-to-Speech）

### 14-1-1 TTS 模型比較

| 模型 | 語言 | 記憶體需求 | 音質 | 速度 |
|------|------|-----------|------|------|
| **Qwen3-TTS** | 中文 + 英文 | ~2 GB | 高 | 快 |
| CosyVoice | 中文 | ~4 GB | 高 | 中等 |
| XTTS v2 | 多語言 | ~4 GB | 中高 | 中等 |
| Piper | 多語言 | ~0.5 GB | 中 | 超快 |

### 14-1-2 Qwen3-TTS 的架構

Qwen3-TTS 基於端到端的神聲網路，直接從文字生成聲波，不需要中間的音素轉換。

好處：
- 中文發音自然
- 支援多情感
- 速度快

### 14-1-3 部署 Qwen3-TTS

```bash
# 用 Docker 部署
docker run -d \
  --name qwen3-tts \
  --gpus all \
  --network host \
  -v ~/tts-output:/app/output \
  ghcr.io/community/qwen3-tts:latest
```

測試：

```bash
curl http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，歡迎使用 DGX Spark 語音合成系統。",
    "output": "/app/output/test.wav"
  }'
```

### 14-1-4 中文語音合成實測

```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "塞爆 128G GPU 記憶體，DGX Spark 全面玩透！",
        "speaker": "zh-CN-YunxiNeural",
        "speed": 1.0,
        "output": "/app/output/dgx-spark.wav"
    }
)

print(response.json())
```

### 14-1-5 語音複製

Qwen3-TTS 支援語音複製（Voice Cloning）：

```bash
curl http://localhost:8000/tts-clone \
  -H "Content-Type: application/json" \
  -d '{
    "text": "這段話是用你的聲音說的。",
    "reference_audio": "/app/reference/my-voice.wav",
    "output": "/app/output/cloned.wav"
  }'
```

只需要 10-30 秒的參考音訊，就能複製聲音。

---

## 14-2 語音辨識（Speech-to-Text）

### 14-2-1 Whisper 模型比較

| 模型 | 大小 | 記憶體需求 | 辨識準確率 | 速度 |
|------|------|-----------|-----------|------|
| tiny | 39M | ~1 GB | 低 | 超快 |
| base | 74M | ~1 GB | 中低 | 快 |
| small | 244M | ~2 GB | 中 | 中等 |
| medium | 769M | ~3 GB | 高 | 中等 |
| **large** | 1550M | ~5 GB | **最高** | 慢 |

### 14-2-2 部署 faster-whisper

faster-whisper 是 Whisper 的加速版本，使用 CTranslate2 引擎。

```bash
# 用 Docker 部署
docker run -d \
  --name faster-whisper \
  --gpus all \
  --network host \
  -v ~/audio-files:/app/audio \
  ghcr.io/community/faster-whisper:latest
```

### 14-2-3 語音辨識實測

```bash
curl http://localhost:8000/transcribe \
  -F "audio=@/app/audio/meeting.wav" \
  -F "language=zh" \
  -F "model=large"
```

### 14-2-4 字幕生成

```bash
curl http://localhost:8000/subtitle \
  -F "audio=@/app/audio/podcast.mp3" \
  -F "language=zh" \
  -F "format=srt" \
  -o /app/audio/podcast.srt
```

### 14-2-5 即時語音轉文字

```bash
# 從麥克風即時轉寫
docker run -it \
  --gpus all \
  --network host \
  --device /dev/snd \
  ghcr.io/community/faster-whisper:latest \
  python live_transcribe.py --language zh
```

---

## 14-3 AI 音樂生成

### 14-3-1 音樂生成模型比較

| 模型 | 記憶體需求 | 生成時長 | 品質 |
|------|-----------|---------|------|
| **ACE-Step 1.5** | ~8 GB | 30-120 秒 | 高 |
| MusicGen | ~6 GB | 10-30 秒 | 中 |
| Riffusion | ~4 GB | 10-15 秒 | 中低 |

### 14-3-2 部署 ACE-Step 1.5

```bash
docker run -d \
  --name ace-step \
  --gpus all \
  --network host \
  -v ~/music-output:/app/output \
  ghcr.io/community/ace-step:1.5
```

### 14-3-3 生成純音樂

```bash
curl http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "輕鬆的鋼琴曲，適合早晨聆聽",
    "duration": 60,
    "output": "/app/output/morning-piano.wav"
  }'
```

### 14-3-4 生成中文歌曲

```bash
curl http://localhost:7860/generate-song \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "今天天氣真好，適合出去走走",
    "style": "流行音樂",
    "language": "zh",
    "duration": 90,
    "output": "/app/output/song.wav"
  }'
```

### 14-3-5 進階設定

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `guidance_scale` | 提示詞遵循度 | 7-9 |
| `steps` | 生成步數 | 50-100 |
| `seed` | 隨機種子 | 固定值可重現 |
| `cfg_rescale` | CFG 重縮放 | 0.7 |

---

## 14-4 音訊處理工具鏈

### 14-4-1 ffmpeg 基本操作

第 3 章已安裝 ffmpeg，這裡複習常用操作：

```bash
# 格式轉換
ffmpeg -i input.wav -codec:a libmp3lame -qscale:a 2 output.mp3

# 剪輯音訊
ffmpeg -i input.mp3 -ss 00:01:00 -to 00:02:00 -c copy output.mp3

# 合併音訊
ffmpeg -i "concat:file1.mp3|file2.mp3|file3.mp3" -c copy output.mp3

# 調整音量
ffmpeg -i input.mp3 -filter:a "volume=1.5" output.mp3

# 淡入淡出
ffmpeg -i input.mp3 -af "afade=t=in:st=0:d=3,afade=t=out:st=57:d=3" output.mp3
```

### 14-4-2 Demucs 人聲分離

Demucs 可以把歌曲中的人聲和伴奏分離。

```bash
# 用 uv 安裝
uv pip install demucs

# 分離人聲
demucs --two-stems=vocals song.mp3 -o ~/separated/
```

輸出：
```
~/separated/
├── song/
│   ├── vocals.wav    # 人聲
│   └── no_vocals.wav # 伴奏
```

### 14-4-3 人聲分離實測

```bash
# 分離所有軌道（人聲、鼓、貝斯、其他）
demucs --all song.mp3 -o ~/separated/

# 輸出
~/separated/
├── song/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
```

### 14-4-4 工作流程串接

一個完整的音訊處理工作流程：

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
```

---

## 14-5 常見問題與疑難排解

### 14-5-1 音訊格式不相容

```bash
# 轉換為標準 WAV 格式
ffmpeg -i input.xxx -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### 14-5-2 Qwen3-TTS 中文語音品質不佳

嘗試調整參數：
```bash
# 降低語速
"speed": 0.9

# 換不同的說話人
"speaker": "zh-CN-YunyangNeural"
```

### 14-5-3 faster-whisper 辨識結果有錯字

- 確認語言設定正確（`--language zh`）
- 嘗試更大的模型（`medium` → `large`）
- 確保音訊品質良好（無明顯雜訊）

### 14-5-4 ACE-Step 生成的歌曲品質不穩定

- 增加 `steps` 參數（50 → 100）
- 提高 `guidance_scale`（7 → 9）
- 使用更具體的提示詞

### 14-5-5 Docker 容器佔用太多磁碟空間

```bash
# 查看磁碟使用
docker system df

# 清理
docker system prune -a
```

---

## 14-6 本章小結

::: success ✅ 你現在知道了
- Qwen3-TTS 可以做高品質的中文語音合成和聲音複製
- faster-whisper 是高效的語音辨識工具
- ACE-Step 1.5 能生成純音樂和中文歌曲
- Demucs 可以分離人聲和伴奏
- ffmpeg 是串接所有工具的核心
:::

::: tip 🚀 第四篇完結！
恭喜！你已經完成了「多媒體 AI 生成」篇。現在你可以用 AI 生成圖片、影片、語音和音樂了！

接下來要進入最硬核的部分 — 模型微調與訓練！

👉 [前往第 15 章：LoRA / QLoRA 微調實戰 →](/guide/chapter15/)
:::

::: info 📝 上一章
← [回到第 13 章：圖片與影片生成](/guide/chapter13/)
:::
