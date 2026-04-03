# 常見問題與故障排除（FAQ）

::: tip 🔍 找不到答案？
如果以下 FAQ 沒有解決你的問題，建議：
1. 回到相關章節重新閱讀
2. 用 Claude Code 協助除錯（告訴它錯誤訊息）
3. 查看 NVIDIA 官方論壇
:::

---

## GPU 與驅動相關

### Q：nvidia-smi 顯示找不到 GPU？

```bash
# 檢查驅動是否載入
lsmod | grep nvidia

# 如果沒有輸出，重新載入驅動
sudo modprobe nvidia

# 重試
nvidia-smi
```

如果還是找不到，嘗試重新開機。

### Q：GPU 溫度過高怎麼辦？

正常運作溫度是 30-70°C。如果超過 80°C：
1. 確認 DGX Spark 周圍有足夠的通風空間
2. 清理散熱孔的灰塵
3. 降低同時運行的模型數量

---

## Docker 相關

### Q：Docker 指令需要 sudo 才能執行？

```bash
# 把使用者加入 docker 群組
sudo usermod -aG docker $USER

# 重新登入（或執行）
newgrp docker
```

### Q：Docker 容器佔用太多磁碟空間？

```bash
# 查看磁碟使用狀況
docker system df

# 清理未使用的映像檔和容器
docker system prune -a

# 清理懸掛的映像檔
docker image prune
```

---

## Ollama 相關

### Q：Ollama 服務無法啟動？

```bash
# 檢查服務狀態
systemctl status ollama

# 查看日誌
journalctl -u ollama -f

# 重新啟動
sudo systemctl restart ollama
```

### Q：模型下載很慢？

Ollama 預設從官方伺服器下載。如果網路慢，可以設定代理：

```bash
# 編輯 Ollama 服務設定
sudo systemctl edit ollama

# 加入
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="HTTPS_PROXY=http://你的代理伺服器:port"
```

---

## 網路與遠端存取

### Q：SSH 無法連線？

1. 確認 DGX Spark 的 IP 正確
2. 確認 SSH 服務已啟動：`sudo systemctl status ssh`
3. 確認防火牆允許 port 22：`sudo ufw status`
4. 確認兩台裝置在同一個網路中

### Q：Tailscale 連線中斷？

```bash
# 檢查 Tailscale 狀態
tailscale status

# 重新連線
sudo tailscale down
sudo tailscale up
```

---

## 微調相關

### Q：訓練時出現 CUDA Out of Memory？

1. 降低 batch size
2. 使用 QLoRA（4-bit 量化）代替 LoRA
3. 減少上下文長度（max_seq_length）
4. 關閉其他占用記憶體的程式

### Q：訓練 Loss 不下降？

1. 檢查學習率（learning rate）是否太高
2. 確認訓練資料格式正確
3. 增加訓練步數
4. 檢查模型是否正確載入

---

## 多機互連

### Q：NCCL 測試失敗？

1. 確認兩台 DGX Spark 之間網路連線正常
2. 確認防火牆允許 NCCL 使用的 port
3. 檢查 NCCL 版本是否一致
4. 用 `nccl-tests` 進行詳細診斷

---

::: tip 🚀 回到導覽
- [首頁](/)
- [第 1 章：DGX Spark 硬體總覽](/guide/chapter1/)
- [推薦模型清單](/guide/models)
:::
