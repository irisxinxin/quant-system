# Kova 指标校准截图归档

每张 Kova 直播截图的原图存这里，按日期分目录、按 ticker 命名：

```
kova_screenshots/
  2026-06-05/ARM.png  MU.png  QQQ.png  SPY.png  SNDK.png
  2026-06-10/OSCR.png CRDO.png ... (27张)
```

## 为什么既存原图又存数据
- `kova_calibration.csv` 抽取的是面板标签（Score/健康/Reduce）+ 站位/动量等结构化维度
- 但原图保留了**全部视觉维度**（六色量柱逐根颜色、动能震荡器形态、画的趋势线、相邻票等），
  以防之后发现需要新维度时无图可查
- CSV 的 `screenshot` 列指向对应原图相对路径

## 命名规范
- 文件名 = `<TICKER>.png`（大写，与 CSV ticker 列一致）
- 同一票同日多张 → `<TICKER>_2.png`
- 历史回看截图（图表滚到过去某日）→ `<TICKER>@<YYYY-MM-DD>.png`

## ⚠ 注意：粘贴进对话的图无法被程序自动保存
Claude 收到的是视觉输入、没有文件句柄。需用户手动把原图拖进对应日期目录，
或告知 Claude 原图所在路径由 Claude 重命名归档。
