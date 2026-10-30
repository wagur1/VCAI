# Báo Cáo Đánh Giá Object Detection (COCO 2017) Trên V34

**Tập dữ liệu:** COCO 2017 (256 ảnh TEST ngẫu nhiên, phân vùng hash cố định)
**Mô hình đánh giá:** Faster R-CNN ResNet-50 FPN (Torchvision official COCO_V1)
**Codec kiểm tra:** H.264 (x264) và H.265 (x265) | Preset: medium
**Các mức QP:** 30, 35, 40, 45

## 1. Bảng Tổng Hợp BD-Rate Trên COCO

| Codec | Nhánh (Arm) | BD-rate (%) | PCHIP BD-rate (%) | BD-quality (mAP pp) | Trạng thái |
|---|---|---:|---:|---:|---:|
| h264 | `calibrated_303401` | -3.71% | -4.28% | +27.13 | Tiết kiệm bitrate |
| h264 | `calibrated_303402` | 11.06% | -9.92% | +60.37 | Không tiết kiệm |
| h264 | `calibrated_303403` | 3.19% | -6.27% | +36.11 | Không tiết kiệm |
| h264 | `fixed` | 1.81% | -5.25% | +22.04 | Không tiết kiệm |
| h264 | `forced_303401` | 3.93% | -11.66% | +75.80 | Không tiết kiệm |
| h264 | `forced_303402` | -4.95% | -9.84% | +67.82 | Tiết kiệm bitrate |
| h264 | `forced_303403` | -0.75% | -10.59% | +67.27 | Tiết kiệm bitrate |
| h264 | `forced_frozen_303401` | 1.81% | -5.25% | +22.04 | Không tiết kiệm |
| h264 | `selected_303401` | -10.19% | -16.84% | +102.25 | Tiết kiệm bitrate |
| h264 | `selected_303402` | -10.82% | -16.37% | +99.23 | Tiết kiệm bitrate |
| h264 | `selected_303403` | -7.52% | -15.83% | +97.20 | Tiết kiệm bitrate |
| h265 | `calibrated_303401` | -10.87% | -4.28% | +34.23 | Tiết kiệm bitrate |
| h265 | `calibrated_303402` | -11.22% | -7.67% | +66.32 | Tiết kiệm bitrate |
| h265 | `calibrated_303403` | -12.05% | -3.90% | +30.76 | Tiết kiệm bitrate |
| h265 | `fixed` | -4.52% | -0.28% | +3.78 | Tiết kiệm bitrate |
| h265 | `forced_303401` | -15.96% | -8.49% | +69.03 | Tiết kiệm bitrate |
| h265 | `forced_303402` | -21.92% | -5.93% | +42.40 | Tiết kiệm bitrate |
| h265 | `forced_303403` | -15.43% | -6.58% | +51.90 | Tiết kiệm bitrate |
| h265 | `forced_frozen_303401` | -4.52% | -0.28% | +3.78 | Tiết kiệm bitrate |
| h265 | `selected_303401` | -4.73% | -15.74% | +131.62 | Tiết kiệm bitrate |
| h265 | `selected_303402` | 21.82% | -17.24% | +142.63 | Không tiết kiệm |
| h265 | `selected_303403` | -1.80% | -15.53% | +130.83 | Tiết kiệm bitrate |

## 2. Chi Tiết Rate (bpp) và mAP@[.50:.95] Theo Từng QP

### Codec: H264

| QP | Metric | Anchor | Fixed (Action 11) | Selected 303401 | Δ mAP (303401) | Δ Rate % (303401) |
|---:|---|---:|---:|---:|---:|---:|
| 30 | mAP@[.5:.95] | 0.2194 | 0.2086 | 0.2148 | -0.47 pp | -18.26% |
| 30 | Rate (bpp) | 0.7936 | 0.6730 | 0.6487 | - | - |
| 35 | mAP@[.5:.95] | 0.2052 | 0.2004 | 0.2081 | +0.29 pp | -16.89% |
| 35 | Rate (bpp) | 0.4829 | 0.4117 | 0.4013 | - | - |
| 40 | mAP@[.5:.95] | 0.1817 | 0.1774 | 0.1836 | +0.20 pp | -13.07% |
| 40 | Rate (bpp) | 0.2954 | 0.2588 | 0.2568 | - | - |
| 45 | mAP@[.5:.95] | 0.1381 | 0.1363 | 0.1470 | +0.89 pp | -8.59% |
| 45 | Rate (bpp) | 0.1834 | 0.1677 | 0.1676 | - | - |

### Codec: H265

| QP | Metric | Anchor | Fixed (Action 11) | Selected 303401 | Δ mAP (303401) | Δ Rate % (303401) |
|---:|---|---:|---:|---:|---:|---:|
| 30 | mAP@[.5:.95] | 0.2158 | 0.2118 | 0.2167 | +0.09 pp | -16.86% |
| 30 | Rate (bpp) | 0.8262 | 0.7130 | 0.6869 | - | - |
| 35 | mAP@[.5:.95] | 0.2078 | 0.2020 | 0.2132 | +0.54 pp | -13.69% |
| 35 | Rate (bpp) | 0.5623 | 0.4946 | 0.4854 | - | - |
| 40 | mAP@[.5:.95] | 0.1858 | 0.1773 | 0.1905 | +0.47 pp | -9.47% |
| 40 | Rate (bpp) | 0.3988 | 0.3634 | 0.3610 | - | - |
| 45 | mAP@[.5:.95] | 0.1397 | 0.1280 | 0.1541 | +1.43 pp | -5.07% |
| 45 | Rate (bpp) | 0.2990 | 0.2839 | 0.2839 | - | - |

## 3. Nhận Xét & Kết Luận
- **Hiệu quả nén OD:** Xem xét BD-rate âm thể hiện mức giảm bitrate ở cùng mAP.
- **Độ chính xác Faster R-CNN:** Bounding boxes được decode từ dòng video thực, giải mã ngược về tọa độ gốc và đánh giá qua `COCOeval` chính thức.
- **Hoàn toàn độc lập:** Quá trình huấn luyện controller chỉ dùng tập train/cal, không chạm vào nhãn ground-truth của tập DEV/TEST.