# PLAN — Dự đoán mảng cháy lan tiếp theo (Fire Spread Direction Prediction)
### Phase tiếp theo của: UAV Wildfire Early Detection System (YOLOv8s-P2)

> **Mục tiêu:** Từ output detection hiện tại (`Early_Fire`, `Early_Smoke`), phát triển module dự đoán **hướng và vùng cháy lan tiếp theo**, trực quan hóa bằng **mũi tên hướng lan + vùng cone/polygon dự đoán** vẽ trực tiếp lên ảnh/video frame, dựa trên: hướng khói (smoke plume motion), hướng gió (wind data), và hình dạng vùng cháy.
>
> **Thời gian:** 4 tuần. **Vai trò:** 1 AI Engineer (chính) + tận dụng hạ tầng sẵn có (Kaggle GPU, Cloud Run, Firebase).

---

## 1. Kiến trúc kỹ thuật đề xuất (Technical Approach)

Pipeline mở rộng, xếp chồng lên hệ thống hiện tại:

```
Video frame (UAV)
   │
   ├─► [A] YOLOv8s-P2 (đã có) ──► bbox Early_Fire / Early_Smoke
   │
   ├─► [B] Segmentation module ──► mask khói & lửa (pixel-level)
   │        (YOLOv8s-seg / YOLO11s-seg fine-tune, hoặc SAM2 prompt bằng bbox từ [A])
   │
   ├─► [C] Smoke Motion Estimator ──► vector chuyển động khói
   │        Optical flow (Farneback baseline → RAFT-small nếu đủ tài nguyên)
   │        chỉ tính trong smoke mask, lấy dominant direction (weighted mean + RANSAC lọc nhiễu)
   │
   ├─► [D] Wind Data Fusion ──► vector gió (tốc độ + hướng)
   │        Nguồn: Open-Meteo/OpenWeatherMap API theo GPS của UAV,
   │        hoặc metadata/anemometer nếu có; chiếu về mặt phẳng ảnh qua heading UAV
   │
   ├─► [E] Spread Direction Estimator
   │        v_spread = w1·v_smoke + w2·v_wind + w3·v_frontier
   │        - v_frontier: hướng phát triển biên vùng cháy giữa các frame (mask t vs t-Δ)
   │        - Làm mượt theo thời gian bằng Kalman/EMA filter
   │        - (Stretch) mô hình elip Huygens đơn giản hóa từ Rothermel:
   │          tốc độ lan tỉ lệ thuận wind speed, eccentricity elip theo gió
   │
   └─► [F] Visualization & Prediction Overlay
            - Mũi tên hướng lan (gốc tại trọng tâm fire mask, độ dài ∝ tốc độ ước lượng)
            - Cone/sector dự đoán (góc mở ∝ độ bất định của hướng)
            - Polygon vùng cháy dự kiến tại t+30s / t+60s (elliptical growth)
            - Vẽ bằng OpenCV lên frame; xuất video annotated + JSON kết quả
```

**Nguyên tắc thiết kế:** module [C]–[F] hoạt động được ở chế độ degraded — nếu không có wind API thì chỉ dùng smoke motion; nếu chỉ có 1 frame (ảnh tĩnh) thì suy hướng từ hình dạng plume (trục chính của smoke mask, đầu plume nghiêng về hướng gió).

---

## 2. Tiêu chí Model

| Hạng mục | Lựa chọn chính | Phương án dự phòng | Lý do |
|---|---|---|---|
| Detection | YOLOv8s-P2 (giữ nguyên, đã đạt mAP@50 ≈ 98%) | — | Tái sử dụng, không train lại |
| Segmentation khói/lửa | YOLOv8s-seg hoặc YOLO11s-seg fine-tune 2 class | SAM2 (prompt = bbox từ detector, không cần train) | Cần mask để tính optical flow đúng vùng khói; YOLO-seg nhẹ, cùng hệ sinh thái Ultralytics, export ONNX dễ |
| Motion estimation | Farneback dense optical flow (OpenCV, CPU-friendly) | RAFT-small (chính xác hơn, cần GPU/ONNX) | Farneback đủ cho dominant direction, chạy realtime trên Cloud Run CPU |
| Spread model | Rule-based fusion (smoke + wind + frontier) + Kalman smoothing | Elliptical Huygens model; (nghiên cứu thêm) ConvLSTM/U-Net dự đoán mask t+Δ | Rule-based khả thi trong 4 tuần, giải thích được (explainable), phù hợp demo |
| Deployment | ONNX Runtime (seg model) + OpenCV, tích hợp Flask backend hiện có | — | Đồng nhất stack hiện tại |

**Ràng buộc hiệu năng mục tiêu:**
- Video pipeline ≥ 5 FPS trên CPU Cloud Run (xử lý mỗi N frame, interpolate giữa các frame).
- Segmentation: mask mAP@50 ≥ 70% trên tập val khói/lửa.
- Hướng lan: sai số góc trung bình (Mean Angular Error) ≤ 30°, accuracy trong ±45° ≥ 80%.
- Vùng dự đoán: IoU giữa polygon dự đoán t+Δ và vùng cháy thực tế tại t+Δ ≥ 0.4 (đánh giá trên FLAME 3 NADIR thermal sequence).

---

## 3. Datasets (Train / Test)

| Dataset | Dùng cho | Vai trò | Ghi chú |
|---|---|---|---|
| **EFSA dataset hiện tại** (13,862 ảnh, 2 class) | Train/val seg (chuyển bbox → mask giả bằng SAM2 auto-labeling) | Train | Tận dụng lại, tự sinh mask bằng SAM2 + QC thủ công ~500 ảnh |
| **FLAME 2** (RGB-T video UAV, prescribed burn) | Train + Test motion/spread | Train/Test | Có video liên tục → ground truth chuyển động khói; nhãn Fire/Smoke |
| **FLAME 3** (RGB + radiometric thermal, **NADIR thermal time-series** chụp mỗi 3–5s) | **Test chính cho spread prediction** | Test | NADIR thermal cho phép đo fire progression thực tế → ground truth hướng & vùng lan |
| **SmokeSeg** (6,144 ảnh thực, mask pixel-wise, thiên về early smoke) | Train segmentation | Train | Khớp đúng bài toán early smoke của dự án |
| **SMOKE5K** (1.36K real + 4K synthetic, mask) | Train segmentation (bổ sung) | Train | Tăng đa dạng |
| **D-Fire** (~21K ảnh fire/smoke bbox) | Augment detection/seg | Train (optional) | Nếu cần thêm dữ liệu lửa |
| **HPWREN FIgLib** | Test smoke direction ảnh tĩnh | Test (optional) | Camera cố định, chuỗi ảnh theo thời gian |
| **Wind data**: Open-Meteo API (miễn phí) / OpenWeatherMap | Runtime input | — | Theo tọa độ GPS + timestamp của video |
| **Tự quay/simulation** (nếu cần): video khói từ Unreal/Blender hoặc video YouTube prescribed burn | Val định tính | Test | Bổ sung ca khó: gió đổi hướng, nhiều đám cháy |

**Ground truth cho hướng lan (annotation protocol — làm ở Tuần 1):**
- Từ FLAME 2/3 sequence: lấy fire mask tại t và t+Δ, hướng ground truth = vector từ trọng tâm mask(t) → trọng tâm phần mask mới xuất hiện tại t+Δ.
- Annotate thủ công ~200 cặp frame làm tập test chuẩn (mỗi cặp: góc hướng lan + polygon vùng mới cháy).

---

## 4. Kế hoạch 4 tuần (Timeline & Deliverables)

### 🗓 Tuần 1 — Nghiên cứu, dữ liệu & baseline (Data + Prototype)
| Ngày | Công việc |
|---|---|
| 1–2 | Đọc paper trọng tâm (xem `resources.md`); chốt kiến trúc [A]–[F]; dựng repo branch `feature/spread-prediction` |
| 2–3 | Tải & chuẩn hóa FLAME 2, FLAME 3 (NADIR set), SmokeSeg, SMOKE5K về format thống nhất; viết script trích frame pairs từ video |
| 3–4 | Auto-label mask cho EFSA dataset bằng SAM2 (prompt bbox), QC 500 ảnh mẫu |
| 4–5 | **Baseline prototype**: Farneback optical flow trong smoke bbox (chưa cần mask) → vẽ mũi tên dominant direction lên video FLAME 2 |
| 5 | Xây annotation protocol + annotate 100/200 cặp frame test; định nghĩa metrics (MAE góc, IoU vùng lan) |
| **Deliverable** | Dataset đã chuẩn hóa; demo video baseline có mũi tên hướng khói; tài liệu metric & protocol |

### 🗓 Tuần 2 — Train segmentation & motion module
| Ngày | Công việc |
|---|---|
| 1–3 | Train YOLOv8s-seg / YOLO11s-seg trên EFSA-mask + SmokeSeg + SMOKE5K (Kaggle 2× T4, imgsz 1024); target mask mAP@50 ≥ 70% |
| 3–4 | Tích hợp seg mask vào optical flow: chỉ tính flow trong smoke mask; lọc nhiễu (magnitude threshold + RANSAC); tính dominant vector + độ bất định (circular variance) |
| 4–5 | Module wind fusion: gọi Open-Meteo API theo GPS/timestamp, chiếu vector gió về mặt phẳng ảnh (giả định nadir/heading từ metadata); công thức fusion w1/w2/w3, tune trên tập val |
| 5 | Kalman/EMA smoothing hướng lan theo thời gian; xử lý ảnh tĩnh (suy hướng từ trục chính plume) |
| **Deliverable** | Checkpoint seg model + báo cáo metric; module `spread_estimator.py` chạy end-to-end trên video, xuất vector hướng mỗi giây |

### 🗓 Tuần 3 — Spread prediction, visualization & tích hợp backend
| Ngày | Công việc |
|---|---|
| 1–2 | Elliptical growth model: từ v_spread + wind speed → polygon vùng cháy dự kiến t+30s/t+60s (elip Huygens đơn giản, tham số fit trên FLAME 3 NADIR) |
| 2–3 | Visualization layer (OpenCV): mũi tên hướng lan, cone bất định, polygon dự đoán (màu + độ trong suốt theo xác suất), legend & timestamp |
| 3–4 | Tích hợp vào Flask backend: endpoint mới `POST /api/predict/spread` (video/ảnh) trả về video annotated + JSON {direction_deg, speed_est, polygon_t30, polygon_t60, confidence}; export seg model sang ONNX |
| 4–5 | Frontend: trang mới hiển thị video kết quả + la bàn hướng lan + thông số gió; tối ưu tốc độ (frame skipping, resize) |
| **Deliverable** | API `/api/predict/spread` hoạt động local; demo frontend; pipeline ≥ 5 FPS CPU |

### 🗓 Tuần 4 — Đánh giá, tối ưu, deploy & tài liệu
| Ngày | Công việc |
|---|---|
| 1–2 | Đánh giá định lượng trên tập test (200 cặp frame FLAME 2/3): MAE góc, accuracy ±45°, IoU vùng dự đoán; ablation: smoke-only vs wind-only vs fusion |
| 2–3 | Fix lỗi từ đánh giá; tune trọng số fusion & tham số elip; test ca khó (gió đổi hướng, nhiều nguồn khói, khói mờ) |
| 3–4 | Deploy: backend Cloud Run (Docker update), frontend Firebase; load test video 1080p |
| 4–5 | Viết README/report: kiến trúc, kết quả, hạn chế, hướng phát triển (ConvLSTM dự đoán mask, multi-UAV, thermal); quay video demo |
| **Deliverable** | Hệ thống deploy hoàn chỉnh; báo cáo đánh giá; video demo; tài liệu kỹ thuật |

---

## 5. Rủi ro & phương án giảm thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| Optical flow nhiễu với khói mờ/trong suốt | Cao | Chỉ tính flow trong mask có confidence cao; temporal smoothing; fallback về trục chính plume |
| Không có GPS/heading trong video test → không chiếu được vector gió | Trung bình | Cho phép nhập hướng gió thủ công trên UI; chế độ smoke-only |
| FLAME 3 full set cần request quyền truy cập | Trung bình | Dùng subset CV public trên Kaggle/IEEE DataPort; bắt đầu bằng FLAME 2 |
| SAM2 auto-label mask kém với khói loãng | Trung bình | Ưu tiên SmokeSeg (mask người gán, thiên early smoke) làm nguồn chính |
| Cloud Run CPU không đủ cho RAFT | Thấp | Chốt Farneback làm mặc định; RAFT chỉ là chế độ "high accuracy" offline |
| 4 tuần không đủ cho learned spread model (ConvLSTM) | Chấp nhận | Scope chính là rule-based + elliptical model; learned model để Phase 3 |

---

## 6. Definition of Done
- [ ] Video/ảnh đầu vào → output có mũi tên hướng lan + polygon vùng cháy dự kiến.
- [ ] MAE hướng ≤ 30°, IoU vùng dự đoán ≥ 0.4 trên tập test FLAME.
- [ ] API + UI deploy production (Cloud Run + Firebase), ≥ 5 FPS CPU.
- [ ] Báo cáo đánh giá + ablation study + video demo.
