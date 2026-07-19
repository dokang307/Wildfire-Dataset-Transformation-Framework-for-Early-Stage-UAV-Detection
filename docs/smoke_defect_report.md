# Báo cáo lỗi: Model không phát hiện được khói trên ảnh UAV thực tế

| | |
|---|---|
| **Ngày** | 2026-07-17 |
| **Mức độ** | 🔴 Nghiêm trọng — vô hiệu hoá luận điểm cốt lõi của dự án |
| **Thành phần** | Dataset & huấn luyện (KHÔNG phải backend) |
| **Trạng thái** | 🟢 Đã khắc phục lỗi cốt lõi bằng retrain trên FASDD — còn chờ nghiệm thu OOD (xem mục 9) |
| **Model liên quan** | (lỗi) `train_optimized/last.pt` → (đã sửa) `results/runs/fasdd_train/weights/best.pt` |

---

## 0. Cập nhật trạng thái (2026-07-19)

Đã **retrain YOLOv8s trên dataset FASDD** (`results/runs/fasdd_train/weights/best.pt`, imgsz=960, best epoch 31) và re-export sang `backend/model/best.onnx`. Kết quả trên chính ảnh tái hiện `backend/sample.jpg`:

| | Trước (phase03) | Sau (FASDD) |
|---|---|---|
| Điểm thô kênh smoke | 0.0063 | **0.6647** |
| Box smoke detect được | 0 | **1** (66.5%) |
| Direction | `None` → Undetermined | **14.6°**, method `A+B_conflict` |
| Vẽ mũi tên + ellipse | Không | **Có** |

Metrics val FASDD (best epoch 31): **P=0.773, R=0.675, mAP50=0.769, mAP50-95=0.472**. Thấp hơn 0.826 cũ nhưng **đáng tin** vì không còn rò rỉ.

**Vẫn còn nợ:** các bước **3** (đo per-class + test-split trên split đúng), **8** (cổng nghiệm thu OOD 30–50 ảnh, smoke recall ≥0.7) và **10** (đồng bộ mọi tuyên bố báo cáo) trong mục 7 chưa hoàn tất. Mục 6.5 (rò rỉ W&B API key) vẫn cần xử lý. Phần dưới giữ nguyên làm hồ sơ điều tra gốc.

---

## 1. Tóm tắt điều hành

Khi tải một ảnh cháy rừng aerial thật lên web app, hệ thống phát hiện lửa bình thường nhưng luôn báo **"Spread Direction: Undetermined"** và không vẽ mũi tên hướng gió hay ellipse lan cháy.

Điều tra cho thấy **backend hoàn toàn không có lỗi**. Nguyên nhân nằm ở **dữ liệu huấn luyện**, với **hai khiếm khuyết chồng nhau**:

1. **Thiếu nhãn khói** — ~60% dataset là ảnh mặt đất chỉ được label `fire`, dù trong ảnh có cột khói rõ rệt. Khói không được label bị huấn luyện thành *background*, triệt tiêu lớp `smoke`.
2. **Rò rỉ dữ liệu theo video** — toàn bộ năng lực nhận khói của model chỉ đến từ **5 video nguồn**. Các frame liền kề (cách nhau 1 frame) nằm ở cả train lẫn test.

Hệ quả: model **chưa từng học "khói" như một khái niệm tổng quát** — nó chỉ học thuộc 5 cảnh cụ thể. Chỉ số **smoke mAP@50 = 0.939 đã công bố là không đáng tin cậy.**

Vì Phase B/C/D đều bắt buộc phải có smoke box, **toàn bộ luận điểm "ước lượng hướng lan cháy không cần cảm biến gió" không hoạt động trên ảnh UAV thực tế.**

---

## 2. Hiện tượng

Tải ảnh aerial cháy rừng (1000×666) lên `/api/detect/image` với `confidence = 0.02`:

```
detections: 3
   fire    80.3%  [380, 248, 724, 422]
   fire    65.8%  [776, 170, 869, 220]
   fire    50.0%  [8, 448, 123, 527]

direction_angle      : None
direction_confidence : 0.0
direction_method     : None     ->  UI hiển thị "Undetermined"
```

Ảnh có khối khói lớn chiếm gần nửa khung hình, nhưng **không một box `smoke` nào** được sinh ra.

---

## 3. Điều tra: những giả thuyết đã bị loại trừ

Ghi lại đầy đủ để tránh điều tra lặp lại.

| # | Giả thuyết | Kết luận | Bằng chứng |
|---|---|---|---|
| 1 | Pipeline direction chưa được đấu nối | ❌ Loại | UI hiện "Undetermined" ⇒ backend đã trả về field `direction_*`. Pipeline abstain **đúng logic** khi không có smoke box. |
| 2 | Export ONNX làm hỏng kênh smoke | ❌ Loại | Chạy chính `last.pt` qua PyTorch trên cùng ảnh → **cũng 0 smoke**. Parity check: 3/3 detection trùng khớp, box lệch ≤3px. |
| 3 | Độ phân giải suy luận quá thấp (640 vs train 960) | ❌ Loại | smoke ≈ 0 ở **mọi** độ phân giải: `0.0063` @640, `0.0032` @960, `0.0020` @1280. |
| 4 | Ngưỡng confidence quá cao | ❌ Loại | smoke max = `0.0063`, thấp hơn mọi ngưỡng dùng được. Hạ xuống 0.02 vẫn không ra. |
| 5 | Deploy nhầm checkpoint | ❌ Loại | `last.pt` = 22,541,034 bytes khớp log Kaggle *"stripped ... last.pt, 22.5MB"*; `results.csv` epochs 192→200 khớp *"9 epochs completed"*. Đúng là checkpoint cuối phase03. |
| 6 | Lệch domain (ảnh aerial ngoài phân phối) | ⚠️ **Không phải nguyên nhân chính** | `fig_spread.png` cho thấy tập `dba_vd` **cũng là ảnh aerial cháy rừng cùng domain**. Vấn đề sâu hơn: xem mục 4. |

**Số liệu chốt:** trên ảnh test, kênh `smoke` cho điểm tối đa **0.0003–0.0063** trên toàn bộ ~8.400 anchor, trong khi `fire` đạt **0.80–0.87**. Model không "yếu" ở lớp khói — nó **im lặng hoàn toàn**.

---

## 4. Nguyên nhân gốc

### 4.1 Khiếm khuyết 1 — Thiếu nhãn khói (label conflict)

Kiểm tra trực quan 24 ảnh ground-truth (`results/runs/eval_test/val_batch{0,1,2}_labels.jpg`):

- **24/24 ảnh chỉ có box `fire`** — không một box `smoke` nào.
- Nhiều ảnh có **cột khói đen rõ rệt nhưng không được annotate**: `dba_img__138` (ô tô cháy, khói mù mịt), `dba_8de0caf9`, `dba_img__143`, `dba_img__81`.
- Dataset **không phải ảnh UAV**: ô tô cháy, nhà cháy, tàu hoả, lính cứu hoả, **bếp gas** (`dba_pic__53`), **ngọn lửa bật lửa** (`dba_small__62`), tia lửa cột điện (`dba_u_718066098`).

YOLO coi mọi vùng không có nhãn là *background*. Ảnh có khói nhưng không label khói ⇒ **gradient âm đẩy logit smoke xuống**. Với ~60% dataset như vậy, lớp `smoke` bị triệt tiêu ở mọi ngữ cảnh ngoài phần đã được label.

> ⚠️ Điều này còn làm **sai lệch chính phép đánh giá**: một model detect khói *đúng* trên ảnh `dba_img*` sẽ bị tính là **false positive** (vì không có GT). Con số `background→smoke = 0.15` trong confusion matrix có thể phần lớn là **smoke detect đúng bị chấm sai**.

### 4.2 Khiếm khuyết 2 — Rò rỉ dữ liệu theo video 🔴 (nghiêm trọng hơn)

Phân tích `results/smoke_directions.csv` (140 ảnh test detect được khói):

| Chỉ số | Giá trị |
|---|---|
| Prefix của cả 140 ảnh | **100% `dba_vd`** (không một prefix nào khác) |
| Số **video nguồn** sinh ra 140 ảnh | **5** (id: 1, 3, 4, 5, 6) |
| Khoảng cách frame nhỏ nhất **trong test** | **1 frame** |
| Ví dụ | `dba_vd3000444` & `dba_vd3000445` → theta 158.62° vs 158.89° |

Cấu trúc tên file là `dba_vd<video><frame 6 chữ số>` — tức **frame trích từ video**, lấy dày (~6 frame/lần).

**Nếu split được chia theo frame (rất nhiều khả năng), frame test là bản gần trùng của frame train.** Khi đó:

> **smoke mAP@50 = 0.939 chỉ đo mức độ học thuộc 5 cảnh, không phải khả năng tổng quát hoá.**

**Độ đa dạng thực của lớp smoke ≈ 5 cảnh**, không phải 5.405 mẫu độc lập.

### 4.3 Hai khiếm khuyết giải thích trọn vẹn hiện tượng

`fig_spread.png` chứng minh `dba_vd` là ảnh aerial **cùng domain với ảnh người dùng tải lên**. Vậy nên đây **không phải** lệch domain đơn thuần — mà là model **chưa bao giờ học khái niệm "khói"**, chỉ ghi nhớ 5 cảnh. Ảnh aerial mới = **cảnh thứ 6 chưa từng thấy** → smoke = 0.006 → `Undetermined`.

---

## 5. Mức độ ảnh hưởng

### 5.1 Sản phẩm

| Chỉ số (từ `smoke_directions.csv`) | Giá trị |
|---|---|
| Ảnh test có ước lượng hướng | **140/392 = 35.7%** |
| **Undetermined ngay trên chính test set** | **252/392 = 64.3%** |
| Ảnh aerial mới (ngoài 5 video) | **0%** |
| `A+B_conflict` (A và B lệch >60°, conf trần 0.3) | **56/140 = 40%** |

### 5.2 Tính toàn vẹn của báo cáo khoa học ⚠️

Các tuyên bố hiện tại **không được số liệu hỗ trợ**:

| Tuyên bố | Thực tế |
|---|---|
| smoke mAP@50 = **93.9%** | Không đáng tin — nhiễm rò rỉ video (mục 4.2) |
| Pipeline **4 pha** đã kiểm chứng | **Phase C (optical flow) và Phase D (EMA) CHƯA TỪNG CHẠY.** `VIDEO_PATH = None` → `MODE='IMAGE'` → nhánh else **chỉ gọi `direction_from_geometry`**. `direction_from_flow`, `combine_directions`, `DirectionSmoother` là **code chết**. Bằng chứng: phân bố method không hề có `flow`/`geom+flow`. |
| `fig_spread.png` minh hoạ năng lực hệ thống | Là **top-6 theo conf trên 392 ảnh = 1.5%**, toàn `A+B` conf ≥0.90 → **cherry-pick**, không đại diện |
| Dataset **5.405 ảnh aerial** | Thực tế **~12.300 ảnh** (1419 batch × 8 + 585 + 392), **đa số là ảnh mặt đất** không phải aerial |
| Metrics đo trên model phase03 | Đo trên **`best.pt` của phase02** (xem mục 6.1) |
| Inference **≈133 FPS** | Notebook đo thực **67.9 FPS** (median 80.5) trên T4 |

---

## 6. Các lỗi phụ phát hiện thêm

### 6.1 `resume=True` khiến `BEST` âm thầm trỏ về checkpoint phase02

Log training thật:

```
Resuming training .../phase02-output/runs/train_optimized/weights/last.pt from epoch 192 to 200
9 epochs completed in 1.138 hours.
Optimizer stripped from /kaggle/working/runs/train_optimized/weights/last.pt, 22.5MB
  Dùng best.pt từ run trước: .../phase02-output/runs/train_optimized/weights/best.pt
```

- Ultralytics khi `resume=True` **khôi phục train_args gốc từ checkpoint và bỏ qua `name='train_resumed'`** → ghi đè vào `train_optimized/`.
- ⇒ `train_resumed/weights/best.pt` **không bao giờ tồn tại** → nhánh fallback kích hoạt → **`BEST` = `best.pt` của phase02**.
- ⇒ Toàn bộ eval, FPS benchmark, `fig_spread.png`, artifact W&B đều đo trên **phase02 best.pt**, không phải model phase03.

### 6.2 Phase03 không cải thiện gì

`fitness = 0.1·mAP50 + 0.9·mAP50-95`:

| | mAP50 | mAP50-95 | fitness |
|---|---|---|---|
| phase02 `best.pt` | 0.801 | **0.471** | **0.5040** |
| phase03 `last.pt` (ep 200) | 0.802 | 0.467 | 0.5003 |
| Fitness cao nhất ở ep 192–200 | | | 0.5009 |

0.5009 < 0.5040 ⇒ **không epoch nào phá kỷ lục** ⇒ Ultralytics không bao giờ ghi `best.pt` mới. Đây là lý do tải lại output phase03 bao nhiêu lần cũng chỉ có `last.pt`.

### 6.3 Markdown notebook ghi sai

Notebook viết *"Train dừng ở epoch 94"* và *"Resume từ epoch 94"* — thực tế **resume từ epoch 192**, chỉ chạy **9 epoch**.

### 6.4 Phụ thuộc ẩn giữa các cell

Section *Smoke Direction + Spread Prediction* dùng `INFER_IMGSZ` — biến được định nghĩa ở cell **FPS BENCHMARK**. Bỏ qua cell đó → `NameError`.

### 6.5 🔑 Rò rỉ W&B API key

Cell `W&B SETUP` **hard-code API key** trong source:

```python
WANDB_API_KEY = "wandb_v1_2kncCZet8PIpj80..."   # ⚠️ paste key, xóa trước khi share
```

Comment tự dặn phải xoá nhưng **chưa xoá**. File `wildfire-spread-phase03.ipynb` hiện đang **untracked** — **tuyệt đối không commit trước khi gỡ key**. Khuyến nghị **thu hồi (revoke) key này ngay** và chuyển sang Kaggle Secrets.

---

## 7. Kế hoạch khắc phục step-by-step

> **Backend KHÔNG cần sửa gì.** Pipeline đã đúng và có 18 unit test bảo vệ. Nó sẽ vẽ mũi tên + ellipse ngay khi detector nhìn thấy khói.

### Bước 1 — Chạy audit dữ liệu ⏱️ 15 phút · 🚧 CHẶN mọi bước sau

```bash
# Trên Kaggle notebook (dataset chỉ tồn tại ở đó)
!python audit_labels.py --root /kaggle/input/datasets/dangnguyenminhduy/wildfireuav/merged
```

Script: [`scripts/audit_labels.py`](../scripts/audit_labels.py). Trả lời:
- Subset nào có **0 nhãn smoke** và chiếm bao nhiêu %?
- Có **bao nhiêu video** thật sự? Train/val/test có **trùng video** không?

**Kết quả cần:** verdict `LEAKAGE CONFIRMED` + số cảnh thật.

### Bước 2 — Re-split theo VIDEO, không theo frame ⏱️ 1 giờ

Dùng `GroupShuffleSplit` với `groups = video_id`. **Không một video nào được xuất hiện ở hai split.**

### Bước 3 — Đo lại mAP thật ⏱️ 1 giờ

Đánh giá lại model hiện tại trên split đã sửa.

> ⚠️ **Chuẩn bị tinh thần: smoke mAP sẽ tụt mạnh khỏi 0.939.** Con số mới mới là con số thật, và phải dùng nó trong báo cáo.

### Bước 4 — Bổ sung dữ liệu aerial 🔴 BẮT BUỘC ⏱️ 1–2 ngày

5 cảnh **không thể** dạy được khái niệm "khói". Cần **hàng chục–hàng trăm cảnh khác nhau** — *thêm frame từ 5 video cũ là vô ích*.

Nguồn ứng viên (cần tự kiểm chứng nội dung + license):
- **FLAME / FLAME2** — UAV cháy rừng thật. Dự án **đang chỉ dùng làm negatives → lãng phí nguồn đúng domain nhất**.
- **FASDD** — có subset UAV / remote-sensing với box fire+smoke.
- **FIgLib / HPWREN** — cột khói cháy rừng.
- **D-Fire** — fire+smoke bbox (chủ yếu mặt đất, dùng bổ trợ).

### Bước 5 — Khử trùng lặp frame ⏱️ 30 phút

Hiện lấy ~6 frame/lần là quá dày. Giảm còn **~1–2 frame/giây**, hoặc lọc theo độ tương đồng ảnh.

### Bước 6 — Loại subset 0-smoke khỏi **cả train LẪN val/test** ⏱️ 1 giờ

Phải loại khỏi val/test nữa — nếu không, model detect khói *đúng* sẽ bị chấm **false positive** (xem cảnh báo mục 4.1).

### Bước 7 — Retrain ⏱️ ~6h GPU

- Train mới từ **`yolov8s.pt` (COCO)**, **không** fine-tune từ `last.pt` (đã học "khói = background", rủi ro giữ lại sự triệt tiêu).
- Giữ config phase03: `imgsz=960, batch=8, cos_lr=True, patience=30`; giảm còn ~100–150 epoch.
- **`resume=False`** để `name` được tôn trọng; **luôn `assert BEST.exists()`** thay vì fallback im lặng (mục 6.1).

### Bước 8 — Cổng nghiệm thu OOD ⏱️ 2 giờ · ⭐ QUAN TRỌNG NHẤT

Bài học đắt nhất: **mAP in-distribution đã che giấu hoàn toàn việc model mù khói ngoài đời.**

- Gom **30–50 ảnh aerial cháy rừng thật** (đúng loại `backend/sample.jpg`), tự label `fire` + `smoke`.
- **Tiêu chí nghiệm thu: smoke recall ≥ 0.7 @ IoU 0.5 trên tập OOD** — *không phải* mAP in-distribution.
- Không đạt → quay lại Bước 4, cần thêm cảnh.

### Bước 9 — Guardrail chống tái phát ⏱️ 30 phút

Integration test dùng `backend/sample.jpg` làm fixture: assert detect được smoke **và** `direction_angle is not None`. CI đỏ ngay nếu model tương lai lại mù khói.

### Bước 10 — Sửa lại các tuyên bố trong báo cáo ⏱️ 1 giờ

Cập nhật theo bảng mục 5.2: smoke mAP thật, số ảnh thật, FPS thật, và **hoặc thực sự chạy Phase C/D (set `VIDEO_PATH`), hoặc bỏ tuyên bố pipeline 4 pha đã kiểm chứng**.

---

### Tổng hợp

| Bước | Nội dung | Thời gian | Chặn? |
|---|---|---|---|
| 1 | Audit dữ liệu | 15 phút | ✅ chặn tất cả |
| 2 | Re-split theo video | 1 giờ | |
| 3 | Đo lại mAP thật | 1 giờ | |
| 4 | **Bổ sung dữ liệu aerial** | 1–2 ngày | ✅ chặn Bước 7 |
| 5 | Khử trùng lặp frame | 30 phút | |
| 6 | Loại subset 0-smoke | 1 giờ | |
| 7 | Retrain | ~6h GPU | |
| 8 | **Cổng OOD** | 2 giờ | ✅ nghiệm thu |
| 9 | Guardrail | 30 phút | |
| 10 | Sửa tuyên bố báo cáo | 1 giờ | |

---

## 8. Phụ lục — Cách tái hiện

```bash
cd backend

# 1. Điểm thô của kênh smoke ở nhiều độ phân giải
python - <<'PY'
import cv2
from inference import OnnxDetector
img = cv2.imread('sample.jpg')
for size in (640, 960, 1280):
    det = OnnxDetector('model/best.onnx', conf_threshold=0.01)
    det.input_height = det.input_width = size
    inp, _, _ = det._preprocess(img)
    cs = det.session.run(None, {det.input_name: inp})[0][0].T[:, 4:]
    print(f"imgsz={size}: fire={cs[:,0].max():.4f}  smoke={cs[:,1].max():.4f}")
PY

# 2. Pipeline đầy đủ -> Undetermined
python - <<'PY'
import cv2
from inference import OnnxDetector, DirectionEstimator
img = cv2.imread('sample.jpg')
det = OnnxDetector('model/best.onnx')
dets = det.detect(img, conf_threshold=0.25)
print(DirectionEstimator().estimate(img, dets, is_video=False))   # -> (None, 0.0, None)
PY
```

```bash
# 3. Chứng minh rò rỉ video từ output của chính notebook
cd ..
python - <<'PY'
import pandas as pd
d = pd.read_csv('results/smoke_directions.csv')
num = d['image'].str.extract(r'dba_vd(\d+)')[0]
d['video'] = num.str[:-6]
print("estimates:", len(d), "/ 392 test images")
print("distinct source videos:", d['video'].nunique(), "->", sorted(d['video'].unique()))
PY
```

**File tham chiếu**
- Ảnh tái hiện: `backend/sample.jpg`, `backend/sample_annotated.jpg`
- Ground truth: `results/runs/eval_test/val_batch{0,1,2}_labels.jpg`
- Output pipeline: `results/smoke_directions.csv`
- Minh hoạ (cherry-picked): `results/fig_spread.png`
- Script audit: `scripts/audit_labels.py`
