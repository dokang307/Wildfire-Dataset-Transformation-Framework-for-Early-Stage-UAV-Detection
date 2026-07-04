# WEEK 1 PLAN — Nghiên cứu, Dữ liệu & Baseline (Data + Prototype)
### Chi tiết hóa Tuần 1 của `plan.md` — Fire Spread Direction Prediction

> **Mục tiêu tuần:** Kết thúc tuần 1 phải có: (1) repo branch mới với cấu trúc code chuẩn, (2) toàn bộ dataset đã tải và chuẩn hóa về format thống nhất, (3) EFSA dataset có mask auto-label bằng SAM2 đã QC, (4) **baseline prototype chạy được**: video FLAME 2 → mũi tên hướng khói vẽ trên frame, (5) tập test chuẩn 100–200 cặp frame có ground truth hướng lan + tài liệu metrics.
>
> **Nguyên tắc:** Baseline trước, tối ưu sau. Mọi script đều chạy được độc lập bằng CLI với đường dẫn config được, để tuần 2–3 tái sử dụng.

---

## Tổng quan phân rã công việc (WBS)

| ID | Công việc | Ngày | Output | Phụ thuộc |
|---|---|---|---|---|
| W1.1 | Đọc paper trọng tâm + chốt design | 1–2 | `docs/design_notes.md` | — |
| W1.2 | Setup repo branch + cấu trúc project | 1 | branch `feature/spread-prediction` | — |
| W1.3 | Tải & chuẩn hóa datasets | 2–3 | `data/` chuẩn hóa + `data_manifest.json` | W1.2 |
| W1.4 | Script trích frame-pairs từ video | 3 | `scripts/extract_frame_pairs.py` | W1.3 |
| W1.5 | Auto-label mask bằng SAM2 + QC | 3–4 | `data/efsa_seg/` (YOLO-seg format) | W1.3 |
| W1.6 | Baseline optical flow prototype | 4–5 | `spread/baseline_flow.py` + video demo | W1.4 |
| W1.7 | Annotation protocol + annotate test set | 5 | `data/testset_spread/` + `docs/annotation_protocol.md` | W1.4 |
| W1.8 | Định nghĩa metrics + code eval | 5 | `spread/metrics.py` + `docs/metrics.md` | W1.7 |

Chi tiết từng công việc bên dưới.

---

## W1.1 — Đọc paper & chốt thiết kế (Ngày 1–2, ~6h)

**Việc cần làm:**
1. Đọc theo thứ tự ưu tiên (từ `resources.md`), mỗi paper trả lời 1 câu hỏi thiết kế:
   - **#9 Dense Optical Flow of Smoke Plume Motion** → cách tính flow trong smoke mask, cách lấy dominant vector. *Câu hỏi: dùng magnitude threshold nào để lọc pixel nhiễu?*
   - **#16 FoSp/SmokeSeg** → đặc thù early smoke (nhỏ, trong suốt) ảnh hưởng segmentation thế nào.
   - **#12 See the wind (VisualWind)** → cách map chuyển động → tốc độ gió khi không có API.
   - **#21 FLAME 3 paper (arXiv 2412.02831)** → cấu trúc NADIR thermal set, tần suất chụp, cách người ta đo rate of spread — đây là nguồn ground truth chính.
   - **#26/#27 Rothermel + FARSITE** → chỉ đọc phần elliptical growth (elip Huygens: eccentricity theo wind speed) — đủ để tuần 3 code.
2. Ghi chú vào `docs/design_notes.md` theo template: *Paper → Kỹ thuật lấy được → Áp dụng vào module nào ([A]–[F]) → Tham số gợi ý*.
3. Chốt 3 quyết định thiết kế và ghi lại lý do (ADR ngắn):
   - ADR-01: Farneback làm optical flow mặc định (CPU), RAFT là optional.
   - ADR-02: SmokeSeg là nguồn mask chính, EFSA-mask (SAM2) là nguồn phụ.
   - ADR-03: Ground truth hướng lan = vector trọng tâm mask(t) → trọng tâm vùng cháy mới tại t+Δ, với Δ = 30s (FLAME 2 RGB) và 3–5 frame (FLAME 3 NADIR).

**Definition of Done:** `design_notes.md` có đủ 5 mục paper + 3 ADR, review lại với team/mentor.

---

## W1.2 — Setup repo & cấu trúc project (Ngày 1, ~2h)

**Việc cần làm:**
```bash
git checkout -b feature/spread-prediction
mkdir -p spread scripts data docs notebooks tests
```

Cấu trúc mới thêm vào repo hiện tại:
```
dsp-uav/
├── spread/                      # module mới — core của phase này
│   ├── __init__.py
│   ├── baseline_flow.py         # W1.6 — Farneback + arrow overlay
│   ├── smoke_masking.py         # tuần 2 — seg mask (stub tuần này)
│   ├── wind_fusion.py           # tuần 2 (stub)
│   ├── spread_estimator.py      # tuần 2 (stub)
│   └── metrics.py               # W1.8
├── scripts/
│   ├── download_datasets.py     # W1.3
│   ├── normalize_datasets.py    # W1.3
│   ├── extract_frame_pairs.py   # W1.4
│   ├── sam2_autolabel.py        # W1.5
│   └── qc_masks.py              # W1.5
├── data/                        # gitignore, chỉ commit manifest
│   └── data_manifest.json
└── docs/
    ├── design_notes.md
    ├── annotation_protocol.md
    └── metrics.md
```

Môi trường (venv riêng cho phase spread, tách khỏi backend venv):
```bash
python -m venv .venv-spread
source .venv-spread/bin/activate        # Windows: .\.venv-spread\Scripts\Activate.ps1
pip install opencv-python numpy scipy tqdm ultralytics kaggle matplotlib pandas
# SAM2 (chỉ cần trên máy có GPU / Kaggle):
pip install "git+https://github.com/facebookresearch/sam2.git"
```

**Definition of Done:** branch push lên remote, CI/lint (nếu có) pass, README trong `spread/` mô tả module.

---

## W1.3 — Tải & chuẩn hóa datasets (Ngày 2–3, ~1 ngày)

**Nguồn tải:**

| Dataset | Cách tải | Kích thước ước tính |
|---|---|---|
| FLAME 2 | IEEE DataPort (đăng ký free account) — tải RGB/IR video pairs | ~10–20 GB |
| FLAME 3 (CV subset) | Kaggle hoặc IEEE DataPort; **gửi email request full NADIR set ngay ngày 2** (có thể mất vài ngày duyệt → làm sớm) | subset ~vài GB |
| SmokeSeg | Theo link trong paper FoSp (arXiv 2306.04474, GitHub của tác giả) | ~2 GB |
| SMOKE5K | GitHub repo Transmission-BVM (paper #17) | ~1 GB |
| EFSA dataset | Đã có sẵn trên Kaggle của dự án (`dsp-data-optimize`) | có sẵn |

**Chuẩn hóa về format thống nhất** — mọi dataset đưa về:
```
data/normalized/<dataset_name>/
├── images/            # .jpg
├── masks/             # .png nhị phân theo class (nếu có), tên trùng ảnh
├── labels/            # YOLO .txt (bbox hoặc seg polygon, nếu có)
├── sequences/         # riêng FLAME: <seq_id>/frame_000001.jpg ... (giữ thứ tự thời gian!)
└── meta.json          # fps, nguồn, class map, license
```

`scripts/normalize_datasets.py` cần xử lý:
1. Đổi tên file về pattern thống nhất `<dataset>_<split>_<idx>.jpg`.
2. Map class về chuẩn chung: `{0: fire, 1: smoke}` (lưu ý bài học remap class trong `Preprocessing.ipynb` — **verify bằng sampling 50 ảnh/dataset, vẽ nhãn lên ảnh xem bằng mắt** trước khi tin nhãn).
3. Với FLAME video: **không** trộn frame giữa các video/sequence — giữ nguyên thư mục theo sequence để tính optical flow và tránh data leakage train/test.
4. Sinh `data_manifest.json`: số ảnh, số mask, phân bố class, checksum thư mục — commit file này vào git thay cho data.

**Quy tắc split (quan trọng, chống leakage):** split theo **sequence/burn event**, không theo frame. FLAME 3 NADIR để riêng làm test-only, tuyệt đối không train.

**Definition of Done:** `data_manifest.json` đầy đủ; notebook `notebooks/eda_datasets.ipynb` hiển thị 10 ảnh mẫu + nhãn của mỗi dataset đã verify đúng class.

---

## W1.4 — Script trích frame-pairs từ video (Ngày 3, ~3h)

**Mục đích:** tạo các cặp (frame_t, frame_t+Δ) từ video FLAME để: (a) chạy optical flow, (b) làm nguyên liệu annotate test set W1.7.

`scripts/extract_frame_pairs.py`:
```python
"""
Usage: python scripts/extract_frame_pairs.py \
    --video data/normalized/flame2/sequences/seq01.mp4 \
    --out data/frame_pairs/flame2_seq01 \
    --delta-sec 1.0 --pair-gap-sec 30 --resize 1280
"""
import cv2, os, argparse, json

def extract(video_path, out_dir, delta_sec, pair_gap_sec, resize):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delta_f   = int(round(delta_sec * fps))      # khoảng cách trong 1 cặp (cho flow)
    gap_f     = int(round(pair_gap_sec * fps))   # khoảng cách giữa các cặp (cho spread GT)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)
    meta = {"video": video_path, "fps": fps, "delta_sec": delta_sec,
            "pair_gap_sec": pair_gap_sec, "pairs": []}
    idx = 0
    for start in range(0, total - delta_f, gap_f):
        frames = []
        for off in (0, delta_f):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + off)
            ok, fr = cap.read()
            if not ok: break
            if resize:
                h, w = fr.shape[:2]
                s = resize / max(h, w)
                fr = cv2.resize(fr, (int(w*s), int(h*s)))
            frames.append(fr)
        if len(frames) == 2:
            a = os.path.join(out_dir, f"pair{idx:04d}_t0.jpg")
            b = os.path.join(out_dir, f"pair{idx:04d}_t1.jpg")
            cv2.imwrite(a, frames[0]); cv2.imwrite(b, frames[1])
            meta["pairs"].append({"id": idx, "t0_frame": start,
                                  "t1_frame": start + delta_f})
            idx += 1
    json.dump(meta, open(os.path.join(out_dir, "pairs_meta.json"), "w"), indent=2)
    print(f"Extracted {idx} pairs -> {out_dir}")
```

**Hai chế độ dùng:**
- `--delta-sec 0.2` (≈5 frame): cặp sát nhau cho **optical flow** (chuyển động khói mượt).
- `--delta-sec 30` : cặp xa nhau cho **ground truth hướng lan** (vùng cháy thay đổi rõ).

**Definition of Done:** trích được ≥ 500 cặp flow-pairs và ≥ 200 cặp spread-pairs từ ≥ 3 sequence FLAME 2 khác nhau, kèm `pairs_meta.json`.

---

## W1.5 — Auto-label mask bằng SAM2 + QC (Ngày 3–4, ~1.5 ngày)

**Ý tưởng:** EFSA dataset chỉ có bbox. Dùng SAM2 với **bbox làm prompt** để sinh mask, chuyển sang YOLO-seg format cho tuần 2 train YOLOv8s-seg. Chạy trên Kaggle GPU (T4) để nhanh.

**Bước 1 — Chạy SAM2 (Kaggle notebook hoặc `scripts/sam2_autolabel.py`):**
```python
import torch, cv2, numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

def bbox_to_mask(image_bgr, bboxes_xyxy):
    """bboxes: list các bbox từ nhãn YOLO đã denormalize."""
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks = []
    for box in bboxes_xyxy:
        m, scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
        masks.append(m[np.argmax(scores)].astype(np.uint8))  # chọn mask score cao nhất
    return masks
```

**Bước 2 — Chuyển mask → YOLO-seg polygon:**
```python
def mask_to_yolo_seg(mask, class_id, img_w, img_h, min_area=64):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        c = cv2.approxPolyDP(c, epsilon=1.5, closed=True).reshape(-1, 2)
        if len(c) < 3: continue
        norm = (c / [img_w, img_h]).clip(0, 1).flatten()
        lines.append(f"{class_id} " + " ".join(f"{v:.5f}" for v in norm))
    return lines
```

**Bước 3 — Tự động flag mask nghi ngờ để QC:** mask bị nghi ngờ nếu:
- `mask_area / bbox_area < 0.15` (SAM2 bắt hụt — hay gặp với khói loãng) hoặc `> 0.98` (mask tràn cả bbox — SAM2 fail).
- Mask có > 3 mảnh rời rạc.
Ghi các ảnh flagged vào `data/efsa_seg/qc_flagged.csv`.

**Bước 4 — QC thủ công 500 ảnh:** 300 ảnh flagged + 200 ảnh random. Dùng notebook hiển thị overlay mask, gán nhãn `ok / fix / drop` (một buổi ~3–4h). Ảnh `drop` loại khỏi train; nếu tỉ lệ ok < 70% cho class smoke → quyết định dựa hoàn toàn vào SmokeSeg cho class smoke (đúng ADR-02).

**Lưu ý rủi ro (đã lường trong plan.md):** SAM2 kém với khói trong suốt. Kỳ vọng thực tế: mask fire tốt, mask smoke trung bình. Con số QC chính là dữ liệu để quyết định tuần 2.

**Definition of Done:** `data/efsa_seg/` đủ images/labels YOLO-seg; báo cáo QC: tỉ lệ ok/fix/drop theo class.

---

## W1.6 — Baseline optical flow prototype (Ngày 4–5, ~1.5 ngày) ⭐ *deliverable quan trọng nhất tuần*

**Mục tiêu:** `spread/baseline_flow.py` — nhận video (hoặc thư mục frame), chạy detector ONNX có sẵn để lấy bbox smoke, tính Farneback flow **trong bbox smoke**, xuất video có mũi tên dominant direction. Chưa cần mask, chưa cần gió — chỉ cần chứng minh tín hiệu hướng khói lấy được từ flow.

**Thiết kế thuật toán:**
```
for mỗi cặp frame (t, t+δ):
  1. detect smoke bbox trên frame t (dùng OnnxDetector có sẵn trong backend/inference.py)
  2. crop vùng bbox (mở rộng 20%), chuyển grayscale
  3. flow = cv2.calcOpticalFlowFarneback(prev, curr,
        None, pyr_scale=0.5, levels=3, winsize=25,
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
  4. lọc pixel: giữ pixel có magnitude > percentile 60 của vùng
     (loại nhiễu rung camera: trừ đi median flow của TOÀN frame — đây là
      global motion do UAV di chuyển; smoke motion = flow_local - flow_global)
  5. dominant vector = mean có trọng số magnitude của các pixel còn lại
     direction = atan2(-vy, vx)  # đổi sang hệ toạ độ ảnh, 0° = Đông, CCW
     uncertainty = circular_variance(angles)
  6. EMA smoothing: dir_t = α·dir_raw + (1-α)·dir_{t-1}, α = 0.3
  7. vẽ: cv2.arrowedLine từ tâm bbox, độ dài ∝ mean magnitude,
     kèm text góc (độ) + compass nhỏ ở góc frame
```

**Điểm kỹ thuật cần chú ý:**
- **Khử chuyển động UAV (bước 4) là bắt buộc** — nếu drone đang bay, toàn frame có flow; trừ median global flow là cách rẻ nhất. Nếu vẫn nhiễu: dùng `cv2.findHomography` trên background (ngoài bbox) để ổn định frame trước khi tính flow.
- Circular statistics: không lấy mean số học của góc (359° và 1° trung bình phải ra 0°, không phải 180°). Dùng `atan2(mean(sin θ), mean(cos θ))`.
- Log kết quả mỗi giây vào CSV: `timestamp, direction_deg, magnitude, uncertainty, bbox` — file này chính là input cho eval W1.8.

**CLI:**
```bash
python -m spread.baseline_flow \
    --input data/normalized/flame2/sequences/seq01 \
    --model backend/model/best.onnx \
    --out outputs/seq01_arrows.mp4 --csv outputs/seq01_directions.csv
```

**Definition of Done:** chạy trên ≥ 3 sequence FLAME 2; video demo cho thấy mũi tên ổn định, cùng chiều với hướng khói nhìn bằng mắt trong ≥ 70% thời lượng (đánh giá định tính, định lượng sang W1.8); FPS xử lý ghi nhận (target ≥ 5 FPS CPU cho vùng crop).

---

## W1.7 — Annotation protocol + tập test chuẩn (Ngày 5, ~4h + có thể lấn sang cuối tuần)

**Protocol (ghi vào `docs/annotation_protocol.md`):**

1. **Đơn vị annotate:** 1 spread-pair = (frame_t, frame_t+30s) từ W1.4 (FLAME 2 RGB) hoặc (frame_t, frame_t+3..5 frame) với FLAME 3 NADIR thermal.
2. **Mỗi pair annotate 3 thứ:**
   - `spread_direction_deg`: người annotate xem 2 frame chồng nhau (blink comparison), vẽ 1 mũi tên từ tâm vùng cháy tại t đến tâm **vùng cháy mới xuất hiện** tại t+Δ. Quy ước góc: 0° = phải (Đông của ảnh), ngược chiều kim đồng hồ, lưu 0–360°.
   - `new_burn_polygon`: polygon bao vùng cháy mới (xuất hiện ở t+Δ, chưa có ở t) — dùng cho metric IoU.
   - `quality_flag`: `clear / ambiguous / no_spread` (nếu không thấy lan rõ → loại khỏi eval hướng nhưng giữ cho thống kê).
3. **Công cụ:** [makesense.ai](https://www.makesense.ai/) hoặc CVAT (nếu đã có server) — vẽ line (2 điểm) cho direction + polygon; export JSON.
4. **Kiểm soát chất lượng:** 20% pairs được annotate bởi 2 người → tính chênh lệch góc trung bình giữa 2 annotator; nếu > 20° → họp thống nhất lại protocol rồi mới annotate tiếp.
5. **Chỉ tiêu tuần 1:** tối thiểu **100 pairs** annotate xong (đủ để eval sơ bộ); đủ 200 pairs muộn nhất giữa tuần 2.

**Lưu format:** `data/testset_spread/annotations.json`:
```json
{"pair_id": "flame2_seq01_pair0003",
 "direction_deg": 47.5, "quality": "clear",
 "new_burn_polygon": [[x1,y1], [x2,y2], ...],
 "annotator": "A", "delta_sec": 30}
```

**Definition of Done:** protocol doc hoàn chỉnh; ≥ 100 pairs annotated; inter-annotator check đạt ≤ 20°.

---

## W1.8 — Metrics & code đánh giá (Ngày 5, ~3h)

**Metrics chốt (code trong `spread/metrics.py`, doc trong `docs/metrics.md`):**

| Metric | Công thức | Target (plan.md) |
|---|---|---|
| **MAE góc** | `mean(min(|pred-gt|, 360-|pred-gt|))` trên pairs `clear` | ≤ 30° (cuối tuần 4) |
| **Direction Accuracy @45°** | % pairs có sai số góc ≤ 45° | ≥ 80% |
| **IoU vùng lan** | IoU(polygon dự đoán t+Δ, `new_burn_polygon` GT) | ≥ 0.4 (từ tuần 3 mới có polygon dự đoán) |
| **Stability** | circular std của direction giữa các frame liên tiếp trong 5s (đo độ giật của mũi tên) | báo cáo, chưa đặt target |
| **FPS** | frame xử lý / giây, CPU | ≥ 5 |

```python
import numpy as np

def angular_error(pred_deg, gt_deg):
    d = np.abs(np.asarray(pred_deg) - np.asarray(gt_deg)) % 360
    return np.minimum(d, 360 - d)

def mae_angle(pred_deg, gt_deg):
    return float(np.mean(angular_error(pred_deg, gt_deg)))

def acc_at(pred_deg, gt_deg, tol=45):
    return float(np.mean(angular_error(pred_deg, gt_deg) <= tol))
```

**Chạy eval sơ bộ ngay cuối tuần 1:** nối output CSV của W1.6 với 100 pairs GT của W1.7 → con số MAE baseline đầu tiên. *Con số này là mốc so sánh cho toàn bộ tuần 2–3 (seg mask, wind fusion phải chứng minh cải thiện so với nó).*

**Definition of Done:** `spread/metrics.py` có unit test (`tests/test_metrics.py`, kiểm tra wrap-around 359°↔1°); báo cáo baseline: MAE, Acc@45°, Stability, FPS trên ≥ 3 sequence.

---

## Checklist tổng kết cuối Tuần 1 (Review meeting thứ 6)

- [ ] `docs/design_notes.md` + 3 ADR
- [ ] Branch `feature/spread-prediction` với cấu trúc module đầy đủ
- [ ] Datasets chuẩn hóa + `data_manifest.json` + EDA notebook đã verify nhãn
- [ ] Đã gửi request FLAME 3 full NADIR set (theo dõi email)
- [ ] ≥ 500 flow-pairs, ≥ 200 spread-pairs trích từ FLAME 2
- [ ] EFSA-seg dataset (SAM2) + báo cáo QC theo class
- [ ] **Video demo baseline: mũi tên hướng khói trên ≥ 3 sequence**
- [ ] ≥ 100 pairs test annotated + protocol + inter-annotator check
- [ ] Báo cáo baseline metrics (MAE góc, Acc@45°, FPS) — mốc so sánh cho tuần 2

**Tín hiệu Go/No-Go cho tuần 2:**
- **Go:** baseline MAE < 60° trên pairs `clear` → tín hiệu flow có ý nghĩa, tuần 2 đầu tư seg + wind fusion để kéo xuống ≤ 30°.
- **Điều chỉnh:** nếu MAE ≥ 60° → dành 1–2 ngày đầu tuần 2 debug khử chuyển động UAV (homography stabilization) trước khi train seg, vì flow nhiễu thì mask tốt đến đâu cũng vô nghĩa.
