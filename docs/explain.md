# Giải thích hệ thống: Ước lượng hướng lan cháy rừng từ phân tích cột khói

> Tài liệu kỹ thuật tổng hợp: dataset, pipeline huấn luyện, phương pháp, thuật toán tính hướng khói và toàn bộ công thức.
> Mọi công thức dưới đây được trích trực tiếp từ mã nguồn đang chạy (`backend/inference.py`), không phải mô tả lý thuyết.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Dataset](#2-dataset)
3. [Pipeline huấn luyện](#3-pipeline-huấn-luyện)
4. [Export & runtime](#4-export--runtime)
5. [Quy ước góc (quan trọng nhất)](#5-quy-ước-góc-quan-trọng-nhất)
6. [Phase A — Detection](#6-phase-a--detection)
7. [Phase B — Hình học](#7-phase-b--hình-học-geometry)
8. [Phase C — Chuyển động](#8-phase-c--chuyển-động-optical-flow)
9. [Phase D — Làm mượt & vẽ vùng lan](#9-phase-d--làm-mượt--vẽ-vùng-lan)
10. [Mở rộng đa đám cháy](#10-mở-rộng-đa-đám-cháy-multi-fire)
11. [Bảng tham số tổng hợp](#11-bảng-tham-số-tổng-hợp)
12. [Kết quả thực nghiệm](#12-kết-quả-thực-nghiệm)
13. [Giới hạn đã biết](#13-giới-hạn-đã-biết)

---

## 1. Tổng quan

### Bài toán

Dự báo hướng lan của đám cháy rừng thông thường phải biết hướng gió, tức cần cảm biến khí tượng. Hệ thống này suy ra hướng lan **chỉ từ một camera RGB / UAV**, dựa trên quan sát vật lý:

> **Cột khói là chất chỉ thị thụ động (passive tracer) của trường gió cục bộ.**

Khói bị gió cuốn đi, nên hình dạng và chuyển động của cột khói mang thông tin về hướng gió — và hướng gió quyết định hướng lan của lửa.

### Kiến trúc 4 pha

```
        ảnh / khung hình
               │
        ┌──────▼──────────────────────────────┐
        │ Phase A — DETECTION                 │
        │ YOLOv8s (ONNX) → box fire + smoke   │
        └──────┬──────────────────────────────┘
               │ boxes
        ┌──────▼──────────────────────────────┐
        │ Phase B — GEOMETRY                  │
        │ Method A: vector fire → smoke       │
        │ Method B: PCA trục cột khói         │
        │ → fuse (ngưỡng lệch 60°)            │
        └──────┬──────────────────────────────┘
               │ θ_geom
        ┌──────▼──────────────────────────────┐
        │ Phase C — MOTION  (chỉ video)       │
        │ Farneback optical flow trên vùng khói│
        │ → fuse B+C (30/70, ngưỡng 45°)      │
        └──────┬──────────────────────────────┘
               │ θ_fused
        ┌──────▼──────────────────────────────┐
        │ Phase D — SPREAD                    │
        │ EMA vòng tròn (α=0.3)               │
        │ → mũi tên + ellipse lan cháy        │
        │ → abstain nếu conf < 0.2            │
        └─────────────────────────────────────┘
```

**Phase C chỉ chạy với video** (optical flow cần 2 khung hình liên tiếp). Với ảnh tĩnh, pipeline dừng ở Phase B rồi vẽ luôn.

---

## 2. Dataset

### Nguồn hiện tại: FASDD

| Thuộc tính | Giá trị |
|---|---|
| Tên | **FASDD** (Fire And Smoke Detection Dataset), subset **CV** |
| Kaggle | `yuulind/fasdd-cv-coco` |
| Định dạng gốc | COCO JSON (`annotations/{train,val,test}.json`) |
| Lớp | `fire` (id 0), `smoke` (id 1) |
| Ảnh âm (negative) | Có — ảnh không nhãn được giữ làm background |

### Chuyển đổi COCO → YOLO

Ánh xạ tên lớp linh hoạt (không phụ thuộc thứ tự category id của COCO):

```python
if 'fire' in name or 'flame' in name:  cls = 0
elif 'smoke' in name:                  cls = 1
```

Chuẩn hoá bbox từ COCO `[x, y, w, h]` (góc trên-trái, đơn vị pixel) sang YOLO (tâm, chuẩn hoá `[0,1]`):

```
x_c = clip( (x + w/2) / W , 0, 1 )
y_c = clip( (y + h/2) / H , 0, 1 )
w_n = clip( w / W , 0, 1 )
h_n = clip( h / H , 0, 1 )
```

Box có `w_n ≤ 0` hoặc `h_n ≤ 0` bị loại. Ảnh không có annotation hợp lệ → ghi file nhãn rỗng (YOLO coi là ảnh nền).

### ⚠️ Dataset cũ đã bị loại bỏ

Trước FASDD, dự án dùng một corpus gộp (DBA-YOLO + FLAME negatives) và **phải loại bỏ** do hai khiếm khuyết nghiêm trọng:

1. **Thiếu nhãn khói** — ~60% ảnh có cột khói rõ nhưng chỉ được gán nhãn `fire`. Khói không nhãn bị huấn luyện thành *background*, triệt tiêu hoàn toàn lớp `smoke`.
2. **Rò rỉ dữ liệu theo video** — toàn bộ năng lực nhận khói chỉ đến từ **5 video nguồn**, các khung hình cách nhau 1 frame nằm ở cả train lẫn test.

Hậu quả: model báo smoke mAP@50 = 0.939 nhưng khi gặp ảnh UAV thật thì điểm lớp smoke chỉ **0.006** (mù hoàn toàn). Đó là lý do dự án chuyển sang FASDD.

> **Bài học:** mAP trên tập test *in-distribution* có thể che giấu hoàn toàn việc model không tổng quát hoá. Luôn cần một tập kiểm tra **ngoài phân phối (OOD)**.

---

## 3. Pipeline huấn luyện

### Cấu hình

| Tham số | Giá trị |
|---|---|
| Kiến trúc | YOLOv8s — **11,136,374** tham số |
| Framework | Ultralytics 8.4.x |
| `imgsz` (train) | 960 |
| `imgsz` (infer) | 640 |
| `batch` | 32 |
| `epochs` | 50 (best tại epoch **47**) |
| `patience` | 15 |
| `lr0` / `lrf` | 0.01 / 0.01 |
| `cos_lr` | True (cosine LR decay) |
| Phần cứng | Kaggle GPU (Tesla T4 / P100) |

Huấn luyện chia nhiều phase do giới hạn thời gian Kaggle, nối tiếp bằng `resume=True` từ `last.pt`.

### ⚠️ Cạm bẫy của `resume=True`

Khi `resume=True`, Ultralytics **khôi phục toàn bộ tham số từ checkpoint** và chỉ nhận lại một whitelist nhỏ (`imgsz`, `batch`, `device`, `close_mosaic`). Nghĩa là:

```python
model.train(resume=True, data=..., epochs=100, project=..., name='new_run')
#                        ^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^^^
#                        TẤT CẢ đều bị BỎ QUA
```

Đây là nguyên nhân một lần chạy trước đó ghi đè vào thư mục cũ và âm thầm đánh giá nhầm checkpoint.

**Muốn đổi số epoch khi resume**, phải sửa giá trị *trong checkpoint*:

```python
ckpt = torch.load(LAST_PT, map_location='cpu', weights_only=False)
ckpt['train_args']['epochs'] = 100
torch.save(ckpt, LAST_PT)
```

hoặc bỏ hẳn `resume=True` và train mới với `epochs` truyền tường minh.

### Chỉ số hiện tại (validation split, epoch 47)

| Precision | Recall | mAP@50 | mAP@50-95 | F1 |
|---|---|---|---|---|
| 0.777 | 0.694 | **0.781** | 0.492 | 0.733 |

Chỉ số chọn model là **fitness** của Ultralytics:

```
fitness = 0.1 · mAP@50 + 0.9 · mAP@50-95
```

Lưu ý: fitness nghiêng hẳn về mAP@50-95, nên một checkpoint có mAP@50 cao hơn vẫn có thể **không** được chọn làm `best.pt`.

---

## 4. Export & runtime

### Xuất ONNX

```python
model.export(format="onnx", imgsz=640, simplify=True, dynamic=True, opset=17)
```

`results/runs/fasdd_train/weights/best.pt` → `backend/model/best.onnx` (~43 MB).

- **`dynamic=True`** → input shape `['batch', 3, 'height', 'width']`, cho phép đổi độ phân giải lúc chạy.
- Runtime: **ONNX Runtime (CPU) + OpenCV + NumPy** — không cần PyTorch khi triển khai, giảm mạnh kích thước container.

### Kiểm chứng tương đương (parity)

Sau mỗi lần export, so sánh ONNX với PyTorch trên cùng ảnh: **5/5 detection khớp lớp, sai lệch box ≤ 3 px, sai lệch confidence ≤ 1 điểm %**.

> Sai lệch nhỏ là do **cách letterbox khác nhau**: Ultralytics `predict` dùng *rect* (đệm tới 640×448 cho ảnh 1000×666), còn `OnnxDetector` đệm vuông 640×640 → tỉ lệ viền xám khác → điểm số lệch nhẹ. Không phải lỗi export.

---

## 5. Quy ước góc (quan trọng nhất)

Đây là điểm dễ sai nhất trong toàn hệ thống. **Toàn bộ phép tính hướng dùng quy ước toán học**, trong khi **ảnh dùng quy ước màn hình**:

| | Trục y | Chiều dương của góc | 0° | 90° |
|---|---|---|---|---|
| **Hệ toán học** (mọi phép tính) | hướng **lên** | ngược chiều kim đồng hồ | Đông (phải) | **Bắc (lên)** |
| **Hệ ảnh** (OpenCV, pixel) | hướng **xuống** | thuận chiều kim đồng hồ | Đông (phải) | **Nam (xuống)** |

**Quy tắc chuyển đổi — mọi nơi đọc toạ độ ảnh phải lật dấu y:**

```
dy_toán_học = − dy_ảnh
```

Áp dụng ở cả 3 chỗ:

```python
# Method A (Phase B)
dy = -(s_cy - f_cy)

# PCA (Phase B)
θ = atan2(-v_y, v_x)

# Optical flow (Phase C)
dy = -mean(fy[strong])
```

Và khi **vẽ ngược lại** ra màn hình:

```python
end_x = fcx + L·cos(θ)
end_y = fcy − L·sin(θ)      # lật y trở lại
ellipse_angle_cv2 = −θ      # cv2 xoay thuận chiều kim đồng hồ
```

> Bỏ sót phép lật dấu này từng khiến backend trả về hướng gió **đối xứng sai** so với kết quả nghiên cứu. Đây là lỗi im lặng: code vẫn chạy, chỉ có kết quả sai.

**Hiệu số góc** luôn tính theo đường tròn (tránh lỗi khi vượt 0°/360°):

```
Δ(θ₁, θ₂) = | ((θ₁ − θ₂ + 180) mod 360) − 180 |     ∈ [0°, 180°]
```

---

## 6. Phase A — Detection

### Tiền xử lý: Letterbox

Giữ nguyên tỉ lệ khung hình, đệm màu xám `(114,114,114)`:

```
r   = min( H_target / H_gốc , W_target / W_gốc )
W'  = round(W_gốc · r) ,  H' = round(H_gốc · r)
pad_x = (W_target − W') / 2 ,  pad_y = (H_target − H') / 2
```

Sau đó: BGR→RGB, chia 255, chuyển `HWC → CHW`, thêm chiều batch.

### Giải mã đầu ra

Đầu ra ONNX có shape `[batch, 6, N_anchors]`, transpose thành `[N_anchors, 6]`:

```
cột 0..3 : cx, cy, w, h   (toạ độ trong ảnh đã letterbox)
cột 4    : điểm lớp fire
cột 5    : điểm lớp smoke
```

Với mỗi anchor: `score = max(cột 4, cột 5)`, `class = argmax(...)`. Giữ lại nếu `score ≥ conf_threshold`.

**Đổi sang góc-góc rồi khử letterbox** (đưa về toạ độ ảnh gốc):

```
x1 = cx − w/2 ,  y1 = cy − h/2
x2 = cx + w/2 ,  y2 = cy + h/2

x_gốc = (x − pad_x) / r
y_gốc = (y − pad_y) / r
```

Cuối cùng clip vào biên ảnh.

### NMS theo từng lớp (class-aware)

Mẹo *offset*: dịch box của mỗi lớp ra một vùng toạ độ riêng để box khác lớp **không bao giờ** triệt tiêu nhau, nhờ đó chỉ cần một lần NMS:

```
offset       = class_id · (max_coordinate + 1)
box_cho_NMS  = box + offset
```

Sau đó NMS tiêu chuẩn theo IoU:

```
IoU(A,B) = |A ∩ B| / (|A| + |B| − |A ∩ B|)
```

Loại các box có `IoU > 0.7` với box điểm cao hơn.

---

## 7. Phase B — Hình học (Geometry)

Đầu vào: danh sách box đã chuẩn hoá `(class, cx, cy, w, h)` với toạ độ ∈ [0,1].

### Method A — Vector fire → smoke

**Trực giác:** khói bay từ đám lửa về phía cuối gió. Vector nối tâm lửa đến tâm khói chính là hướng gió.

Chọn đám lửa lớn nhất và cột khói lớn nhất (theo diện tích `w·h`):

```
dx = s_cx − f_cx
dy = −(s_cy − f_cy)                    ← lật y sang hệ toán học
d  = √(dx² + dy²)

nếu d > 0.02:
    θ_A    = atan2(dy, dx) mod 360
    conf_A = min(5d, 1.0)
```

- **Ngưỡng `d > 0.02`**: hai tâm quá gần thì hướng chỉ là nhiễu.
- **`conf_A = min(5d, 1)`**: khói càng lệch xa lửa, bằng chứng về gió càng mạnh. Đạt tối đa khi `d ≥ 0.2` (20% chiều ảnh).

### Method B — PCA trục cột khói

**Trực giác:** gió kéo cột khói dài ra theo hướng thổi. Trục chính (principal axis) của đám mây điểm ảnh khói chính là hướng gió.

**Bước 1 — Tách mặt nạ khói trong HSV** (khói = xám nhạt: bão hoà thấp, độ sáng cao):

```
mask = (S < 60) ∧ (V > 90)
```

Yêu cầu số điểm `|mask| > 200`, nếu không thì bỏ qua.

**Bước 2 — PCA.** Với tập toạ độ điểm `(xᵢ, yᵢ)` đã trừ trung bình:

```
X = [ x − x̄ ;  y − ȳ ]           (ma trận 2×n)
C = cov(X)                        (ma trận hiệp phương sai 2×2)
λ₁ ≤ λ₂ , v₁, v₂ = eig(C)         (dùng eigh: trị riêng tăng dần)
v = v₂                            (vector riêng ứng với λ lớn nhất)
```

**Bước 3 — Độ thuôn dài (elongation):**

```
E = λ₂ / max(λ₁, 1e-6)
```

Yêu cầu `E > 1.5` — cột khói phải đủ *dài* mới xác định được trục; khói tròn thì trục vô nghĩa.

**Bước 4 — Khử nhập nhằng dấu.** PCA cho ra một *trục*, không phải một *chiều* (`v` và `−v` tương đương). Giả định khói bốc lên:

```
nếu v_y > 0:  v ← −v          (ép về phía "lên" trong ảnh)
```

**Bước 5 — Góc và độ tin cậy:**

```
θ_B    = atan2(−v_y, v_x) mod 360
conf_B = min( (E − 1)/4 , 1.0 )
```

`conf_B` đạt tối đa khi `E ≥ 5` (trục dài gấp 5 lần trục ngang).

### Hợp nhất A + B

```
Δ = Δ(θ_A, θ_B)
best = A nếu conf_A ≥ conf_B, ngược lại là B

┌ có cả A và B:
│   Δ ≤ 60°  →  θ = θ_best ,  conf = (conf_A + conf_B)/2      [A+B]
│   Δ > 60°  →  θ = θ_best ,  conf = conf_best · 0.3          [A+B_conflict]
├ chỉ có A   →  θ = θ_A ,     conf = conf_A · 0.8             [A]
├ chỉ có B   →  θ = θ_B ,     conf = conf_B · 0.7             [B]
└ không có   →  None (không xác định)
```

**Ý nghĩa các hệ số:**

- **`Δ ≤ 60°` → lấy trung bình conf**: hai phương pháp độc lập cùng đồng ý ⇒ bằng chứng mạnh.
- **`Δ > 60°` → phạt xuống 0.3**: mâu thuẫn. Vẫn xuất kết quả (không im lặng) nhưng đánh dấu độ tin cậy thấp, thường rơi dưới ngưỡng abstain 0.2 nên **không vẽ**.
- **`0.8` / `0.7`**: một phương pháp đơn lẻ luôn kém tin cậy hơn hai phương pháp đồng thuận. Method B bị phạt nặng hơn vì phụ thuộc ngưỡng HSV, dễ nhiễu với mây/sương.

---

## 8. Phase C — Chuyển động (Optical Flow)

> Chỉ chạy với **video**. Ảnh tĩnh không có thông tin chuyển động.

**Trực giác:** thay vì suy ra gió từ *hình dạng* khói, đo trực tiếp *khói đang trôi về đâu* giữa hai khung hình.

### Farneback dense optical flow

```python
flow = cv2.calcOpticalFlowFarneback(
    gray_prev, gray_curr, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
```

Kết quả `flow[y,x] = (fx, fy)` — vector dịch chuyển của từng pixel.

### Lọc theo vùng khói và theo cường độ

**Bước 1 — Mặt nạ:** hợp của tất cả box lớp `smoke`. Yêu cầu `|mask| ≥ 100` pixel.

**Bước 2 — Chỉ giữ chuyển động mạnh:**

```
m       = √(fx² + fy²)
ngưỡng  = percentile_75( m )
strong  = { pixel : m > ngưỡng }
```

Yêu cầu `|strong| ≥ 50`. Lọc percentile-75 loại nhiễu nền và các pixel gần như đứng yên, chỉ giữ 25% điểm chuyển động rõ nhất — chính là phần khói thực sự đang trôi.

**Bước 3 — Vector trung bình:**

```
dx = mean( fx[strong] )
dy = −mean( fy[strong] )                   ← lật y

θ_C    = atan2(dy, dx) mod 360
conf_C = min( √(dx² + dy²) / 5 , 1.0 )
```

`conf_C` đạt tối đa khi dịch chuyển trung bình ≥ 5 pixel/khung.

### Hợp nhất B + C (trọng số 30/70)

```
Δ = Δ(θ_geom, θ_flow)

┌ Δ ≤ 45°  (đồng thuận) → cộng vector có trọng số:
│     sx = 0.3·conf_G·cos θ_G  +  0.7·conf_C·cos θ_C
│     sy = 0.3·conf_G·sin θ_G  +  0.7·conf_C·sin θ_C
│     θ    = atan2(sy, sx) mod 360
│     conf = min( √(sx² + sy²) · 1.2 , 1.0 )              [geom+flow]
│
└ Δ > 45°  (mâu thuẫn) → tin chuyển động:
      θ = θ_C ,  conf = conf_C · 0.7                      [flow_override]
```

**Vì sao ưu tiên flow 70/30?** Hình học đọc *trạng thái tĩnh* (khói đã bị thổi tới đâu), còn optical flow đo *chuyển động tức thời* — bằng chứng trực tiếp và tức thời hơn về gió hiện tại.

Việc cộng **vector** (thay vì trung bình số học của góc) là bắt buộc: nó xử lý đúng tính tuần hoàn của góc và tự động cho phương pháp có confidence cao hơn "kéo" kết quả về phía mình.

Hệ số `1.2` bù cho việc hai vector không bao giờ hoàn toàn cùng phương nên tổng độ dài luôn nhỏ hơn tổng độ lớn.

---

## 9. Phase D — Làm mượt & vẽ vùng lan

### EMA trên đường tròn đơn vị

Không thể làm mượt góc trực tiếp: trung bình của 350° và 10° cho ra **180°** (hoàn toàn sai) thay vì **0°**. Giải pháp — làm mượt trên **vector đơn vị**:

```
cx = conf · cos θ
cy = conf · sin θ

sx ← α·cx + (1−α)·sx           α = 0.3
sy ← α·cy + (1−α)·sy

θ_mượt    = atan2(sy, sx) mod 360
conf_mượt = √(sx² + sy²)
```

Khởi tạo: khung đầu tiên gán thẳng `sx, sy = cx, cy`.

**Hai lợi ích cùng lúc:**
- Biên độ vector (`conf`) tự trở thành trọng số — khung có độ tin cậy thấp tự động ít ảnh hưởng.
- `conf_mượt` đo luôn **tính nhất quán thời gian**: hướng ổn định → các vector cộng dồn → biên độ lớn; hướng dao động → triệt tiêu nhau → biên độ nhỏ.

### Cơ chế từ chối (abstain)

```
nếu conf < 0.2:  KHÔNG vẽ overlay
```

Thà không đưa ra dự đoán còn hơn đưa ra dự đoán sai — với bài toán an toàn sinh mạng, một mũi tên chỉ sai hướng nguy hiểm hơn là không có mũi tên nào.

### Ellipse lan cháy (mô hình Rothermel giản lược)

Đám cháy lan theo hình **ellipse** kéo dài về phía cuối gió, không lan tròn đều.

```
d_lửa = √(w_lửa² + h_lửa²)                    (đường chéo box lửa)

a = clip( 0.9 · d_lửa ,  0.05·max(W,H) ,  0.15·max(W,H) )     bán trục lớn
b = a · (1 − e) = 0.2a           với e = 0.8                   bán trục nhỏ
o = 0.4 · a                                                    độ lệch tâm
```

**Vị trí và hướng vẽ** (chuyển về hệ toạ độ ảnh):

```
tâm_x = f_cx + o·cos θ
tâm_y = f_cy − o·sin θ                 ← lật y
góc_cv2 = −θ                           ← cv2 xoay thuận chiều kim đồng hồ
```

**Mũi tên gió:**

```
L     = max(a, 40)  px
end_x = f_cx + L·cos θ
end_y = f_cy − L·sin θ
```

**Vì sao kẹp `a` trong `[5%, 15%]` của cạnh lớn nhất?** Ellipse tỉ lệ theo kích thước đám lửa (lửa lớn ⇒ vùng nguy cơ lớn), nhưng phải chặn hai đầu: đám lửa nhỏ xíu vẫn cần ellipse nhìn thấy được, đám lửa lớn không được phủ kín khung hình.

### Quy ước màu

| Thành phần | Màu | BGR |
|---|---|---|
| Box `fire` | Đỏ | `(0,0,255)` |
| Box `smoke` | Xanh cyan | `(255,191,0)` |
| Mũi tên gió | Xanh lá | `(0,255,0)` |
| Ellipse lan cháy | Cam, mờ 40% + viền đứt | `(0,140,255)` |

---

## 10. Mở rộng đa đám cháy (Multi-fire)

Thuật toán gốc chỉ cho **một** hướng toàn khung (lấy lửa lớn nhất + khói lớn nhất). Với ảnh nhiều đám cháy, điều này bỏ sót thông tin.

### Ghép cặp theo khói gần nhất

Với **mỗi** đám lửa `f`, chọn cột khói `s` gần nhất theo khoảng cách tâm bình phương:

```
s* = argmin_s  ( (s_cx − f_cx)² + (s_cy − f_cy)² )
```

rồi chạy đầy đủ Method A + Method B + hợp nhất cho riêng cặp `(f, s*)`.

**Kết quả:** mỗi đám lửa có mũi tên và ellipse riêng, kích thước tỉ lệ với chính nó.

Nhiều đám lửa có thể cùng chia sẻ một cột khói lớn — khi đó Method B (PCA) giống nhau nhưng Method A (vector) khác nhau, nên mỗi đám vẫn cho hướng riêng biệt. Điều này đúng về mặt vật lý: cùng một trường gió, nhưng vị trí tương đối giữa lửa và khói khác nhau.

### Khác biệt giữa ảnh và video

| | Ảnh tĩnh | Video |
|---|---|---|
| Số hướng | **Nhiều** (mỗi lửa một hướng) | **Một** hướng toàn cục |
| Phase C/D | Không (không có chuyển động) | Có (flow + EMA) |
| Neo mũi tên | Từng box lửa | Box lửa lớn nhất |

Video giữ một hướng toàn cục vì EMA cần chuỗi thời gian **liên tục theo từng đối tượng** — muốn EMA đa đám cháy thì phải có object tracking, hiện chưa triển khai.

---

## 11. Bảng tham số tổng hợp

| Tham số | Giá trị | Pha | Ý nghĩa |
|---|---|---|---|
| `conf_threshold` | 0.25 | A | Ngưỡng điểm detection |
| `iou_threshold` | 0.7 | A | Ngưỡng NMS |
| `imgsz` | 640 | A | Độ phân giải suy luận |
| ngưỡng khoảng cách | 0.02 | B-A | `d` tối thiểu (chuẩn hoá) |
| thang conf Method A | ×5 | B-A | `min(5d, 1)` |
| mặt nạ HSV | `S<60 ∧ V>90` | B-B | Tách pixel khói |
| số pixel tối thiểu | 200 | B-B | Đủ điểm để PCA |
| ngưỡng elongation | 1.5 | B-B | Cột khói phải đủ dài |
| thang conf Method B | ÷4 | B-B | `min((E−1)/4, 1)` |
| ngưỡng mâu thuẫn A/B | 60° | B | Trên mức này → phạt 0.3 |
| phạt đơn phương pháp | 0.8 / 0.7 | B | Chỉ A / chỉ B |
| pixel mặt nạ tối thiểu | 100 | C | Vùng khói tối thiểu |
| percentile chuyển động | 75 | C | Chỉ giữ 25% mạnh nhất |
| pixel mạnh tối thiểu | 50 | C | Đủ mẫu để lấy trung bình |
| thang conf flow | ÷5 | C | `min(‖v‖/5, 1)` |
| ngưỡng mâu thuẫn B/C | 45° | C | Trên mức này → flow thắng |
| trọng số hợp nhất | 0.3 / 0.7 | C | geom / flow |
| hệ số bù | ×1.2 | C | Bù suy hao khi cộng vector |
| `EMA_ALPHA` | 0.3 | D | Hệ số làm mượt |
| `DIRECTION_ABSTAIN_CONF` | 0.2 | D | Dưới mức này → không vẽ |
| `SPREAD_RADIUS` | 0.15 | D | Trần bán trục lớn |
| `SPREAD_ECC` | 0.8 | D | `b = a(1−e)` |
| `SPREAD_OFFSET` | 0.4 | D | Lệch tâm cuối gió |
| xử lý video | ~5 FPS | — | `frame_skip = fps/5` |
| giới hạn video | 30 s | — | Tránh timeout Cloud Run |

---

## 12. Kết quả thực nghiệm

### Detection (FASDD validation, epoch 47)

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.777 | 0.694 | 0.781 | 0.492 |

*Chỉ số per-class (fire / smoke riêng) và test-split chưa chạy — cần `model.val()` trên Kaggle.*

### Validation Phase C/D trên video thật

Video: *"Raw Video Shows How Fast Texas Wildfire Spread"* (640×358, 29.97 fps, 30 s xử lý, 180 khung được suy luận).

| Method | Số khung | Ý nghĩa |
|---|---|---|
| `flow_override` | 109 | Phase C ghi đè hình học |
| `geom+flow` | 54 | Hợp nhất B+C |
| `A+B_conflict` | 1 | Chỉ hình học |
| *(không xác định)* | 16 | |

**Phase C kích hoạt trên 163/180 khung (91%)** — đây là lần đầu tiên optical flow và EMA được kiểm chứng trên dữ liệu thật.

### Ảnh tĩnh (ảnh UAV thật, ngoài phân phối)

Điểm thô lớp `smoke`: **0.0063 → 0.6647** sau khi chuyển sang FASDD (cải thiện ~100×). Pipeline cho ra 4 hướng riêng cho 4 đám lửa.

---

## 13. Giới hạn đã biết

### 1. Overlay nhấp nháy trên video

Chỉ **72/180 khung (40%)** thực sự được vẽ overlay, dù 164 khung có hướng. Nguyên nhân:

- Confidence trung vị chỉ **0.162**, thấp hơn ngưỡng abstain 0.2.
- `flow_override` nhân conf với 0.7, mà `conf_C = ‖v‖/5` vốn đã nhỏ khi khói trôi chậm.
- Overlay còn cần **có box lửa** để neo (150/180 khung có lửa).

*Hướng khắc phục:* hiệu chỉnh lại thang confidence của flow, hoặc giữ overlay của khung trước trong một khoảng ngắn (temporal hold).

### 2. Phase C/D chưa validate định lượng

Đã chứng minh **có chạy** (91% khung), nhưng chưa có ground-truth hướng gió để đo **sai số góc**. Cần video có dữ liệu gió tham chiếu.

### 3. Chưa có cổng nghiệm thu OOD

Model mới chỉ được kiểm tra trên vài ảnh/video lẻ. Cần tập 30–50 ảnh aerial thật có nhãn, tiêu chí **smoke recall ≥ 0.7 @ IoU 0.5**.

### 4. FASDD-CV, không phải FASDD-UAV

FASDD có 3 subset (CV / UAV / RS). Đang dùng **CV** (ảnh camera giám sát tổng quát). Sản phẩm hướng tới UAV nên **FASDD-UAV** mới khớp domain nhất.

### 5. Giả định của Method B

Bước khử nhập nhằng dấu (`v_y > 0 → v ← −v`) **giả định khói bốc lên trong ảnh**. Với ảnh chụp thẳng từ trên xuống (nadir), giả định này có thể sai.

### 6. Mặt nạ HSV không phân biệt khói với mây/sương

Ngưỡng `S < 60 ∧ V > 90` bắt mọi vùng xám nhạt. Mây, sương mù, bầu trời sáng đều lọt qua, làm nhiễu PCA khi chúng nằm trong box khói.

---

## Tham chiếu mã nguồn

| Thành phần | File |
|---|---|
| Phase A/B/C/D (nguồn chân lý) | [`backend/inference.py`](../backend/inference.py) |
| REST API | [`backend/app.py`](../backend/app.py) |
| Kiểm thử (24 test) | [`backend/test_direction.py`](../backend/test_direction.py) |
| Xuất ONNX | [`scripts/export_onnx.py`](../scripts/export_onnx.py) |
| Notebook huấn luyện (4 pha) | `fadd-training-batch32-phase03-fixed.ipynb` |
| Kết quả đánh giá test | [`eval_results_explaination.md`](eval_results_explaination.md) |
| Audit nhãn | [`scripts/audit_labels.py`](../scripts/audit_labels.py) |
