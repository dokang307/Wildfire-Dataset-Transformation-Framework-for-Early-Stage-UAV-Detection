# RESOURCES — Papers & Links: Dự đoán hướng/mảng cháy lan từ ảnh & video UAV
### (Fire Spread Direction Prediction dựa trên hướng khói, hướng gió)

---

## 1. Fire Spread Prediction (Deep Learning)

1. **Wildfire Spread Prediction Through Remote Sensing and UAV Imagery-Driven Machine Learning Models** (IEEE)
   - Dự đoán đường lan của lửa từ dữ liệu UAV; trích xuất các yếu tố then chốt: mật độ thực vật, **hướng & tốc độ gió**, độ ẩm, địa hình → rất sát đề tài.
   - Link: https://openreview.net/attachment?id=B6yXACltr5&name=pdf

2. **Improving wildland fire spread prediction using deep U-Nets (FU-NetCastV2)** — ScienceDirect, 2023
   - CNN U-Net dự đoán vùng có nguy cơ lan cháy quanh đám cháy hiện tại; accuracy 94.6%, AUC 97.7%.
   - Link: https://www.sciencedirect.com/science/article/pii/S2666017223000263

3. **Machine Learning and Deep Learning for Wildfire Spread Prediction: A Review** — Fire (MDPI), 2024
   - Review hệ thống các phương pháp ML/DL dự đoán lan cháy, dataset phổ biến, hạn chế; có case dùng camera hồng ngoại gắn UAV + anemometer đo gió (FNU-LSTM).
   - Link: https://www.mdpi.com/2571-6255/7/12/482

4. **Wildfire spreading prediction using multimodal data and deep neural network approach (MA-Net)** — Scientific Reports, 2024
   - Dự đoán lan cháy 1–5 ngày; kết luận **hướng gió và land cover là feature quan trọng nhất** → củng cố lựa chọn wind fusion.
   - Link: https://www.nature.com/articles/s41598-024-52821-x

5. **Wildfire spread forecasting with Deep Learning (U-Net3D + ViT, Mesogeos)** — arXiv, 2025
   - So sánh baseline lan tròn đều vs mô hình temporal nắm bắt được lan **bất đẳng hướng theo gió**.
   - Link: https://arxiv.org/abs/2505.17556

6. **Predicting Next-Day Wildfire Spread with Time Series and Attention (SwinUnet, WSTS/WSTS+)** — arXiv, 2025
   - SOTA trên benchmark WildfireSpreadTS; framing spread prediction thành bài toán segmentation.
   - Link: https://arxiv.org/abs/2502.12003

7. **Generative AI for Predicting 2D and 3D Wildfire Spread** — arXiv, 2025
   - Review hướng generative/transformer cho dự đoán lan cháy 2D/3D, có phần về smoke plume detection.
   - Link: https://arxiv.org/abs/2506.02485

8. **Deep Learning Approaches for Wildland Fires Using Satellite Remote Sensing Data: Detection, Mapping, and Prediction** — Fire (MDPI), 2023
   - Review tổng quan detection → mapping → spread prediction, danh sách dataset.
   - Link: https://www.mdpi.com/2571-6255/6/5/192

---

## 2. Smoke Motion / Hướng khói / Ước lượng gió từ video

9. **Dense Optical Flow Retrieval of Wildfire Smoke Plume Motion from Spaceborne and Airborne Imagery** — Remote Sensing, 2026
   - Dùng dense optical flow (total-variation) trích **vector chuyển động khói**, kết hợp smoke mask từ segmentation → đúng kỹ thuật lõi của module [C].
   - Link: https://doi.org/10.3390/rs18121868

10. **Unsupervised Segmentation of Fire and Smoke from Infra-Red Videos (Horn–Schunck optical flow)** — arXiv, 2019
    - Dùng optical flow làm feature chuyển động của lửa/khói trong video.
    - Link: https://arxiv.org/abs/1909.12937

11. **A Daytime Smoke Detection Method Based on Variances of Optical Flow and HSV Color** — Fire Technology (Springer), 2024
    - Kết hợp Farneback optical flow + đặc trưng màu HSV để phát hiện & phân tích chuyển động khói.
    - Link: https://link.springer.com/article/10.1007/s10694-023-01522-4

12. **See the wind: Wind scale estimation with optical flow and VisualWind dataset** — Science of the Total Environment, 2022
    - Ước lượng cấp gió (Beaufort) từ video bằng optical flow — tham khảo cho việc suy tốc độ gió khi không có API/anemometer.
    - Link: https://www.sciencedirect.com/science/article/pii/S0048969722043029

13. **Autonomous Drone for Dynamic Smoke Plume Tracking** — 2025
    - Drone bám theo plume bằng optical flow realtime + DRL; NeRF tái tạo 3D plume, ghi nhận **directional shift theo gió**.
    - Link: https://www.researchgate.net/publication/390892888_Autonomous_Drone_for_Dynamic_Smoke_Plume_Tracking

14. **Reliable Smoke Detection via Optical Flow-Guided Feature Fusion and Transformer-Based Uncertainty Modeling** — arXiv, 2025
    - Fusion optical flow + transformer, có mô hình hóa độ bất định — tham khảo cho "cone bất định" của hướng lan.
    - Link: https://arxiv.org/abs/2508.14597

15. **Evaluation of wildland fire smoke plume dynamics using UV scanning lidar and fire-atmosphere modelling (Meso-NH + ForeFire)** — 2013
    - Đo hướng/tốc độ dịch chuyển plume bằng video + lidar, đối chiếu mô hình lan cháy vật lý có gió, dốc, nhiên liệu.
    - Link: https://www.researchgate.net/publication/260702572

---

## 3. Segmentation khói/lửa (đầu vào cho motion estimation)

16. **FoSp: Focus and Separation Network for Early Smoke Segmentation (+ SmokeSeg dataset)** — arXiv, 2023
    - Segmentation **khói giai đoạn sớm** (nhỏ, trong suốt); SmokeSeg = 6,144 ảnh thực có mask — dataset train chính đề xuất.
    - Link: https://arxiv.org/abs/2306.04474

17. **Transmission-Guided Bayesian Generative Model for Smoke Segmentation (+ SMOKE5K dataset)** — AAAI 2022
    - Dataset SMOKE5K (1,360 real + 4,000 synthetic, pixel mask).
    - Link: https://arxiv.org/abs/2303.00900

18. **AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset** — arXiv, 2026
    - Benchmark segmentation khói lớn nhất (~70K ảnh thực) — nguồn train/generalization tốt.
    - Link: https://arxiv.org/abs/2604.23542

19. **Eyes on the Environment: AI-Driven Analysis for Fire and Smoke Classification, Segmentation, and Detection / Fire and Smoke Datasets in 20 Years: An In-depth Review** — arXiv, 2025
    - Review toàn diện dataset fire/smoke 20 năm (FLAME series, BA-UAV, D-Fire, FASDD...).
    - Link: https://arxiv.org/abs/2503.14552

---

## 4. Datasets chính

20. **FLAME 2: Fire detection and modeLing — Aerial Multi-spectral imagE dataset** (IEEE DataPort)
    - Video RGB-T song song từ UAV (prescribed burn) — nguồn frame-pairs cho optical flow & test hướng lan.
    - Link: https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset

21. **FLAME 3 Dataset: Radiometric Thermal UAV Imagery for Wildfire Management** — arXiv 2024 + IEEE DataPort
    - Có **NADIR thermal time-series (3–5s/frame)** cho phép đo fire progression theo thời gian → **ground truth cho spread direction/region**.
    - Paper: https://arxiv.org/abs/2412.02831
    - Data: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management

22. **Next Day Wildfire Spread (Google Research)** — arXiv, 2021
    - Benchmark dự đoán lan cháy từ remote sensing (lead time 1 ngày); có trên Kaggle.
    - Link: https://arxiv.org/abs/2112.02447

23. **WildfireSpreadTS (WSTS): A dataset of multi-modal time series for wildfire spread prediction** — NeurIPS 2023
    - 607 sự kiện cháy 2018–2021, 23 kênh (active fire, **wind**, địa hình, thực vật) — tham khảo cho learned spread model (Phase 3).
    - Link: https://www.researchgate.net/publication/380718870
    - (Phiên bản mở rộng WSTS+ trong paper #6)

24. **FASDD: An Open-access 100,000-level Flame and Smoke Detection Dataset** — ESSD
    - ~100K ảnh flame/smoke đa nguồn (camera giám sát, drone, vệ tinh) — augment detection/segmentation.
    - Link: https://essd.copernicus.org/preprints/essd-2022-394/essd-2022-394.pdf

25. **Nguồn dữ liệu gió runtime:**
    - Open-Meteo API (miễn phí, không cần key): https://open-meteo.com/
    - OpenWeatherMap: https://openweathermap.org/api

---

## 5. Tài liệu nền tảng (mô hình lan cháy vật lý — cho elliptical spread model)

26. **Rothermel surface fire spread model** — Rothermel, R.C. (1972), *A mathematical model for predicting fire spread in wildland fuels*, USDA Forest Service.
    - Link: https://www.fs.usda.gov/rm/pubs_int/int_rp115.pdf

27. **FARSITE: Fire Area Simulator** — Finney, M.A. (1998), USDA Forest Service (mô phỏng lan cháy bằng nguyên lý Huygens với elip theo gió/dốc).
    - Link: https://www.fs.usda.gov/rm/pubs/rmrs_rp004.pdf

28. **Ultralytics YOLO Segmentation docs** (train YOLOv8/11-seg): https://docs.ultralytics.com/tasks/segment/
29. **SAM 2 (Segment Anything 2, Meta)** — auto-labeling mask từ bbox: https://github.com/facebookresearch/sam2
30. **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow** — ECCV 2020: https://arxiv.org/abs/2003.12039
31. **OpenCV Optical Flow (Farneback) tutorial**: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

---

*Ghi chú: Nhóm 1–2 là lõi phương pháp; nhóm 3–4 phục vụ train/test theo plan.md (Tuần 1–2); nhóm 5 phục vụ elliptical growth model (Tuần 3).*
