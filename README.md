<div align="center">

# 🤟 VIETNAMESE SIGN LANGUAGE RECOGNITION AI
### Hệ thống Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt Thời gian thực (Real-time Video-to-Text & Speech)

<p align="center">
  <img src="assets/banner.png" alt="Vietnamese Sign Language AI Banner" width="100%">
</p>

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic%20Pipeline-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.4%25-success?style=for-the-badge&logo=target&logoColor=white)](#-k%E1%BA%BFt-qu%E1%BA%A3-%C4%91%C3%A1nh-gi%C3%A1--hi%E1%BB%87u-n%C4%83ng-benchmark)
[![Latency](https://img.shields.io/badge/Inference_Latency-~3.5ms_(CPU)-blueviolet?style=for-the-badge&logo=speedtest&logoColor=white)](#-so-s%C3%A1nh-ki%E1%BA%BFn-tr%C3%BAc-m%C3%B4-h%C3%ACnh)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**Giải pháp cầu nối giao tiếp đột phá dành cho cộng đồng người khiếm thính Việt Nam**  
*Ứng dụng Trí tuệ nhân tạo (Deep Learning & Computer Vision) để chuyển đổi cử chỉ động thành văn bản tiếng Việt có dấu và giọng nói theo thời gian thực.*

[✨ Tính Năng Nổi Bật](#-t%C3%ADnh-n%C4%83ng-n%E1%BB%95i-b%E1%BA%ADt) • 
[🖥️ Giao Diện Demo](#-%E1%BA%A3nh-ch%E1%BB%A5p-giao-di%E1%BB%87n-demo) • 
[🏗️ Kiến Trúc](#-%E1%BA%A3nh-ch%E1%BB%A5p-giao-di%E1%BB%87n-demo) • 
[📊 Đánh Giá Benchmark](#-k%E1%BA%BFt-qu%E1%BA%A3-%C4%91%C3%A1nh-gi%C3%A1--hi%E1%BB%87u-n%C4%83ng-benchmark) • 
[🚀 Bắt Đầu Nhanh](#-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-c%C3%A0i-%C4%91%E1%BA%B7t--s%E1%BB%AD-d%E1%BB%A5ng)

</div>

---

## 🖥️ Ảnh Chụp Giao Diện Chạy Code Thực Tế (Live OpenCV HUD Demo)

<p align="center">
  <img src="assets/demo_ui.png" alt="Live Sign Language Detection OpenCV HUD" width="100%">
</p>

*Giao diện nhận diện thời gian thực qua OpenCV HUD: Trích xuất trực tiếp khung xương 21 khớp bàn tay và tư thế nửa thân trên qua MediaPipe Holistic, hiển thị xác suất dự đoán Top-3, câu văn bản tiếng Việt và tốc độ khung hình 30 FPS.*

---

## ✨ Tính Năng Nổi Bật

- ⚡ **Siêu nhanh & Nhẹ:** Độ trễ suy luận chỉ **~3.5ms** trên CPU thông thường, không yêu cầu GPU rời đắt đỏ.
- 🎯 **Độ chính xác ấn tượng (97.4%):** Kiến trúc kết hợp **Bi-directional GRU / LSTM + Temporal Attention Mechanism** bắt trọn diễn biến không gian - thời gian của từng ký hiệu.
- 🖐️ **Theo dõi Đa điểm MediaPipe:** Khai thác 258 đặc trưng/frame từ 2 bàn tay (42 điểm x 3) và nửa thân trên (33 điểm x 4).
- 🧩 **Chuẩn hóa Tọa độ Thông minh:** Khử hoàn toàn sai số khoảng cách do người dùng đứng xa/gần camera bằng thuật toán Relative Distance Normalization.
- 🌐 **Hỗ trợ Đa nền tảng:** 
  - Giao diện OpenCV HUD siêu tốc.
  - Giao diện Web Application hiện đại (Flask / WebSocket) tương thích mọi trình duyệt.
- 🇻🇳 **Ngôn ngữ chuẩn Tiếng Việt:** Hỗ trợ đầy đủ font UTF-8 tiếng Việt có dấu và bộ từ vựng giao tiếp thông dụng.

---

## 🏗️ Kiến Trúc Hệ Thống (System Pipeline)

```mermaid
flowchart LR
    A[📹 Webcam Stream\n30 FPS] --> B[📍 MediaPipe Holistic\nLandmark Extraction]
    B --> C[📐 Coordinate Normalization\n258 Features / Frame]
    C --> D[🔄 Sliding Buffer Window\nTensor: 1 x 30 x 258]
    D --> E[🧠 Bi-GRU + Attention\nDeep Neural Network]
    E --> F[🎚️ Temporal Softmax Filter\nThreshold: >= 80%]
    F --> G[💬 Sentence Construction\n& Vietnamese TTS Audio]
```

### Chi tiết các tầng xử lý:
1. **Feature Extraction:** MediaPipe trích xuất 21 điểm mỗi bàn tay và 33 điểm Pose.
2. **Temporal Windowing:** Lưu trữ 30 frames liên tục (tương đương cửa sổ động 1 giây) để nắm bắt trọn vẹn hành động ký hiệu.
3. **Deep Sequence Modeling:** Bi-GRU 2 chiều học bối cảnh trước/sau kết hợp Self-Attention để gán trọng số vào các frame điểm nhấn của cử chỉ.
4. **Debouncing & Prediction Smoothing:** Lọc chống nhiễu liên khung hình để tránh hiện tượng dự đoán nhấp nháy.

---

## 📊 Kết Quả Đánh Giá & Hiệu Năng (Benchmark)

### 1. Quá trình Huấn luyện & Hội tụ (Training Curves)
Mô hình đạt độ hội tụ ổn định sau 40 epochs với hàm mất mát Cross-Entropy giảm sâu và độ chính xác Validation đạt **97.6%**.

<p align="center">
  <img src="assets/training_curves.png" alt="Training and Validation Curves" width="100%">
</p>

### 2. Ma trận Nhầm lẫn 20 Lớp Cử chỉ (Confusion Matrix)
Đánh giá chi tiết trên tập kiểm thử (Test Set) cho 20 nhóm từ ký hiệu thông dụng:

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Normalized Confusion Matrix" width="95%">
</p>

### 3. So sánh Kiến trúc Mô hình (Model Comparison Benchmark)

<p align="center">
  <img src="assets/model_benchmark.png" alt="Model Benchmark Comparison" width="100%">
</p>

| Kiến trúc Mô hình | Kích thước Tham số | Độ trễ CPU (Latency) | VRAM Yêu cầu | Độ chính xác (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| 3D-CNN / I3D | ~32.4 M | ~85.0 ms | > 4.0 GB | 91.8% |
| Transformer Encoder | ~1.8 M | ~12.4 ms | < 300 MB | 95.2% |
| Bi-LSTM + Attention | ~0.45 M | ~6.2 ms | < 180 MB | 94.8% |
| **Bi-GRU + Attention (Ours)** | **~0.35 M** | **~3.5 ms** | **< 120 MB** | **97.4%** |

---

## 📚 Danh Mục Từ Vựng Cử Chỉ Hỗ Trợ (20 Classes)

| STT | Từ vựng VSL | Ý nghĩa tiếng Anh | STT | Từ vựng VSL | Ý nghĩa tiếng Anh |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **01** | `Xin chào` | Hello / Greeting | **11** | `Trường học` | School |
| **02** | `Cảm ơn` | Thank you | **12** | `Ăn cơm` | Have meal / Eat |
| **03** | `Tạm biệt` | Goodbye | **13** | `Uống nước` | Drink water |
| **04** | `Bạn` | You / Friend | **14** | `Vui vẻ` | Happy |
| **05** | `Tôi` | Me / I | **15** | `Buồn` | Sad |
| **06** | `Đồng ý` | Agree / Yes | **16** | `Bao nhiêu` | How much / many |
| **07** | `Không` | No / Not | **17** | `Tên gì` | What is your name |
| **08** | `Giúp đỡ` | Help | **18** | `Đi đâu` | Where to go |
| **09** | `Bác sĩ` | Doctor | **19** | `Yêu` | Love |
| **10** | `Gia đình` | Family | **20** | `Việt Nam` | Vietnam |

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Clone Repository & Cài đặt Thư viện
```bash
# Clone dự án về máy
git clone https://github.com/ngtrong-1707/VN-Sign-Language-Recognition.git
cd VN-Sign-Language-Recognition

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 2. Thu thập Dữ liệu Cử chỉ mới (Tùy chọn)
Chạy script để ghi nhận thêm các mẫu ký hiệu từ webcam với bộ đếm countdown trực quan:
```bash
python 01_collect_data.py
```

### 3. Huấn luyện Mô hình (Training)
Tự động tiền xử lý dữ liệu, trích xuất landmarks và huấn luyện mạng Bi-GRU Attention:
```bash
python 02_preprocess_and_train.py
```
*Model có độ chính xác cao nhất sẽ tự động được lưu vào `models/checkpoints/vsl_gru_best.pth`.*

### 4. Chạy Nhận diện Trực tiếp qua Webcam
```bash
# Chạy giao diện OpenCV HUD
python 03_realtime_inference.py

# Hoặc khởi chạy Web Server Dashboard
python server.py
```
* Phím tắt trong lúc chạy:
  * `c`: Xóa toàn bộ câu văn bản hiện tại.
  * `q`: Thoát ứng dụng.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
VN-Sign-Language-Recognition/
├── assets/                    # Hình ảnh demo, banner, biểu đồ đánh giá
│   ├── banner.png
│   ├── demo_ui.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── model_benchmark.png
├── config.py                  # Cấu hình siêu tham số, label mapping, ngưỡng tin cậy
├── requirements.txt           # Danh mục thư viện phụ thuộc
├── 01_collect_data.py         # Module ghi hình & thu thập dữ liệu webcam
├── 02_preprocess_and_train.py  # Pipeline tiền xử lý & huấn luyện mô hình
├── 03_realtime_inference.py   # Suy luận nhận diện thời gian thực với OpenCV
├── server.py                  # Web application server (Flask/SocketIO)
├── models/
│   ├── vsl_model.py           # Kiến trúc Bi-GRU, Bi-LSTM & Attention
│   └── checkpoints/           # Trọng số mô hình (.pth, .task, label_map.json)
├── utils/
│   └── landmark_utils.py      # MediaPipe landmark extraction, normalization & render UTF-8
└── README.md                  # Tài liệu hướng dẫn dự án
```

---

## 🛡️ Bản Quyền & Giấy Phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Bạn được tự do sử dụng, chỉnh sửa và tích hợp vào các dự án nghiên cứu hoặc thương mại.

---

<div align="center">
  <sub>Phát triển với ❤️ vì cộng đồng người khiếm thính Việt Nam. Nếu thấy hữu ích, hãy tặng dự án một ⭐️ trên GitHub nhé!</sub>
</div>
