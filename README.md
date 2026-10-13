<div align="center">

# VN-Sign-Language-Recognition

Hệ thống nhận diện cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt (VSL) theo thời gian thực dựa trên MediaPipe Holistic và mạng nơ-ron chuỗi thời gian (Bi-GRU + Temporal Attention).

<br/>

<p align="center">
  <img src="assets/banner.png" alt="VN-Sign-Language-Recognition Banner" width="90%">
</p>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_%3C1ms-005ced?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-8%2F8_Passed-brightgreen?style=flat-square&logo=pytest&logoColor=white)](#-kiểm-thử-tự-động)

<br/>

[Tổng quan](#-tổng-quan) • 
[Demo UI](#-giao-diện-chạy-thực-tế) • 
[Kiến trúc hệ thống](#-kiến-trúc-hệ-thống) • 
[Benchmark](#-đánh-giá-hiệu-năng--benchmark) • 
[Cài đặt & Khởi chạy](#-cài-đặt--sử-dụng) • 
[Triển khai Docker](#-triển-khai-docker) • 
[Giấy phép](#-giấy-phép)

</div>

---

## 📌 Tổng quan

Dự án xử lý bài toán nhận diện ngôn ngữ ký hiệu động (Dynamic Sign Language Recognition) bằng cách trích xuất tọa độ khớp xương (Landmarks) thay vì truyền trực tiếp toàn bộ khung hình RGB vào mạng 3D-CNN nặng nề:

- **Đặc trưng đầu vào:** 258 tọa độ/frame (33 điểm cơ thể Pose $\times$ 4 + 21 điểm bàn tay trái $\times$ 3 + 21 điểm bàn tay phải $\times$ 3).
- **Cửa sổ thời gian (Temporal Window):** 30 frames liên tiếp ($\approx 1.0$ giây ở tốc độ 30 FPS).
- **Mô hình suy luận:** Bi-directional GRU kết hợp cơ chế Temporal Attention, phân loại 20 nhãn ký hiệu thông dụng.
- **Tối ưu hóa:** Hỗ trợ ONNX Runtime giúp độ trễ suy luận giảm xuống **~0.88ms trên CPU thông thường**.
- **Ứng dụng:** Web Studio (FastAPI + WebSocket streaming + Web Speech TTS) và CLI OpenCV HUD thời gian thực.

---

## 🖥️ Giao diện chạy thực tế

<p align="center">
  <img src="assets/demo_ui.png" alt="OpenCV HUD Realtime Demo" width="90%">
</p>

*Màn hình nhận diện thời gian thực qua OpenCV HUD: trích xuất trực tiếp khung xương 2 bàn tay và nửa thân trên qua MediaPipe Tasks, hiển thị Top-3 xác suất dự đoán, tốc độ khung hình và câu dịch tiếng Việt có dấu.*

---

## 💻 Trích đoạn Mã nguồn & Kiểm tra Thực nghiệm (CodeSnap)

<p align="center">
  <img src="assets/code_snapshot.png" alt="Bi-GRU Architecture CodeSnap" width="90%">
</p>

<p align="center">
  <img src="assets/terminal_snapshot.png" alt="Pytest & ONNX Benchmark Terminal" width="90%">
</p>

---

## 🏗️ Kiến trúc hệ thống

```mermaid
flowchart LR
    A[Webcam / Video 30 FPS] --> B[MediaPipe Tasks\nPose + Hand Detector]
    B --> C[Feature Extractor\nVector 258-d / frame]
    C --> D[Sliding Window Buffer\nShape: 1 x 30 x 258]
    D --> E[ONNX Runtime Engine\nBi-GRU + Attention]
    E --> F[Temporal Smoother\nMajority Vote & Debounce]
    F --> G[Sentence Builder\n+ Vietnamese Text-to-Speech]
```

### Các thành phần chính:

| Thư mục / File | Trách nhiệm |
| :--- | :--- |
| `src/vsl/core/` | Trích xuất vector landmark 258 chiều, bộ lọc làm mượt chống rung giật (debouncing), hàm vẽ HUD tiếng Việt UTF-8. |
| `src/vsl/models/` | Định nghĩa kiến trúc PyTorch (Bi-GRU, Bi-LSTM, Transformer) và lớp bọc suy luận ONNX Runtime đa luồng. |
| `src/vsl/db/` | Mô hình SQLAlchemy 2.0 Async (SQLite/PostgreSQL) lưu trữ từ điển, phiên dịch và phản hồi người dùng. |
| `src/vsl/api/` | Máy chủ FastAPI, WebSocket streaming hai chiều, endpoint phân tích video timeline và tra cứu từ điển. |
| `src/vsl/cli/` | Giao diện dòng lệnh chạy trực tiếp OpenCV HUD và công cụ thu thập mẫu ký hiệu mới. |

---

## 📊 Đánh giá hiệu năng & Benchmark

### 1. Quá trình hội tụ mô hình (Training & Validation Curves)
Mô hình hội tụ sau khoảng 40 epochs với hàm mất mát Cross-Entropy giảm sâu và độ chính xác Validation đạt **97.6%**.

<p align="center">
  <img src="assets/training_curves.png" alt="Training and Validation Curves" width="90%">
</p>

### 2. Ma trận nhầm lẫn (Confusion Matrix - 20 nhãn)

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="85%">
</p>

### 3. So sánh kiến trúc mô hình

<p align="center">
  <img src="assets/model_benchmark.png" alt="Model Benchmark Comparison" width="90%">
</p>

| Kiến trúc | Số tham số | Độ trễ CPU (Inference) | Yêu cầu VRAM | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| 3D-CNN / I3D | ~32.4M | ~85.0 ms | > 4.0 GB | 91.8% |
| CNN-2D + LSTM | ~11.2M | ~32.0 ms | ~1.5 GB | 94.2% |
| **VSL Bi-GRU + Attention (PyTorch)** | **~428K** | **~3.5 ms** | **Không cần** | **97.4%** |
| **VSL Bi-GRU (ONNX Runtime)** | **~428K** | **~0.88 ms** | **Không cần** | **97.4%** |

---

## 🚀 Cài đặt & Sử dụng

### Yêu cầu hệ thống
- Python $\ge$ 3.10
- Webcam (đối với chế độ nhận diện trực tiếp)

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/ngtrong-1707/VN-Sign-Language-Recognition.git
cd VN-Sign-Language-Recognition

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# .venv\Scripts\activate   # Trên Windows

# Cài đặt package và dependencies
pip install -e ".[dev]"
```

### 2. Chạy Web Studio (Giao diện Trình duyệt)

```bash
python server.py
```
Mở trình duyệt tại: `http://localhost:8000`

- Chế độ **Trực tiếp Webcam**: Nhận diện cử chỉ qua camera, phát âm TTS tiếng Việt.
- Chế độ **Tải lên Video**: Đọc video file và trích xuất dòng thời gian các ký hiệu được thực hiện.
- Chế độ **Từ điển VSL**: Tra cứu danh mục 20 từ vựng kèm hướng dẫn cử chỉ.

### 3. Chạy OpenCV HUD (Desktop App)

```bash
# Quét danh sách các camera đang kết nối:
python 03_realtime_inference.py --list-cameras

# Chạy với camera mặc định (Camera 0):
python 03_realtime_inference.py

# Hoặc chọn nguồn camera cụ thể (Camera 1, URL RTSP, hoặc video stream):
python 03_realtime_inference.py --camera 1 --width 1280 --height 720
```
- Phím `c`: Xóa câu văn bản dịch hiện tại.
- Phím `q`: Thoát ứng dụng.

### 4. Huấn luyện lại mô hình (Tùy chọn)

```bash
python train_pipeline.py
```

---

## 🐳 Triển khai Docker

Khởi chạy đồng thời API Server và PostgreSQL bằng Docker Compose:

```bash
docker compose up --build -d
```

Kiểm tra trạng thái container:
```bash
docker compose ps
docker compose logs -f vsl-api
```

---

## 🧪 Kiểm thử tự động

Bộ kiểm thử bao gồm kiểm tra tính hợp lệ của vector trích xuất 258 chiều, forward pass của mô hình nơ-ron, bộ lọc consensus và các REST endpoints:

```bash
pytest tests/ -v
```

---

## 📜 Giấy phép

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
Tự do sử dụng, chỉnh sửa và phân phối cho mục đích học tập cũng như thương mại.
