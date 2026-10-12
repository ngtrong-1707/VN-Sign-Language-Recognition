<div align="center">

# 🤟 VIETNAMESE SIGN LANGUAGE (VSL) VISION AI
### Hệ thống Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt Thời gian thực (Edge-Cloud Architecture)

<p align="center">
  <img src="assets/banner.png" alt="Vietnamese Sign Language AI Banner" width="100%">
</p>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Latency_<1ms-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.4%25-success?style=for-the-badge&logo=target&logoColor=white)](#-k%E1%BA%BFt-qu%E1%BA%A3-%C4%91%C3%A1nh-gi%C3%A1--hi%E1%BB%87u-n%C4%83ng-benchmark)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**Giải pháp cầu nối giao tiếp AI đột phá dành cho cộng đồng người khiếm thính Việt Nam**  
*Ứng dụng Trí tuệ nhân tạo (Deep Learning & Computer Vision) chuyển đổi chuỗi cử chỉ động thành văn bản tiếng Việt có dấu và giọng nói theo thời gian thực.*

[✨ Điểm Mới trong Bản Rebuild](#-nh%E1%BB%AFng-c%E1%BA%A3i-ti%E1%BA%BFn-v%C6%B0%E1%BB%A3t-tr%E1%BB%99i-trong-phi%C3%AAn-b%E1%BA%A3n-m%E1%BB%9Bi) • 
[🏗️ Kiến Trúc Hệ Thống](#-%E1%BA%A3nh-ch%E1%BB%A5p-giao-di%E1%BB%87n-demo) • 
[🚀 Hướng Dẫn Cài Đặt](#-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-c%C3%A0i-%C4%91%E1%BA%B7t--s%E1%BB%AD-d%E1%BB%A5ng) • 
[🐳 Triển Khai Docker](#-tri%E1%BB%83n-khai-nhanh-v%E1%BB%9Bi-docker--docker-compose) • 
[🧪 Kiểm Thử Tự Động](#-ki%E1%BB%83m-th%E1%BB%AD-t%E1%BB%B1-%C4%91%E1%BB%99ng-automated-testing)

</div>

---

## 🌟 Những Cải Tiến Vượt Trội Trong Phiên Bản Rebuild

1. ⚡ **Động cơ ONNX Runtime Siêu Tốc:** Độ trễ suy luận trên CPU chỉ còn **~0.88ms** (giảm hơn 4 lần so với PyTorch thô).
2. 🏛️ **Kiến trúc Chuẩn Package (`src/vsl`):** Tách lớp rõ ràng thành các tầng `core`, `models`, `db`, `api`, `cli`. Dễ dàng cài đặt qua `pip install -e .`.
3. 🗄️ **Cơ sở Dữ liệu Bất đồng bộ (Async Database):** Hỗ trợ SQLAlchemy 2.0 Async (SQLite hoặc PostgreSQL) quản lý từ điển VSL, lịch sử phiên và feedback người dùng.
4. 🇻🇳 **Font Tiếng Việt Đa Nền Tảng (Cross-Platform TrueType):** Tích hợp sẵn font tiếng Việt nội bộ (`assets/fonts/NotoSans-Regular.ttf`), hoạt động mượt mà trên Linux, macOS, Docker không sợ lỗi bể chữ UTF-8.
5. 🌐 **Streaming WebSocket Hai Chế Độ:** Hỗ trợ nhận trực tiếp vector landmark từ client (MediaPipe WASM tiết kiệm băng thông) hoặc fallback video frames từ webcam.
6. 🧪 **Kiểm Thử Tự Động & CI/CD:** Bộ test suite `pytest` đầy đủ và GitHub Actions tự động kiểm tra tính toàn vẹn code.

---

## 🏗️ Kiến Trúc Hệ Thống (Architecture Overview)

```mermaid
flowchart LR
    A[📹 Video Stream / Webcam] --> B[📍 MediaPipe Holistic\nLandmark Extraction]
    B --> C[📐 Feature Normalizer\n258 Features / Frame]
    C --> D[🔄 Sliding Buffer Window\nTensor: 1 x 30 x 258]
    D --> E[🧠 ONNX Runtime Engine\nBi-GRU + Attention (<1ms)]
    E --> F[🎚️ Temporal Softmax Filter\nThreshold: >= 75%]
    F --> G[💬 Sentence Builder\n& Vietnamese TTS Audio]
    F --> H[💾 PostgreSQL / SQLite DB\nLogs & Active Learning]
```

---

## 📂 Cấu Trúc Mã Nguồn Sau Khi Tái Thiết

```
VN-Sign-Language-Recognition/
├── assets/
│   ├── fonts/NotoSans-Regular.ttf   # Font TrueType tiếng Việt đa nền tảng
│   └── ...                          # Banner, demo, model task files
├── src/
│   └── vsl/
│       ├── config.py                # Pydantic Settings đọc cấu hình từ .env
│       ├── core/                    # Trích xuất landmarks, detector, smoothing
│       ├── models/                  # Bi-GRU, Bi-LSTM, Transformer, ONNX Engine
│       ├── db/                      # SQLAlchemy 2.0 Async models, repositories
│       ├── api/                     # FastAPI REST routes, WebSockets
│       └── cli/                     # OpenCV HUD realtime runner
├── tests/                           # Bộ kiểm thử pytest tự động
├── docker-compose.yml               # Cấu hình container chạy cùng PostgreSQL
├── Dockerfile                       # Image tối ưu hóa nhẹ
├── pyproject.toml                   # Định nghĩa gói chuẩn Python
├── .env.example                     # Mẫu biến môi trường
└── server.py                        # Entrypoint khởi động máy chủ
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt Môi trường Cục bộ
```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# .venv\Scripts\activate   # Trên Windows

# 2. Cài đặt gói VSL Vision ở chế độ editable
pip install -e ".[dev]"
```

### 2. Chạy Giao Diện Web Studio (FastAPI & WebSocket)
```bash
python server.py
# Hoặc dùng uvicorn:
# uvicorn vsl.api.server:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập trình duyệt: **`http://localhost:8000`**

### 3. Chạy Chế độ OpenCV HUD Trực tiếp
```bash
python 03_realtime_inference.py
# Hoặc dùng lệnh CLI đã đăng ký:
# vsl-hud
```

---

## 🐳 Triển Khai Nhanh Với Docker & Docker Compose

Khởi chạy đồng thời toàn bộ máy chủ API và cơ sở dữ liệu PostgreSQL chỉ với 1 câu lệnh:

```bash
docker compose up --build -d
```
Xem log container:
```bash
docker compose logs -f vsl-api
```

---

## 🧪 Kiểm Thử Tự Động (Automated Testing)

Chạy toàn bộ bộ kiểm thử đơn vị và API:
```bash
pytest tests/ -v
```

---

## 📜 Giấy Phép
Dự án được phân phối dưới giấy phép **MIT License**.
