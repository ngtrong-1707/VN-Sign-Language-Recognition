"""
Configuration settings for Real-Time Vietnamese Sign Language (VSL) Recognition System.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 20 Common Vietnamese Sign Language (VSL) Actions with Metadata
ACTIONS_METADATA = [
    {
        "id": 0,
        "name": "Xin chao",
        "display": "Xin chào",
        "category": "Chào hỏi",
        "desc": "Bàn tay phải mở, đưa ngang tầm thái dương rồi vẫy nhẹ hoặc đưa nhẹ về phía trước.",
        "icon": "👋"
    },
    {
        "id": 1,
        "name": "Cam on",
        "display": "Cảm ơn",
        "category": "Giao tiếp",
        "desc": "Bàn tay chạm nhẹ vào cằm/ngực rồi mở hướng về phía người đối diện kèm cúi nhẹ đầu.",
        "icon": "🙏"
    },
    {
        "id": 2,
        "name": "Tam biet",
        "display": "Tạm biệt",
        "category": "Chào hỏi",
        "desc": "Nâng bàn tay lên ngang vai và vẫy qua lại nhịp nhàng.",
        "icon": "🙋"
    },
    {
        "id": 3,
        "name": "Toi",
        "display": "Tôi / Bản thân",
        "category": "Đại từ",
        "desc": "Ngón trỏ chỉ thẳng vào giữa ngực của bản thân.",
        "icon": "👤"
    },
    {
        "id": 4,
        "name": "Ban",
        "display": "Bạn / Đối phương",
        "category": "Đại từ",
        "desc": "Ngón trỏ hướng thẳng về phía người đối diện.",
        "icon": "👉"
    },
    {
        "id": 5,
        "name": "Yeu",
        "display": "Yêu / Thương",
        "category": "Cảm xúc",
        "desc": "Hai tay bắt chéo ôm trước ngực hoặc ký hiệu I Love You (ngón cái, trỏ, út mở).",
        "icon": "❤️"
    },
    {
        "id": 6,
        "name": "Gia dinh",
        "display": "Gia đình",
        "category": "Xã hội",
        "desc": "Hai bàn tay tạo hình chữ F hoặc khum tròn rồi mở rộng tạo thành vòng tròn gắn kết.",
        "icon": "👨‍👩‍👧‍👦"
    },
    {
        "id": 7,
        "name": "Giup do",
        "display": "Giúp đỡ",
        "category": "Giao tiếp",
        "desc": "Bàn tay trái xòe ngửa, tay phải nắm đặt lên lòng bàn tay trái rồi cả 2 tay cùng nâng lên.",
        "icon": "🤝"
    },
    {
        "id": 8,
        "name": "Bac si",
        "display": "Bác sĩ",
        "category": "Nghề nghiệp",
        "desc": "Hai ngón tay phải gõ nhẹ lên cổ tay trái (tượng trưng cho việc bắt mạch).",
        "icon": "👨‍⚕️"
    },
    {
        "id": 9,
        "name": "Truong hoc",
        "display": "Trường học",
        "category": "Địa điểm",
        "desc": "Hai lòng bàn tay vỗ nhẹ vào nhau hai lần theo chiều ngang.",
        "icon": "🏫"
    },
    {
        "id": 10,
        "name": "An com",
        "display": "Ăn cơm / Đói",
        "category": "Đời sống",
        "desc": "Các đầu ngón tay chụm lại đưa về phía miệng 2-3 lần (động tác và cơm).",
        "icon": "🍚"
    },
    {
        "id": 11,
        "name": "Uong nuoc",
        "display": "Uống nước",
        "category": "Đời sống",
        "desc": "Bàn tay làm hình chữ C cầm cốc tưởng tượng đưa lên miệng và ngửa nhẹ.",
        "icon": "🥛"
    },
    {
        "id": 12,
        "name": "Vui ve",
        "display": "Vui vẻ / Hạnh phúc",
        "category": "Cảm xúc",
        "desc": "Lòng bàn tay xoa nhẹ lên ngực theo chiều kim đồng hồ, nét mặt rạng rỡ.",
        "icon": "😊"
    },
    {
        "id": 13,
        "name": "Buon",
        "display": "Buồn bã",
        "category": "Cảm xúc",
        "desc": "Bàn tay mở trước mặt rồi từ từ vuốt nhẹ xuống dưới kèm nét mặt buồn.",
        "icon": "😔"
    },
    {
        "id": 14,
        "name": "Viet Nam",
        "display": "Việt Nam",
        "category": "Địa danh",
        "desc": "Bàn tay tạo hình chữ V hoặc chữ V đưa lên ngực áo trái vị trí trái tim.",
        "icon": "🇻🇳"
    },
    {
        "id": 15,
        "name": "Di dau",
        "display": "Đi đâu / Ở đâu",
        "category": "Câu hỏi",
        "desc": "Hai bàn tay mở ngửa, lắc nhẹ sang hai bên kèm nhướn mày hỏi.",
        "icon": "❓"
    },
    {
        "id": 16,
        "name": "Bao nhieu",
        "display": "Bao nhiêu tiền",
        "category": "Câu hỏi",
        "desc": "Các ngón tay ngửa lên và cử động miết ngón cái qua các đầu ngón tay.",
        "icon": "💰"
    },
    {
        "id": 17,
        "name": "Dong y",
        "display": "Đồng ý / Đúng",
        "category": "Giao tiếp",
        "desc": "Bàn tay nắm lại thành nắm đấm rồi gật nhẹ nắm tay xuống (như gật đầu).",
        "icon": "✅"
    },
    {
        "id": 18,
        "name": "Khong",
        "display": "Không / Không phải",
        "category": "Giao tiếp",
        "desc": "Ngón trỏ giơ lên và lắc qua lắc lại sang 2 bên.",
        "icon": "❌"
    },
    {
        "id": 19,
        "name": "Ten gi",
        "display": "Tên là gì?",
        "category": "Câu hỏi",
        "desc": "Hai ngón tay chữ H/U gõ chéo vào nhau 2 lần.",
        "icon": "🏷️"
    }
]

ACTIONS = [item["name"] for item in ACTIONS_METADATA]
ACTION_DISPLAY_MAP = {item["name"]: item["display"] for item in ACTIONS_METADATA}
ACTION_ICON_MAP = {item["name"]: item["icon"] for item in ACTIONS_METADATA}

# Dataset & Sequence parameters
NO_SEQUENCES = 40       # Number of video samples per sign
SEQUENCE_LENGTH = 30    # 30 frames (~1.0s at 30 FPS)
CAPTURE_DELAY_SEC = 2

# Landmark dimensions: Pose 33*4 + LH 21*3 + RH 21*3 = 258
POSE_LANDMARKS_COUNT = 33
HAND_LANDMARKS_COUNT = 21
INPUT_FEATURE_DIM = (POSE_LANDMARKS_COUNT * 4) + (HAND_LANDMARKS_COUNT * 3 * 2)  # 258

# Model & Training Hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.25
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Real-time Inference Thresholds
PREDICTION_THRESHOLD = 0.75
HISTORY_SMOOTHING_LEN = 5

# Paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "vsl_gru_best.pth")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "vsl_model.onnx")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Server Config
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
