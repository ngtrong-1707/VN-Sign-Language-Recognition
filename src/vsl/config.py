"""
Centralized Configuration for Vietnamese Sign Language Recognition (VSL).
Uses Pydantic Settings with .env file support and smart environment detection.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

# Determine root repository directory
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent


# 20 Common Vietnamese Sign Language (VSL) Actions Metadata
DEFAULT_ACTIONS_METADATA: List[Dict[str, Any]] = [
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General App Config
    APP_NAME: str = "VSL Vision"
    VERSION: str = "2.1.0"
    DEBUG: bool = False
    
    # Base Directories
    BASE_DIR: Path = ROOT_DIR
    DATA_DIR: Path = ROOT_DIR / "data" / "sequences"
    MODEL_DIR: Path = ROOT_DIR / "models" / "checkpoints"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    STATIC_DIR: Path = ROOT_DIR / "static"
    TEMPLATES_DIR: Path = ROOT_DIR / "templates"
    FONTS_DIR: Path = ROOT_DIR / "assets" / "fonts"

    # Database Configuration
    DATABASE_URL: str = f"sqlite+aiosqlite:///{ROOT_DIR / 'vsl.db'}"

    # Landmark extraction parameters
    POSE_LANDMARKS_COUNT: int = 33
    HAND_LANDMARKS_COUNT: int = 21
    INPUT_FEATURE_DIM: int = (33 * 4) + (21 * 3 * 2)  # 258 features
    
    # Sequence & Kinematics
    SEQUENCE_LENGTH: int = 30
    NO_SEQUENCES: int = 40
    CAPTURE_DELAY_SEC: float = 2.0

    # Model & Training Parameters
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.25
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 80
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # Real-Time Inference
    PREDICTION_THRESHOLD: float = 0.75
    HISTORY_SMOOTHING_LEN: int = 5
    USE_ONNX: bool = True

    # Server settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    @property
    def DEVICE(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def BEST_MODEL_PATH(self) -> str:
        return str(self.MODEL_DIR / "vsl_gru_best.pth")

    @property
    def ONNX_MODEL_PATH(self) -> str:
        return str(self.MODEL_DIR / "vsl_model.onnx")

    @property
    def LABEL_MAP_PATH(self) -> str:
        return str(self.MODEL_DIR / "label_map.json")

    @property
    def POSE_TASK_PATH(self) -> str:
        return str(self.MODEL_DIR / "pose_landmarker.task")

    @property
    def HAND_TASK_PATH(self) -> str:
        return str(self.MODEL_DIR / "hand_landmarker.task")

    @property
    def DEFAULT_FONT_PATH(self) -> str:
        local_font = self.FONTS_DIR / "NotoSans-Regular.ttf"
        if local_font.exists():
            return str(local_font)
        win_font = Path("C:/Windows/Fonts/arial.ttf")
        if win_font.exists():
            return str(win_font)
        return ""


settings = Settings()

# Ensure critical directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.FONTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS_METADATA = DEFAULT_ACTIONS_METADATA
ACTIONS = [item["name"] for item in ACTIONS_METADATA]
ACTION_DISPLAY_MAP = {item["name"]: item["display"] for item in ACTIONS_METADATA}
ACTION_ICON_MAP = {item["name"]: item["icon"] for item in ACTIONS_METADATA}
