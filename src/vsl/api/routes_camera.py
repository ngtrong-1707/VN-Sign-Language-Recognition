"""
REST API routes for Camera discovery and hardware telemetry.
"""

from typing import List, Dict, Any
import cv2
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/cameras", tags=["Camera"])


class CameraDevice(BaseModel):
    index: int
    name: str
    width: int
    height: int
    fps: int
    is_available: bool


@router.get("")
async def list_available_cameras() -> Dict[str, Any]:
    """
    Probes system video devices to detect connected webcams and capture cards.
    """
    available_cameras = []
    
    # Check first 4 camera indices
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            if ret:
                available_cameras.append({
                    "index": idx,
                    "name": f"Camera {idx} ({w}x{h} @ {fps if fps > 0 else 30}fps)",
                    "width": w,
                    "height": h,
                    "fps": fps if fps > 0 else 30,
                    "is_available": True
                })

    return {
        "status": "success",
        "total": len(available_cameras),
        "cameras": available_cameras
    }
