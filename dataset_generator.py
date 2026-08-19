"""
Module: Dynamic Spatial-Temporal Dataset Generator for Vietnamese Sign Language (VSL).
Generates realistic kinematic trajectories for 20 VSL signs with biomechanical variations,
allowing complete offline training and pre-trained checkpoint initialization.
"""

import os
import sys
import math
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config



def generate_vsl_kinematics(action_id, sequence_len=30, variation_seed=None):
    """
    Generates a realistic 30-frame x 258-feature sequential landmark trajectory
    based on the biomechanical signature of each Vietnamese Sign Language action.
    """
    if variation_seed is not None:
        np.random.seed(variation_seed)

    # Base person anatomical anchors (normalized camera space [0, 1])
    # Nose around (0.5, 0.2, 0.0)
    # Left shoulder (0.6, 0.38, 0.0), Right shoulder (0.4, 0.38, 0.0)
    # Left elbow (0.65, 0.55, 0.0), Right elbow (0.35, 0.55, 0.0)
    # Left wrist (0.62, 0.70, 0.0), Right wrist (0.38, 0.70, 0.0)
    
    person_scale = np.random.uniform(0.92, 1.08)
    person_offset_x = np.random.uniform(-0.04, 0.04)
    person_offset_y = np.random.uniform(-0.03, 0.03)

    t = np.linspace(0, 1, sequence_len)
    speed_factor = np.random.uniform(0.85, 1.15)
    t = np.power(t, speed_factor)  # Simulate non-linear acceleration

    sequence = np.zeros((sequence_len, config.INPUT_FEATURE_DIM))

    # Distinct kinematic signatures for 20 VSL signs:
    # We model 3D trajectory curves for Right Hand (RH), Left Hand (LH), and Pose
    for frame_idx, cur_t in enumerate(t):
        # 1. Pose Base (33 landmarks * 4 = 132 values)
        pose = np.zeros(33 * 4)
        # Nose (idx 0)
        pose[0] = 0.5 + person_offset_x + 0.01 * np.sin(cur_t * np.pi)
        pose[1] = 0.22 + person_offset_y
        pose[2] = 0.0
        pose[3] = 0.99  # Visibility

        # Left Shoulder (idx 11) & Right Shoulder (idx 12)
        pose[11*4] = 0.6 * person_scale + person_offset_x
        pose[11*4+1] = 0.38 + person_offset_y
        pose[11*4+3] = 0.98

        pose[12*4] = 0.4 * person_scale + person_offset_x
        pose[12*4+1] = 0.38 + person_offset_y
        pose[12*4+3] = 0.98

        # 2. Hand Keypoint Initializations
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        # Hand dynamics per action:
        # Phase 0 -> 1: Preparation -> Core Gesture Stroke -> Retraction
        stroke = math.sin(cur_t * math.pi)  # 0 -> 1 -> 0

        if action_id == 0:  # "Xin chao" (Wave near right temple)
            rh_x = 0.35 + 0.06 * math.sin(cur_t * 4 * math.pi) * stroke
            rh_y = 0.45 - 0.25 * stroke
            rh_z = -0.1 * stroke
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, open_palm=True, spread=0.03)

        elif action_id == 1:  # "Cam on" (Chin to forward opening)
            rh_x = 0.5 - 0.05 * cur_t
            rh_y = 0.30 + 0.15 * cur_t
            rh_z = -0.05 - 0.15 * stroke
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, open_palm=True, spread=0.02)

        elif action_id == 2:  # "Tam biet" (Wave at shoulder level)
            rh_x = 0.30 + 0.08 * math.sin(cur_t * 6 * math.pi) * stroke
            rh_y = 0.35 - 0.05 * stroke
            rh_z = -0.05
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, open_palm=True, spread=0.04)

        elif action_id == 3:  # "Toi" (Index finger pointing to chest)
            rh_x = 0.48 + 0.02 * (1 - stroke)
            rh_y = 0.42 + 0.02 * stroke
            rh_z = -0.02
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, index_point=True)

        elif action_id == 4:  # "Ban" (Index pointing forward)
            rh_x = 0.45 - 0.05 * stroke
            rh_y = 0.46
            rh_z = -0.25 * stroke
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, index_point=True)

        elif action_id == 5:  # "Yeu" (I Love You sign / Crossed arms)
            rh_x = 0.42 + 0.08 * stroke
            rh_y = 0.40 - 0.05 * stroke
            rh_z = -0.1
            lh_x = 0.58 - 0.08 * stroke
            lh_y = 0.40 - 0.05 * stroke
            lh_z = -0.1
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, ily_sign=True)
            lh = fill_hand_landmarks(lh_x, lh_y, lh_z, ily_sign=True)

        elif action_id == 6:  # "Gia dinh" (Two hands circular enclosure)
            angle = cur_t * math.pi
            rh_x = 0.5 - 0.12 * math.cos(angle)
            rh_y = 0.48 + 0.06 * math.sin(angle)
            lh_x = 0.5 + 0.12 * math.cos(angle)
            lh_y = 0.48 + 0.06 * math.sin(angle)
            rh = fill_hand_landmarks(rh_x, rh_y, -0.05, open_palm=True)
            lh = fill_hand_landmarks(lh_x, lh_y, -0.05, open_palm=True)

        elif action_id == 7:  # "Giup do" (Right fist on flat left palm, lifting)
            lift = 0.12 * stroke
            lh_x, lh_y = 0.50, 0.55 - lift
            rh_x, rh_y = 0.50, 0.51 - lift
            lh = fill_hand_landmarks(lh_x, lh_y, 0.0, open_palm=True)
            rh = fill_hand_landmarks(rh_x, rh_y, -0.05, fist=True)

        elif action_id == 8:  # "Bac si" (Pulse tapping on wrist)
            lh_x, lh_y = 0.55, 0.52
            tap = 0.03 * math.sin(cur_t * 8 * math.pi) * stroke
            rh_x, rh_y = 0.55, 0.49 + tap
            lh = fill_hand_landmarks(lh_x, lh_y, 0.0, open_palm=True)
            rh = fill_hand_landmarks(rh_x, rh_y, -0.03, index_point=True)

        elif action_id == 9:  # "Truong hoc" (Two palms clapping horizontally twice)
            clap = 0.06 * abs(math.sin(cur_t * 4 * math.pi)) * stroke
            lh_x, lh_y = 0.50 + clap, 0.48
            rh_x, rh_y = 0.50 - clap, 0.48
            lh = fill_hand_landmarks(lh_x, lh_y, 0.0, open_palm=True)
            rh = fill_hand_landmarks(rh_x, rh_y, 0.0, open_palm=True)

        elif action_id == 10:  # "An com" (Fingertips to mouth repeatedly)
            mouth_dist = 0.1 * abs(math.sin(cur_t * 4 * math.pi)) * stroke
            rh_x = 0.50
            rh_y = 0.28 + mouth_dist
            rh = fill_hand_landmarks(rh_x, rh_y, -0.08, pinch=True)

        elif action_id == 11:  # "Uong nuoc" (Cup to mouth tilting)
            tilt = 0.15 * stroke
            rh_x = 0.48
            rh_y = 0.42 - 0.14 * stroke
            rh_z = -0.1 - tilt * 0.1
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, c_shape=True)

        elif action_id == 12:  # "Vui ve" (Chest circular stroking)
            cir_t = cur_t * 3 * math.pi
            rh_x = 0.46 + 0.06 * math.cos(cir_t) * stroke
            rh_y = 0.42 + 0.04 * math.sin(cir_t) * stroke
            rh = fill_hand_landmarks(rh_x, rh_y, -0.05, open_palm=True)

        elif action_id == 13:  # "Buon" (Downward hand sweep on face)
            rh_x = 0.48
            rh_y = 0.22 + 0.28 * cur_t
            rh_z = -0.05
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, open_palm=True)

        elif action_id == 14:  # "Viet Nam" (V sign at chest / heart)
            rh_x = 0.55 - 0.05 * stroke
            rh_y = 0.38 - 0.02 * stroke
            rh_z = -0.08
            rh = fill_hand_landmarks(rh_x, rh_y, rh_z, v_sign=True)

        elif action_id == 15:  # "Di dau" (Open palms shaking horizontally)
            shake = 0.05 * math.sin(cur_t * 6 * math.pi) * stroke
            rh_x, rh_y = 0.38 + shake, 0.50
            lh_x, lh_y = 0.62 + shake, 0.50
            rh = fill_hand_landmarks(rh_x, rh_y, 0.0, open_palm=True)
            lh = fill_hand_landmarks(lh_x, lh_y, 0.0, open_palm=True)

        elif action_id == 16:  # "Bao nhieu" (Money rubbing gesture)
            rub = 0.02 * math.sin(cur_t * 8 * math.pi) * stroke
            rh_x, rh_y = 0.44, 0.48 + rub
            rh = fill_hand_landmarks(rh_x, rh_y, -0.05, pinch=True)

        elif action_id == 17:  # "Dong y" (Nodding fist)
            nod = 0.06 * math.sin(cur_t * 4 * math.pi) * stroke
            rh_x, rh_y = 0.42, 0.45 + nod
            rh = fill_hand_landmarks(rh_x, rh_y, -0.05, fist=True)

        elif action_id == 18:  # "Khong" (Index finger wagging)
            wag = 0.06 * math.sin(cur_t * 6 * math.pi) * stroke
            rh_x, rh_y = 0.45 + wag, 0.40 - 0.05 * stroke
            rh = fill_hand_landmarks(rh_x, rh_y, -0.08, index_point=True)

        elif action_id == 19:  # "Ten gi" (Two index/middle fingers tapping cross)
            tap = 0.04 * abs(math.sin(cur_t * 5 * math.pi)) * stroke
            rh_x, rh_y = 0.48 - tap, 0.45
            lh_x, lh_y = 0.52 + tap, 0.45
            rh = fill_hand_landmarks(rh_x, rh_y, 0.0, v_sign=True)
            lh = fill_hand_landmarks(lh_x, lh_y, 0.0, v_sign=True)

        # Add Gaussian jitter (simulate sensor/camera noise)
        jitter = np.random.normal(0, 0.003, size=(258,))
        frame_features = np.concatenate([pose, lh, rh]) + jitter
        sequence[frame_idx] = frame_features

    return sequence


def fill_hand_landmarks(base_x, base_y, base_z, open_palm=False, fist=False,
                        index_point=False, ily_sign=False, pinch=False,
                        c_shape=False, v_sign=False, spread=0.03):
    """
    Constructs a 21-landmark 3D hand topology around base wrist coordinate.
    """
    hand = np.zeros((21, 3))
    hand[0] = [base_x, base_y, base_z]  # Wrist

    # Finger lengths & offsets relative to wrist
    finger_offsets = [
        (-0.03, -0.02),  # Thumb
        (-0.015, -0.05), # Index
        (0.0, -0.06),    # Middle
        (0.015, -0.05),  # Ring
        (0.03, -0.035)   # Pinky
    ]

    for f_idx, (dx, dy) in enumerate(finger_offsets):
        base_node = 1 + f_idx * 4
        # Multipliers based on posture
        curl = 1.0
        if fist:
            curl = 0.3
        elif index_point and f_idx != 1:
            curl = 0.3
        elif ily_sign and f_idx in [1, 2]: # middle & ring curled
            if f_idx in [2, 3]: curl = 0.25
        elif pinch and f_idx not in [0, 1]:
            curl = 0.3
        elif v_sign and f_idx not in [1, 2]:
            curl = 0.25

        for joint in range(4):
            frac = (joint + 1) / 4.0
            node = base_node + joint
            hand[node, 0] = base_x + dx * frac * (1.0 + spread)
            hand[node, 1] = base_y + dy * frac * curl
            hand[node, 2] = base_z + (0.02 * (1 - curl))

    return hand.flatten()


def generate_and_save_dataset():
    print("\n" + "=" * 65)
    print(" KHỞI TẠO BỘ DỮ LIỆU ĐỘNG HỌC KHÔNG GIAN - THỜI GIAN VSL (20 TỪ VỰNG)")
    print("=" * 65)

    total_samples = 0
    for meta in config.ACTIONS_METADATA:
        action_name = meta["name"]
        action_id = meta["id"]
        dir_name = action_name.replace(" ", "_")

        action_dir = os.path.join(config.DATA_DIR, dir_name)
        os.makedirs(action_dir, exist_ok=True)

        for seq_idx in range(config.NO_SEQUENCES):
            seq_dir = os.path.join(action_dir, str(seq_idx))
            os.makedirs(seq_dir, exist_ok=True)

            seed = action_id * 1000 + seq_idx
            seq_data = generate_vsl_kinematics(action_id, config.SEQUENCE_LENGTH, variation_seed=seed)

            for frame_idx in range(config.SEQUENCE_LENGTH):
                np.save(os.path.join(seq_dir, f"{frame_idx}.npy"), seq_data[frame_idx])

            total_samples += 1

        print(f" [+] Đã tạo {config.NO_SEQUENCES} chuỗi mẫu cho ký hiệu [{meta['display']}] ({action_name})")

    print("=" * 65)
    print(f"[HOÀN TẤT] Tổng cộng đã tạo: {total_samples} chuỗi ({total_samples * config.SEQUENCE_LENGTH} frames)")
    print(f"Thư mục lưu trữ: {config.DATA_DIR}\n")


if __name__ == "__main__":
    generate_and_save_dataset()
