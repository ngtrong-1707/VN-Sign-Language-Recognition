"""
Module 02: Preprocessing, Data Augmentation, and Model Training Pipeline.
Trains a lightweight Bidirectional GRU/LSTM on sequential landmark data.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import config
from models.vsl_model import VSL_BiGRU, VSL_BiLSTM, VSL_Transformer


class LandmarkAugmentor:
    """
    Data augmentation techniques tailored for 1D/2D landmark coordinates.
    Increases robustness against hand jitter, varying signing speed, and small scale shifts.
    """
    @staticmethod
    def add_gaussian_noise(sequence, noise_std=0.015):
        noise = np.random.normal(0, noise_std, sequence.shape)
        # Apply noise only to valid non-zero landmarks
        mask = (sequence != 0).astype(float)
        return sequence + (noise * mask)

    @staticmethod
    def random_scale(sequence, scale_range=(0.95, 1.05)):
        factor = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * factor

    @staticmethod
    def temporal_resample(sequence, target_len=30):
        """Simulate slight speed variation in gesture performance."""
        curr_len = len(sequence)
        indices = np.linspace(0, curr_len - 1, target_len)
        resampled = np.zeros((target_len, sequence.shape[1]))
        for dim in range(sequence.shape[1]):
            resampled[:, dim] = np.interp(indices, np.arange(curr_len), sequence[:, dim])
        return resampled


class VSLDataset(Dataset):
    def __init__(self, sequences, labels, is_train=False):
        self.sequences = sequences
        self.labels = labels
        self.is_train = is_train
        self.augmentor = LandmarkAugmentor()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        
        # Apply augmentations during training
        if self.is_train:
            if np.random.rand() > 0.5:
                seq = self.augmentor.add_gaussian_noise(seq)
            if np.random.rand() > 0.5:
                seq = self.augmentor.random_scale(seq)

        tensor_x = torch.tensor(seq, dtype=torch.float32)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y


def load_dataset():
    """
    Scans data directory and loads sequential .npy landmark files.
    """
    sequences = []
    labels = []
    
    label_map = {action: i for i, action in enumerate(config.ACTIONS)}
    has_real_data = False

    for action in config.ACTIONS:
        action_dir_name = action.replace(" ", "_")
        action_dir = os.path.join(config.DATA_DIR, action_dir_name)
        
        if not os.path.exists(action_dir):
            continue

        for seq_idx in range(config.NO_SEQUENCES):
            seq_folder = os.path.join(action_dir, str(seq_idx))
            if not os.path.exists(seq_folder):
                continue

            window = []
            valid_seq = True
            for frame_num in range(config.SEQUENCE_LENGTH):
                file_path = os.path.join(seq_folder, f"{frame_num}.npy")
                if os.path.exists(file_path):
                    res = np.load(file_path)
                    window.append(res)
                else:
                    valid_seq = False
                    break

            if valid_seq and len(window) == config.SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])
                has_real_data = True

    # If no real data has been collected yet, generate synthetic demonstration data
    if not has_real_data or len(sequences) == 0:
        print("[THÔNG BÁO] Chưa tìm thấy dữ liệu thu thập thực tế từ webcam.")
        print(">> Đang khởi tạo dữ liệu mẫu (synthetic benchmark) để kiểm thử pipeline...")
        for act_idx, action in enumerate(config.ACTIONS):
            for _ in range(20):  # 20 synthetic samples per action
                # Base dynamic trajectory
                t = np.linspace(0, 1, config.SEQUENCE_LENGTH)
                synth_seq = np.zeros((config.SEQUENCE_LENGTH, config.INPUT_FEATURE_DIM))
                # Distinct frequency/amplitude per action
                freq = (act_idx + 1) * 0.5
                for frame in range(config.SEQUENCE_LENGTH):
                    synth_seq[frame, :33*4] = 0.5 + 0.1 * np.sin(freq * t[frame] * 2 * np.pi)
                    synth_seq[frame, 33*4:33*4+63] = 0.4 + 0.2 * np.cos(freq * t[frame] * np.pi)
                    synth_seq[frame, 33*4+63:] = 0.6 + 0.15 * np.sin(freq * t[frame] * np.pi)
                sequences.append(synth_seq)
                labels.append(act_idx)

    return np.array(sequences), np.array(labels), label_map


def train_model():
    print("\n" + "=" * 60)
    print(" HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN VSL (Bi-GRU + TEMPORAL ATTENTION)")
    print("=" * 60)

    X, y, label_map = load_dataset()
    print(f"Tổng số chuỗi dữ liệu (Sequences): {len(X)}")
    print(f"Kích thước mỗi chuỗi: {X.shape[1]} frames × {X.shape[2]} features")
    print(f"Số lớp phân loại (Actions): {len(label_map)}")

    # Save label map metadata
    label_map_path = os.path.join(config.MODEL_DIR, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(config.ACTIONS, f, ensure_ascii=False, indent=2)

    # Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = VSLDataset(X_train, y_train, is_train=True)
    val_dataset = VSLDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model instantiation
    model = VSL_BiGRU(
        input_dim=config.INPUT_FEATURE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=len(config.ACTIONS),
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_acc = 0.0

    print(f"\nThiết bị huấn luyện: {config.DEVICE}")
    print("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # 1. Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in recurrent networks
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        scheduler.step()

        # 2. Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        epoch_train_acc = train_correct / total_train
        epoch_val_acc = val_correct / total_val
        epoch_val_loss = val_loss / total_val

        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
            print(
                f"Epoch [{epoch:03d}/{config.NUM_EPOCHS:03d}] | "
                f"Train Acc: {epoch_train_acc * 100:.2f}% | "
                f"Val Acc: {epoch_val_acc * 100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f}"
            )

        # Save best model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print("-" * 60)
    print(f"[HOÀN TẤT] Model tốt nhất đã lưu tại: {config.BEST_MODEL_PATH}")
    print(f"Độ chính xác Validation cao nhất: {best_val_acc * 100:.2f}%\n")

    # Evaluation on Validation Set
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    target_names = [config.ACTIONS[i] for i in sorted(set(all_targets))]
    print("Báo cáo đánh giá chi tiết (Classification Report):")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    train_model()
