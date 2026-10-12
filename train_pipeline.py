"""
Training Pipeline for VSL Models (Bi-GRU + Temporal Attention).
Supports automated training, validation evaluation, and ONNX export.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vsl.config import settings, ACTIONS, ACTIONS_METADATA
from vsl.models.architectures import VSL_BiGRU
from vsl.models.engine import VSLInferenceEngine
from dataset_generator import generate_and_save_dataset


class LandmarkAugmentor:
    @staticmethod
    def add_gaussian_noise(sequence, noise_std=0.012):
        noise = np.random.normal(0, noise_std, sequence.shape)
        mask = (sequence != 0).astype(float)
        return sequence + (noise * mask)

    @staticmethod
    def random_scale(sequence, scale_range=(0.94, 1.06)):
        factor = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * factor


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
        if self.is_train:
            if np.random.rand() > 0.4:
                seq = self.augmentor.add_gaussian_noise(seq)
            if np.random.rand() > 0.4:
                seq = self.augmentor.random_scale(seq)

        tensor_x = torch.tensor(seq, dtype=torch.float32)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y


def load_dataset():
    sequences = []
    labels = []
    label_map = {action: i for i, action in enumerate(ACTIONS)}

    first_action_dir = settings.DATA_DIR / ACTIONS[0].replace(" ", "_")
    if not first_action_dir.exists():
        generate_and_save_dataset()

    for action_idx, action in enumerate(ACTIONS):
        action_dir_name = action.replace(" ", "_")
        action_dir = settings.DATA_DIR / action_dir_name

        if not action_dir.exists():
            continue

        for seq_idx in range(settings.NO_SEQUENCES):
            seq_folder = action_dir / str(seq_idx)
            if not seq_folder.exists():
                continue

            window = []
            valid_seq = True
            for frame_num in range(settings.SEQUENCE_LENGTH):
                file_path = seq_folder / f"{frame_num}.npy"
                if file_path.exists():
                    res = np.load(str(file_path))
                    window.append(res)
                else:
                    valid_seq = False
                    break

            if valid_seq and len(window) == settings.SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    return np.array(sequences), np.array(labels), label_map


def train_pipeline():
    print("\n" + "=" * 65)
    print(" HUẤN LUYỆN MÔ HÌNH VSL (Bi-GRU + TEMPORAL ATTENTION - 20 LỚP CỬ CHỈ)")
    print("=" * 65)

    X, y, label_map = load_dataset()
    print(f"Tổng số chuỗi (Sequences): {len(X)}")
    print(f"Kích thước mỗi chuỗi: {X.shape[1]} frames × {X.shape[2]} features")
    print(f"Số lớp phân loại: {len(label_map)}")

    # Save label metadata map
    with open(settings.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(ACTIONS_METADATA, f, ensure_ascii=False, indent=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = VSLDataset(X_train, y_train, is_train=True)
    val_dataset = VSLDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    model = VSL_BiGRU(
        input_dim=settings.INPUT_FEATURE_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        num_layers=settings.NUM_LAYERS,
        num_classes=len(ACTIONS),
        dropout=settings.DROPOUT
    ).to(settings.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.NUM_EPOCHS)

    best_val_acc = 0.0
    print(f"\nThiết bị huấn luyện: {settings.DEVICE}")
    print("-" * 65)

    for epoch in range(1, settings.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(settings.DEVICE)
            batch_y = batch_y.to(settings.DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(settings.DEVICE)
                batch_y = batch_y.to(settings.DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val

        if epoch % 10 == 0 or epoch == settings.NUM_EPOCHS:
            print(f"Epoch [{epoch:03d}/{settings.NUM_EPOCHS:03d}] | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss/total_val:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), settings.BEST_MODEL_PATH)

    print("-" * 65)
    print(f"[HOÀN TẤT] Model tốt nhất đã lưu tại: {settings.BEST_MODEL_PATH}")
    print(f"Độ chính xác Validation cao nhất: {best_val_acc * 100:.2f}%\n")

    # Load and export to ONNX
    try:
        VSLInferenceEngine.export_pytorch_to_onnx()
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể xuất ONNX: {e}")


if __name__ == "__main__":
    train_pipeline()
