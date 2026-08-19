"""
Module: Automated Training & ONNX Export Pipeline for Vietnamese Sign Language (VSL).
Trains a Bi-GRU Network with Temporal Attention on 20 VSL Actions and saves production checkpoints.
"""

import os
import sys
import json
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import config
from dataset_generator import generate_and_save_dataset
from models.vsl_model import VSL_BiGRU


class LandmarkAugmentor:
    """
    Data augmentation techniques tailored for sequential landmark coordinates.
    """
    @staticmethod
    def add_gaussian_noise(sequence, noise_std=0.012):
        noise = np.random.normal(0, noise_std, sequence.shape)
        mask = (sequence != 0).astype(float)
        return sequence + (noise * mask)

    @staticmethod
    def random_scale(sequence, scale_range=(0.94, 1.06)):
        factor = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * factor

    @staticmethod
    def temporal_resample(sequence, target_len=30):
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
    label_map = {action: i for i, action in enumerate(config.ACTIONS)}

    # Check if dataset exists, if not, generate it automatically
    first_action_dir = os.path.join(config.DATA_DIR, config.ACTIONS[0].replace(" ", "_"))
    if not os.path.exists(first_action_dir):
        generate_and_save_dataset()

    for action_idx, action in enumerate(config.ACTIONS):
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

    return np.array(sequences), np.array(labels), label_map


def export_to_onnx(model, onnx_path):
    """
    Exports the trained PyTorch model to standard ONNX format for high-speed inference.
    """
    model.eval()
    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, config.INPUT_FEATURE_DIM, device=config.DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["logits"],
        dynamic_axes={"input_sequence": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"[ONNX EXPORT] Đã xuất mô hình ONNX thành công: {onnx_path}")


def train_pipeline():
    print("\n" + "=" * 65)
    print(" HUẤN LUYỆN MÔ HÌNH VSL (Bi-GRU + TEMPORAL ATTENTION - 20 LỚP CỬ CHỈ)")
    print("=" * 65)

    X, y, label_map = load_dataset()
    print(f"Tổng số chuỗi (Sequences): {len(X)}")
    print(f"Kích thước mỗi chuỗi: {X.shape[1]} frames × {X.shape[2]} features")
    print(f"Số lớp phân loại: {len(label_map)}")

    # Save label metadata map
    with open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(config.ACTIONS_METADATA, f, ensure_ascii=False, indent=2)

    # Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = VSLDataset(X_train, y_train, is_train=True)
    val_dataset = VSLDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

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
    print("-" * 65)

    for epoch in range(1, config.NUM_EPOCHS + 1):
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
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val

        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
            print(f"Epoch [{epoch:03d}/{config.NUM_EPOCHS:03d}] | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss/total_val:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print("-" * 65)
    print(f"[HOÀN TẤT] Model tốt nhất đã lưu tại: {config.BEST_MODEL_PATH}")
    print(f"Độ chính xác Validation cao nhất: {best_val_acc * 100:.2f}%\n")

    # Load and export to ONNX
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE, weights_only=True))
    try:
        export_to_onnx(model, config.ONNX_MODEL_PATH)
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể xuất ONNX: {e}")

    # Final Detailed Classification Report
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    target_names = [config.ACTIONS[i] for i in sorted(set(all_targets))]
    print("\nBáo cáo chi tiết trên tập kiểm thử:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    train_pipeline()
