import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
import io
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networks.unet import UNet
from networks.transunet import TransUNet
from data.dataset import BrainTumorDataset
from utils.utils import calculate_metrics

def main():
    print("="*60)
    print("   DEMO PHÂN ĐOẠN KHỐI U NÃO: U-NET vs TRANS-UNET")
    print("="*60)

    # 1. Khởi tạo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(current_dir, "archive", "kaggle_3m")
    UNET_PATH = os.path.join(current_dir, "Unet", "unet_best_model.pth")
    TRANSUNET_PATH = os.path.join(current_dir, "Transunet", "transunet_best_model.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}\n")

    # 2. Load Dữ liệu Validation
    print("1. Đang tải tập dữ liệu...")
    all_patients = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    all_patients.sort()
    random.seed(42)  # Cố định seed để luôn lấy đúng tập validation
    random.shuffle(all_patients)
    split_idx = int(0.8 * len(all_patients))
    val_patients = all_patients[split_idx:]
    
    val_dataset = BrainTumorDataset(DATASET_PATH, patient_list=val_patients, transform=None)
    print(f"   Đã tải {len(val_dataset)} ảnh kiểm thử.\n")

    # 3. Load Mô hình
    print("2. Đang tải mô hình U-Net...")
    unet = UNet(in_channels=3, out_channels=1).to(device)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=device, weights_only=True))
    unet.eval()

    print("3. Đang tải mô hình TransUNet...")
    transunet = TransUNet(in_channels=3, out_channels=1, img_size=256).to(device)
    state_dict_tu = torch.load(TRANSUNET_PATH, map_location=device, weights_only=True)
    new_state_dict_tu = {k.replace('module.', ''): v for k, v in state_dict_tu.items()}
    transunet.load_state_dict(new_state_dict_tu)
    transunet.eval()

    # 4. Tìm các ảnh CÓ KHỐI U để demo
    print("\n4. Đang quét tìm các ca bệnh có khối u rõ ràng để Demo...")
    demo_indices = []
    for idx in range(len(val_dataset)):
        _, mask = val_dataset[idx]
        if mask.sum() > 500:  # Tìm ảnh có khối u khá lớn (trên 500 pixel)
            demo_indices.append(idx)
        if len(demo_indices) >= 5:  # Lấy 5 ảnh để demo
            break

    # 5. Chạy Inference và Vẽ
    print("\n5. BẮT ĐẦU DEMO TRỰC QUAN:")
    
    fig, axes = plt.subplots(len(demo_indices), 4, figsize=(16, 4 * len(demo_indices)))
    plt.suptitle("SO SÁNH TRỰC QUAN: U-NET vs TRANS-UNET", fontsize=20, fontweight='bold', y=0.98)
    
    with torch.no_grad():
        for row, idx in enumerate(demo_indices):
            image, mask = val_dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            mask_input = mask.unsqueeze(0).to(device)

            # Dự đoán U-Net
            unet_logits = unet(image_input)
            unet_preds = torch.sigmoid(unet_logits)
            unet_preds = (unet_preds > 0.5).float()
            _, unet_dice = calculate_metrics(unet_logits, mask_input)

            # Dự đoán TransUNet
            transunet_logits = transunet(image_input)
            transunet_preds = torch.sigmoid(transunet_logits)
            transunet_preds = (transunet_preds > 0.5).float()
            _, transunet_dice = calculate_metrics(transunet_logits, mask_input)

            # Chuẩn bị ảnh để vẽ
            img_show = image.permute(1, 2, 0).cpu().numpy()
            mask_show = mask.squeeze().cpu().numpy()
            unet_show = unet_preds.squeeze().cpu().numpy()
            transunet_show = transunet_preds.squeeze().cpu().numpy()

            # Vẽ cột 1: Ảnh gốc
            axes[row, 0].imshow(img_show)
            axes[row, 0].set_title(f"Ảnh gốc (MRI)\nImage #{idx}", fontsize=12)
            axes[row, 0].axis('off')

            # Vẽ cột 2: Ground Truth
            axes[row, 1].imshow(mask_show, cmap='gray')
            axes[row, 1].set_title(f"Nhãn thực tế\n(Ground Truth)", fontsize=12)
            axes[row, 1].axis('off')

            # Vẽ cột 3: U-Net
            axes[row, 2].imshow(unet_show, cmap='gray')
            axes[row, 2].set_title(f"U-Net\nDice: {unet_dice:.4f}", fontsize=12)
            axes[row, 2].axis('off')

            # Vẽ cột 4: TransUNet
            axes[row, 3].imshow(transunet_show, cmap='gray')
            axes[row, 3].set_title(f"TransUNet\nDice: {transunet_dice:.4f}", fontsize=12)
            axes[row, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = "demo_result.png"
    plt.savefig(save_path, dpi=150)
    print(f"-> Đã lưu ảnh Demo trực quan tại: {save_path}")
    print("-> Bạn có thể mở ảnh này lên để trình bày với thầy!\n")

if __name__ == "__main__":
    main()
