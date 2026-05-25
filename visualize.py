import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

# Cấu hình mã hóa UTF-8 cho console Windows tránh lỗi UnicodeEncodeError
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from configs.config import get_args
from data.dataset import BrainTumorDataset
from networks.unet import UNet
from networks.transunet import TransUNet


def visualize_results():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    local_data_path = args.dataset_path_local
    if not os.path.exists(local_data_path):
        print(f"Không tìm thấy thư mục dữ liệu tại: {local_data_path}")
        return

    dataset = BrainTumorDataset(local_data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Tìm ảnh có khối u rõ ràng (mask có diện tích đủ lớn)
    print("Đang tìm ảnh có khối u...")
    sample_image, sample_mask = None, None
    checked = 0
    for images, masks in loader:
        checked += 1
        tumor_pixels = masks.sum().item()
        if tumor_pixels > 500:   # Ít nhất 500 pixel khối u → khối u đủ to để thấy rõ
            sample_image = images
            sample_mask = masks
            print(f"  Tìm thấy sau {checked} ảnh — Khối u: {int(tumor_pixels)} pixels "
                  f"({100*tumor_pixels/(256*256):.1f}% diện tích ảnh)")
            break
        if checked >= 500:
            print("Không tìm thấy ảnh có khối u đủ lớn, dùng ảnh cuối cùng có tumor.")
            break

    if sample_image is None:
        print("Không tìm thấy ảnh nào!")
        return

    sample_image = sample_image.to(device)

    # 2. Load mô hình
    print("\nĐang khởi tạo và nạp trọng số các mô hình...")
    models = {}

    unet_path = "unet_best_model.pth"
    transunet_path = "transunet_best_model.pth"

    if os.path.exists(unet_path):
        unet_model = UNet(in_channels=3, out_channels=1).to(device)
        unet_model.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
        unet_model.eval()
        models['U-Net'] = unet_model
        print(f"  ✓ Nạp U-Net từ {unet_path}")
    else:
        print(f"  ✗ Không tìm thấy {unet_path}")

    if os.path.exists(transunet_path):
        transunet_model = TransUNet(in_channels=3, out_channels=1, img_size=args.img_size).to(device)
        transunet_model.load_state_dict(torch.load(transunet_path, map_location=device, weights_only=True))
        transunet_model.eval()
        models['TransUNet'] = transunet_model
        print(f"  ✓ Nạp TransUNet từ {transunet_path}")
    else:
        print(f"  ✗ Không tìm thấy {transunet_path}")

    # 3. Inference + debug stats
    print("\n--- Thống kê xác suất đầu ra (sigmoid) ---")
    predictions_prob = {}
    predictions_bin = {}

    with torch.no_grad():
        for name, model in models.items():
            logits = model(sample_image)
            prob = torch.sigmoid(logits)
            prob_np = prob.cpu().squeeze().numpy()
            predictions_prob[name] = prob_np

            print(f"  {name}: min={prob_np.min():.4f} | max={prob_np.max():.4f} | "
                  f"mean={prob_np.mean():.4f} | >0.5: {(prob_np > 0.5).sum()} pixels")

            binary = (prob_np > 0.5).astype(np.float32)
            predictions_bin[name] = binary

    # 4. Chuẩn bị dữ liệu hiển thị
    # Ảnh gốc: denormalize từ [0,1] về [0,1] (to_tensor đã làm việc này)
    img_np = sample_image.cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    gt_mask = sample_mask.cpu().squeeze().numpy()

    # Tính Dice score cho từng mô hình
    def dice_score(pred, gt, smooth=1e-5):
        intersection = (pred * gt).sum()
        return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    # 5. Vẽ layout đẹp
    # Hàng 1: Ảnh gốc | Ground Truth | [Prob heatmap mỗi model]
    # Hàng 2: [trống] | [trống]       | [Binary mask mỗi model]

    n_models = len(models)
    n_cols = 2 + n_models  # Ảnh gốc + GT + mỗi model
    fig = plt.figure(figsize=(4.5 * n_cols, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.15)

    title_color = 'white'
    cmap_prob = 'hot'

    def styled_ax(ax, title, fontsize=11):
        ax.set_title(title, color=title_color, fontsize=fontsize, pad=8, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Hàng 1 ---
    # Cột 0: Ảnh MRI gốc (span 2 hàng)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img_np)
    styled_ax(ax_img, "Ảnh MRI Gốc", fontsize=12)

    # Cột 1: Ground Truth mask (span 2 hàng)
    ax_gt = fig.add_subplot(gs[:, 1])
    ax_gt.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    gt_pixels = int(gt_mask.sum())
    styled_ax(ax_gt, f"Ground Truth\n({gt_pixels} pixels khối u)", fontsize=12)

    # Cột 2+: Từng mô hình
    for col_i, (name, prob_np) in enumerate(predictions_prob.items()):
        bin_np = predictions_bin[name]
        dice = dice_score(bin_np, gt_mask)
        col = 2 + col_i

        # Hàng 0: Probability heatmap
        ax_prob = fig.add_subplot(gs[0, col])
        im = ax_prob.imshow(prob_np, cmap=cmap_prob, vmin=0, vmax=1)
        styled_ax(ax_prob, f"{name}\nXác suất (Heatmap)", fontsize=11)
        plt.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)

        # Hàng 1: Binary mask
        ax_bin = fig.add_subplot(gs[1, col])
        ax_bin.imshow(bin_np, cmap='gray', vmin=0, vmax=1)
        pred_pixels = int(bin_np.sum())
        styled_ax(ax_bin, f"{name} — Binary (>0.5)\nDice: {dice:.4f} | {pred_pixels} pixels", fontsize=11)

    plt.savefig("comparison_result.png", dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nĐã lưu kết quả vào 'comparison_result.png'!")
    plt.show()


if __name__ == "__main__":
    visualize_results()
