import os
import random
import glob
import torch
import numpy as np
from PIL import Image
import customtkinter as ctk
import matplotlib.cm as cm
from tkinter import filedialog
import torchvision.transforms.functional as TF

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

# --- Cấu hình giao diện ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Brain Tumor Segmentation Demo - U-Net vs Mô hình cải tiến")
        self.geometry("1300x850")
        self.minsize(1000, 650)

        # Cấu hình grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="DEMO HỆ THỐNG\nPHÂN ĐOẠN U NÃO", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Nút chọn ảnh to nổi bật
        self.btn_choose_image = ctk.CTkButton(self.sidebar_frame, text="📂 Chọn Ảnh Cần Chẩn Đoán", command=self.choose_image_file, height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_choose_image.grid(row=1, column=0, padx=20, pady=(30, 20), sticky="ew")

        # Dashboard Hệ số
        self.metrics_frame = ctk.CTkFrame(self.sidebar_frame)
        self.metrics_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(self.metrics_frame, text="KẾT QUẢ ĐÁNH GIÁ", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(10, 5))
        self.lbl_gt_px = ctk.CTkLabel(self.metrics_frame, text="Khối u thực tế: -- px", text_color="lightgreen", font=ctk.CTkFont(weight="bold"))
        self.lbl_gt_px.pack(pady=(0, 10))
        
        # U-Net metrics
        ctk.CTkLabel(self.metrics_frame, text="U-Net", text_color="orange", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        self.lbl_unet_dice = ctk.CTkLabel(self.metrics_frame, text="Dice: --")
        self.lbl_unet_dice.pack()
        self.lbl_unet_iou = ctk.CTkLabel(self.metrics_frame, text="IoU: --")
        self.lbl_unet_iou.pack()
        self.lbl_unet_px = ctk.CTkLabel(self.metrics_frame, text="Dự đoán: -- px")
        self.lbl_unet_px.pack()

        # TransUNet metrics
        ctk.CTkLabel(self.metrics_frame, text="Mô hình cải tiến", text_color="cyan", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 0))
        self.lbl_trans_dice = ctk.CTkLabel(self.metrics_frame, text="Dice: --")
        self.lbl_trans_dice.pack()
        self.lbl_trans_iou = ctk.CTkLabel(self.metrics_frame, text="IoU: --")
        self.lbl_trans_iou.pack()
        self.lbl_trans_px = ctk.CTkLabel(self.metrics_frame, text="Dự đoán: -- px")
        self.lbl_trans_px.pack()

        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="Vui lòng bấm 'Tải ảnh ngẫu nhiên'\nđể bắt đầu.", text_color="gray")
        self.lbl_status.grid(row=7, column=0, padx=20, pady=20, sticky="s")

        # --- Main View (Tabs) ---
        self.tabview = ctk.CTkTabview(self, width=800)
        self.tabview.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.tab_basic = self.tabview.add("So sánh Tổng quan")
        self.tab_advanced = self.tabview.add("Phân tích Lớp phủ (Overlay)")
        
        self.setup_tab_basic()
        self.setup_tab_advanced()

        # --- Khởi tạo dữ liệu ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.current_idx = None
        self.viewed_val_indices = set()
        
        self.init_models_and_data()

    def setup_tab_basic(self):
        self.tab_basic.grid_columnconfigure((0, 1), weight=1)
        self.tab_basic.grid_rowconfigure((0, 1, 2), weight=1)

        # Hàng 0
        self.img_lbl_original = ctk.CTkLabel(self.tab_basic, text="Ảnh MRI Gốc", compound="bottom", font=ctk.CTkFont(weight="bold"))
        self.img_lbl_original.grid(row=0, column=0, padx=10, pady=(10, 5))

        self.img_lbl_gt = ctk.CTkLabel(self.tab_basic, text="Nhãn thực tế (Ground Truth)", compound="bottom", font=ctk.CTkFont(weight="bold"))
        self.img_lbl_gt.grid(row=0, column=1, padx=10, pady=(10, 5))

        # Hàng 1
        self.img_lbl_unet = ctk.CTkLabel(self.tab_basic, text="Dự đoán U-Net", compound="bottom", text_color="orange", font=ctk.CTkFont(weight="bold"))
        self.img_lbl_unet.grid(row=1, column=0, padx=10, pady=(25, 5))

        self.img_lbl_trans = ctk.CTkLabel(self.tab_basic, text="Dự đoán (Cải tiến)", compound="bottom", text_color="cyan", font=ctk.CTkFont(weight="bold"))
        self.img_lbl_trans.grid(row=1, column=1, padx=10, pady=(25, 5))
        
        # Hàng 2 (Heatmap gộp chung vào)
        self.heatmap_lbl_unet = ctk.CTkLabel(self.tab_basic, text="Heatmap U-Net", compound="bottom", text_color="orange", font=ctk.CTkFont(weight="bold"))
        self.heatmap_lbl_unet.grid(row=2, column=0, padx=10, pady=(25, 5))

        self.heatmap_lbl_trans = ctk.CTkLabel(self.tab_basic, text="Heatmap (Cải tiến)", compound="bottom", text_color="cyan", font=ctk.CTkFont(weight="bold"))
        self.heatmap_lbl_trans.grid(row=2, column=1, padx=10, pady=(25, 5))

    def setup_tab_advanced(self):
        self.tab_advanced.grid_columnconfigure((0, 1), weight=1)
        self.tab_advanced.grid_rowconfigure(1, weight=1)

        legend_frame = ctk.CTkFrame(self.tab_advanced, height=40, fg_color="transparent")
        legend_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        ctk.CTkLabel(legend_frame, text="CHÚ GIẢI MÀU SẮC:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(legend_frame, text="🟩 Đúng (TP)", text_color="green", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(legend_frame, text="🟨 Khoanh Thừa (FP)", text_color="yellow", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(legend_frame, text="🟥 Bỏ Sót (FN)", text_color="red", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)

        self.overlay_lbl_unet = ctk.CTkLabel(self.tab_advanced, text="Phân tích U-Net", compound="bottom", text_color="orange", font=ctk.CTkFont(weight="bold"))
        self.overlay_lbl_unet.grid(row=1, column=0, padx=10, pady=10)

        self.overlay_lbl_trans = ctk.CTkLabel(self.tab_advanced, text="Phân tích (Cải tiến)", compound="bottom", text_color="cyan", font=ctk.CTkFont(weight="bold"))
        self.overlay_lbl_trans.grid(row=1, column=1, padx=10, pady=10)

    def init_models_and_data(self):
        self.lbl_status.configure(text="Đang tải dữ liệu và mô hình...\n(Vui lòng chờ)", text_color="yellow")
        self.update()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        UNET_PATH = os.path.join(current_dir, "Unet", "unet_best_model.pth")
        TRANSUNET_PATH = os.path.join(current_dir, "Transunet", "transunet_best_model.pth")
        
        try:
            # 2. Models
            self.unet = UNet(in_channels=3, out_channels=1).to(self.device)
            state_dict = torch.load(UNET_PATH, map_location=self.device, weights_only=True)
            self.unet.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
            self.unet.eval()

            self.transunet = TransUNet(in_channels=3, out_channels=1, img_size=256).to(self.device)
            state_dict_tu = torch.load(TRANSUNET_PATH, map_location=self.device, weights_only=True)
            self.transunet.load_state_dict({k.replace('module.', ''): v for k, v in state_dict_tu.items()})
            self.transunet.eval()

            self.transunet.eval()

            self.lbl_status.configure(text="Sẵn sàng! Hệ thống đã nạp xong\ndữ liệu kiểm thử.", text_color="lightgreen")
        except Exception as e:
            self.lbl_status.configure(text=f"Lỗi tải mô hình:\n{str(e)}", text_color="red")

    def create_overlay(self, mri_np, truth_np, pred_np):
        color_mask = np.zeros_like(mri_np)
        
        TP = (truth_np == 1) & (pred_np == 1)
        FP = (truth_np == 0) & (pred_np == 1)
        FN = (truth_np == 1) & (pred_np == 0)
        
        color_mask[TP] = [0, 255, 0]
        color_mask[FP] = [255, 255, 0]
        color_mask[FN] = [255, 0, 0]
        
        has_color = TP | FP | FN
        overlay_img = mri_np.copy()
        overlay_img[has_color] = (mri_np[has_color] * 0.4 + color_mask[has_color] * 0.6).astype(np.uint8)
        
        return overlay_img

    def set_ctk_image(self, label, pil_img, size=(220, 220)):
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
        label.configure(image=ctk_img)
        label.image = ctk_img

    def choose_image_file(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        initial_dir = os.path.join(current_dir, "archive", "kaggle_3m")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.join(current_dir, "archive", "lgg-mri-segmentation", "kaggle_3m")
            
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh MRI",
            initialdir=initial_dir,
            filetypes=[("TIFF images", "*.tif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        self.lbl_status.configure(text="Đang xử lý ảnh...", text_color="yellow")
        self.update()

        if "_mask.tif" in file_path:
            mask_path = file_path
            file_path = file_path.replace('_mask.tif', '.tif')
        else:
            mask_path = file_path.replace('.tif', '_mask.tif')
            
        if not os.path.exists(mask_path):
            self.lbl_status.configure(text="Không tìm thấy nhãn (_mask.tif)\nở cùng thư mục!", text_color="red")
            return

        self.process_and_display_image(file_path, mask_path)

    def process_and_display_image(self, file_path, mask_path):
        try:
            image = Image.open(file_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            image = image.resize((256, 256), resample=Image.BILINEAR)
            mask = mask.resize((256, 256), resample=Image.NEAREST)
            
            image_np = np.array(image)
            mask_np = np.array(mask)
            mask_np = (mask_np > 0).astype(np.float32)
            
            image_t = TF.to_tensor(image_np)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            
            image_input = image_t.unsqueeze(0).to(self.device)
            mask_input = mask_t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                unet_logits = self.unet(image_input)
                unet_iou, unet_dice = calculate_metrics(unet_logits, mask_input)

                trans_logits = self.transunet(image_input)
                trans_iou, trans_dice = calculate_metrics(trans_logits, mask_input)
                
            unet_probs = torch.sigmoid(unet_logits)
            unet_preds = (unet_probs > 0.5).float()
            
            trans_probs = torch.sigmoid(trans_logits)
            trans_preds = (trans_probs > 0.5).float()
                
            truth_np = mask_t.squeeze().cpu().numpy()
            unet_pred_np = unet_preds.squeeze().cpu().numpy()
            trans_pred_np = trans_preds.squeeze().cpu().numpy()

            gt_px = int(truth_np.sum())
            unet_px = int(unet_pred_np.sum())
            trans_px = int(trans_pred_np.sum())
            self.lbl_gt_px.configure(text=f"Khối u thực tế: {gt_px} pixel")
            self.lbl_unet_px.configure(text=f"Dự đoán: {unet_px} px ({abs(unet_px - gt_px)} sai lệch)")
            self.lbl_trans_px.configure(text=f"Dự đoán: {trans_px} px ({abs(trans_px - gt_px)} sai lệch)")

            self.lbl_unet_dice.configure(text=f"Dice: {unet_dice:.4f}")
            self.lbl_unet_iou.configure(text=f"IoU: {unet_iou:.4f}")
            self.lbl_trans_dice.configure(text=f"Dice: {trans_dice:.4f}")
            self.lbl_trans_iou.configure(text=f"IoU: {trans_iou:.4f}")

            mri_np = (image_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            truth_rgb = np.stack([truth_np*255]*3, axis=-1).astype(np.uint8)
            unet_rgb = np.stack([unet_pred_np*255]*3, axis=-1).astype(np.uint8)
            trans_rgb = np.stack([trans_pred_np*255]*3, axis=-1).astype(np.uint8)

            # Tạo Heatmap
            unet_probs_np = unet_probs.squeeze().cpu().numpy()
            trans_probs_np = trans_probs.squeeze().cpu().numpy()

            unet_heatmap_rgb = (cm.jet(unet_probs_np)[:, :, :3] * 255).astype(np.uint8)
            trans_heatmap_rgb = (cm.jet(trans_probs_np)[:, :, :3] * 255).astype(np.uint8)

            mri_gray = np.mean(mri_np, axis=2).astype(np.uint8)
            mri_gray_3c = np.stack((mri_gray,)*3, axis=-1)
            
            unet_hm_blend = (mri_gray_3c * 0.3 + unet_heatmap_rgb * 0.7).astype(np.uint8)
            trans_hm_blend = (mri_gray_3c * 0.3 + trans_heatmap_rgb * 0.7).astype(np.uint8)

            # TAB 1
            self.set_ctk_image(self.img_lbl_original, Image.fromarray(mri_np), (200, 200))
            self.set_ctk_image(self.img_lbl_gt, Image.fromarray(truth_rgb), (200, 200))
            self.set_ctk_image(self.img_lbl_unet, Image.fromarray(unet_rgb), (200, 200))
            self.set_ctk_image(self.img_lbl_trans, Image.fromarray(trans_rgb), (200, 200))
            self.set_ctk_image(self.heatmap_lbl_unet, Image.fromarray(unet_hm_blend), (200, 200))
            self.set_ctk_image(self.heatmap_lbl_trans, Image.fromarray(trans_hm_blend), (200, 200))

            # TAB 2
            overlay_unet_np = self.create_overlay(mri_np, truth_np, unet_pred_np)
            overlay_trans_np = self.create_overlay(mri_np, truth_np, trans_pred_np)
            
            self.set_ctk_image(self.overlay_lbl_unet, Image.fromarray(overlay_unet_np), (400, 400))
            
            self.current_overlay_trans_pil = Image.fromarray(overlay_trans_np)
            self.set_ctk_image(self.overlay_lbl_trans, self.current_overlay_trans_pil, (400, 400))

            self.lbl_status.configure(text=f"Đã tải xong ảnh!", text_color="lightgreen")
        except Exception as e:
            self.lbl_status.configure(text=f"Lỗi: {str(e)}", text_color="red")
if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
