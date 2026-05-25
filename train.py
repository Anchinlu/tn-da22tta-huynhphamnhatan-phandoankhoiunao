import os
import sys
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingLR

# Cấu hình mã hóa UTF-8 cho console Windows tránh lỗi UnicodeEncodeError
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from configs.config import get_args
from data.dataset import BrainTumorDataset
from utils.utils import DiceBCELoss
from trainer import trainer

from networks.unet import UNet
from networks.transunet import TransUNet

if __name__ == "__main__":
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if os.path.exists('/kaggle/input'):
        dataset_path = args.dataset_path_kaggle
        is_kaggle = True
    else:
        dataset_path = args.dataset_path_local
        is_kaggle = False
        
    # Lấy danh sách bệnh nhân và phân chia ở cấp độ bệnh nhân
    all_patients = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    all_patients.sort()
    random.seed(42)
    random.shuffle(all_patients)
    
    if args.test_mode or (not is_kaggle and device.type == 'cpu'):
        train_patients = all_patients[:2] if len(all_patients) >= 2 else all_patients
        val_patients = all_patients[2:3] if len(all_patients) >= 3 else all_patients
        args.epochs = 2
        args.batch_size = 4
    else:
        split_idx = int(0.8 * len(all_patients))
        train_patients = all_patients[:split_idx]
        val_patients = all_patients[split_idx:]
        
    print(f"Tổng số bệnh nhân: {len(all_patients)}")
    print(f"Số bệnh nhân huấn luyện: {len(train_patients)} | Số bệnh nhân kiểm thử: {len(val_patients)}")
    
    # Định nghĩa Data Augmentation bằng Albumentations cho tập train
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    
    train_dataset = BrainTumorDataset(dataset_path, patient_list=train_patients, transform=train_transform)
    val_dataset = BrainTumorDataset(dataset_path, patient_list=val_patients, transform=None)
    
    print(f"Số ảnh huấn luyện: {len(train_dataset)} | Số ảnh kiểm thử: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == 'transunet':
        model = TransUNet(in_channels=3, out_channels=1, img_size=args.img_size).to(device)
    else:
        raise ValueError("Invalid model name")
        
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        
    criterion = DiceBCELoss(pos_weight=args.pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Cấu hình Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(f"\n{'='*50}")
    print(f"  Model      : {args.model.upper()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  pos_weight : {args.pos_weight}  (BCE weight cho pixel khối u)")
    print(f"{'='*50}\n")
    
    trainer(args, model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)

