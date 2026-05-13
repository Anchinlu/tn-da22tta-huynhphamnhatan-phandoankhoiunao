import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

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
        
    full_dataset = BrainTumorDataset(dataset_path)
    
    if args.test_mode or (not is_kaggle and device.type == 'cpu'):
        subset_indices = list(range(50))
        dataset = Subset(full_dataset, subset_indices)
        train_size = 40
        val_size = 10
        args.epochs = 2
        args.batch_size = 4
    else:
        dataset = full_dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == 'transunet':
        model = TransUNet(in_channels=3, out_channels=1, img_size=args.img_size).to(device)
    else:
        raise ValueError("Invalid model name")
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    trainer(args, model, train_loader, val_loader, criterion, optimizer, device)
