import torch
import json
from utils.utils import calculate_metrics

def trainer(args, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
    best_dice = 0.0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        train_dice = 0
        train_iou = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            
            # Xử lý Deep Supervision nếu model trả về tuple (ví dụ: logits, ds1, ds2)
            if isinstance(preds, tuple):
                main_pred = preds[0]
                # Tính loss cho main output và các auxiliary outputs với trọng số giảm dần
                loss = criterion(main_pred, masks)
                loss += 0.3 * criterion(preds[1], masks)
                loss += 0.2 * criterion(preds[2], masks)
                preds = main_pred  # Dùng main_pred để tính metrics
            else:
                loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Tính toán metric trong lúc huấn luyện (detach để không tính gradient)
            iou, dice = calculate_metrics(preds.detach(), masks)
            train_iou += iou
            train_dice += dice
            
            print(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
        train_loss = epoch_loss / len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
            
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                
                iou, dice = calculate_metrics(preds, masks)
                val_dice += dice
                val_iou += iou
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        if scheduler is not None:
            scheduler.step()
            
        print(f"\n=== EPOCH {epoch+1} ===")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}\n")
        
        # Lưu vào lịch sử
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f'{args.model}_best_model.pth')
            print(f"Saved {args.model}_best_model.pth with Dice: {best_dice:.4f}\n")
            
    # Xuất lịch sử ra file JSON
    history_filename = f"{args.model}_history.json"
    with open(history_filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Đã lưu lịch sử huấn luyện thành công vào '{history_filename}'!")

