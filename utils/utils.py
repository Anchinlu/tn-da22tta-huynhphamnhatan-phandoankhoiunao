import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
    Kết hợp BCE Loss (có pos_weight để xử lý mất cân bằng lớp)
    và Dice Loss.
    
    pos_weight: Trọng số cho pixel khối u (dương tính).
                Dataset có ~98% nền, 2% khối u → pos_weight ~ 10-15
                giúp mô hình tập trung học khối u thay vì chỉ đoán toàn nền.
    """
    def __init__(self, pos_weight=10.0):
        super(DiceBCELoss, self).__init__()
        # pos_weight: mỗi pixel khối u được coi nặng gấp pos_weight lần pixel nền
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        # BCE với pos_weight — tạo tensor weight trên cùng device
        pos_weight_tensor = torch.tensor([self.pos_weight], device=inputs.device)
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        bce_loss = bce_fn(inputs, targets)

        # Dice Loss
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)

        return bce_loss + dice_loss


def calculate_metrics(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return iou.item(), dice.item()
