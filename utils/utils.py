import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: Giúp cân bằng linh hoạt giữa việc phạt False Positive (khoanh lố) 
    và False Negative (đoán hụt) thông qua alpha và beta.
    Đồng thời dùng Gamma (Focal) để ép mô hình tập trung vào các ranh giới khó đoán.
    
    alpha: Trọng số phạt cho False Positive (khoanh lố ra ngoài).
    beta: Trọng số phạt cho False Negative (bỏ sót khối u).
    gamma: Trọng số focal (càng lớn càng tập trung vào ca khó).
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Áp dụng sigmoid để đưa logit về xác suất (0, 1)
        probs = torch.sigmoid(inputs)
        
        # Làm phẳng tensor
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Tính toán True Positives, False Positives, False Negatives
        TP = (probs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * probs_flat).sum()
        FN = (targets_flat * (1 - probs_flat)).sum()
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal Tversky Loss
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky

class HybridLoss(nn.Module):
    """
    Kết hợp sức mạnh cục bộ của DiceBCE và khả năng bắt ngữ cảnh của FocalTversky.
    Dành riêng cho TransUNet V3.
    """
    def __init__(self, pos_weight=15.0, alpha=0.3, beta=0.7, gamma=4.0/3.0):
        super(HybridLoss, self).__init__()
        self.dice_bce = DiceBCELoss(pos_weight=pos_weight)
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, inputs, targets):
        return 0.5 * self.dice_bce(inputs, targets) + 0.5 * self.focal_tversky(inputs, targets)


def calculate_metrics(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return iou.item(), dice.item()
