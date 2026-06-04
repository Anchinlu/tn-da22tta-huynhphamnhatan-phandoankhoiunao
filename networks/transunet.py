import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import DoubleConv

class AttentionGate(nn.Module):
    """
    Attention Gate: Lọc các đặc trưng cục bộ (x) từ Encoder dựa trên thông tin ngữ cảnh (g) từ Decoder.
    Giúp triệt tiêu các pixel nhiễu (background/vân não mờ) và làm nổi bật vùng có khả năng là khối u.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        # Tương thích kích thước nếu g nhỏ hơn x
        if g1.shape != x.shape:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=512, patch_size=1, emb_size=768, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_size))
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=12, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + res
        
        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + res
        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=256):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.patch_embed = PatchEmbedding(in_channels=512, patch_size=1, emb_size=768, img_size=img_size//8)
        
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size=768, num_heads=12) for _ in range(4)]
        )
        
        self.conv_trans = nn.Conv2d(768, 512, kernel_size=3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        patches = self.patch_embed(x4)
        trans_out = self.transformer(patches) 
        
        B, N, C = trans_out.shape
        H = W = int(N ** 0.5)
        trans_out = trans_out.transpose(1, 2).contiguous().view(B, C, H, W)
        trans_out = self.conv_trans(trans_out)
        
        # Residual Connection over Transformer
        trans_out = trans_out + x4
        
        up1 = self.up1(trans_out)
        x3_att = self.ag1(g=up1, x=x3)
        cat1 = torch.cat([up1, x3_att], dim=1)
        dec1 = self.conv_up1(cat1)
        
        up2 = self.up2(dec1) 
        x2_att = self.ag2(g=up2, x=x2)
        cat2 = torch.cat([up2, x2_att], dim=1)
        dec2 = self.conv_up2(cat2)
        
        up3 = self.up3(dec2) 
        x1_att = self.ag3(g=up3, x=x1)
        cat3 = torch.cat([up3, x1_att], dim=1)
        dec3 = self.conv_up3(cat3)
        
        logits = self.outc(dec3)
        return logits
