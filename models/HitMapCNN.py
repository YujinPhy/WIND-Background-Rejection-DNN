import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.20):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # GroupNorm의 num_groups가 num_channels의 약수여야 하므로 8로 설정 시 out_ch가 8의 배수인지 확인 필요 (32, 64 모두 만족)
            nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class HitMapLightningModel(pl.LightningModule):
    def __init__(self, args, image_h: int = 256, image_w: int = 256, n_classes: int = 2, lr: float = 1e-3):
        super().__init__()
        # 하이퍼파라미터 저장 (self.hparams.lr 등으로 접근 가능)
        self.save_hyperparameters(args)
        
        # Sparse한 PMT hit map 특성에 맞춘 Regularized CNN 백본
        self.features = nn.Sequential(
            DoubleConvBlock(2, 32, kernel_size=7, dropout=0.15),
            DoubleConvBlock(32, 32, kernel_size=5, dropout=0.20),
            DoubleConvBlock(32, 64, kernel_size=3, dropout=0.25),
            DoubleConvBlock(64, 64, kernel_size=3, dropout=0.25),
            DoubleConvBlock(64, 64, kernel_size=3, dropout=0.30),
        )

        # Feature map의 최종 크기 계산을 위한 Dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 2, image_h, image_w)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(128, n_classes),
        )
        
        # 2. 손실 함수 정의 (내부에서 선언하거나 외부에서 주입)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        # on_step과 on_epoch를 모두 켜서 스텝별/에폭별 로그 기록
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        # PyTorch Lightning 2.0+ 버전에서는 sync_dist=True를 넣어주면 분산 학습(Multi-GPU) 시 안전합니다.
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # self.hparams.lr을 통해 __init__에서 입력받은 인자에 안전하게 접근합니다.
        return optim.Adam(self.parameters(), lr=self.hparams.lr)