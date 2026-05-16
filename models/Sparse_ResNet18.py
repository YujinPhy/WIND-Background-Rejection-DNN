import torch
import torch.nn as nn
from torchvision.models import resnet18
import pytorch_lightning as pl

class sparse_resnet18(pl.LightningModule):
    def __init__(self, lr=1e-4, is_gpu=True):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 기본 ResNet18 구조 가져오기 (사전학습 없이)
        self.model = resnet18(weights=None)
        
        # 2. 입력을 (2, 91, 142)에 맞게 첫 레이어 수정
        # Sparse 데이터 대응: 원래 7x7 stride 2 대신 3x3 stride 1로 시작하여 정보 손실 방지
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 3. Sparse 데이터 대응: 첫 MaxPool의 영향력 조절 (선택 사항)
        # 정보가 너무 부족하면 초기 MaxPool이 신호를 지워버릴 수 있음. 일단은 유지하되 stride를 조절 가능.
        self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 

        # 4. 출력층을 이진 분류(2 클래스)로 수정
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3), # 과적합 방지를 위해 드롭아웃 추가
            nn.Linear(num_ftrs, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # Sparse 데이터는 초기 학습률을 조금 낮게 가져가는 것이 안정적입니다.
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)