import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl

class resnet18(pl.LightningModule):
    def __init__(self, lr=1e-3, is_gpu=True):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 사전학습된 ResNet18 로드
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = models.resnet18(weights=None)
        
        # 2. 입력 채널 수정 (mcPECharge, mcPEHitTime 2개 채널)
        n_input_channels = 2
        self.model.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. 출력 레이어 수정 (ES, 16N 2개 클래스)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        self.criterion = nn.CrossEntropyLoss()

    def get_current_device(self):
        """is_gpu 설정에 따라 적절한 장치 객체를 반환합니다."""
        if self.hparams.is_gpu:
            return torch_directml.device()
        return torch.device("cpu")

    def forward(self, x):
        # 현재 설정된 장치로 데이터 이동
        device = self.get_current_device()
        
        # 장치 타입이 다를 때만 전송 (예: CPU -> PrivateUse1)
        if x.device.type != device.type:
            x = x.to(device)
            
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        device = self.get_current_device()
        
        # 데이터와 모델을 설정된 장치로 명시적 이동
        inputs, targets = inputs.to(device), targets.to(device)
        self.model.to(device) 
        
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        device = self.get_current_device()
        
        inputs, targets = inputs.to(device), targets.to(device)
        self.model.to(device)
        
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)