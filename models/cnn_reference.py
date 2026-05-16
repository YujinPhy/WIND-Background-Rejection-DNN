import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch_directml  # DirectML 사용

class DoubleConvPackage(nn.Module):
    def __init__(self, in_channels, out_channels, use_aux=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.use_aux = use_aux
        if self.use_aux:
            self.bn = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.use_aux:
            x = self.bn(x)
            x = self.dropout(x)
        return x

class cnn_reference(pl.LightningModule):
    def __init__(self, lr=1e-3, is_gpu=True):
        super().__init__()
        self.save_hyperparameters()
        
        # 6개의 Double Convolutional Packages
        self.package1 = DoubleConvPackage(2, 64, use_aux=True)
        self.package2 = DoubleConvPackage(64, 64, use_aux=True)
        self.package3 = DoubleConvPackage(64, 64, use_aux=True)
        self.package4 = DoubleConvPackage(64, 64, use_aux=True)
        self.package5 = DoubleConvPackage(64, 64, use_aux=False)
        self.package6 = DoubleConvPackage(64, 64, use_aux=False)
        
        self.flatten_dim = 64 * 1 * 2 
        
        self.fc1 = nn.Linear(self.flatten_dim, 384)
        self.fc2 = nn.Linear(384, 256)
        self.final_bn = nn.BatchNorm1d(256)
        self.final_dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(256, 2)
        
        self.criterion = nn.CrossEntropyLoss()

    def get_current_device(self):
        """DirectML 장치를 명시적으로 반환"""
        if self.hparams.is_gpu:
            return torch_directml.device()
        return torch.device("cpu")

    def forward(self, x):
        device = self.get_current_device()
        
        # [중요] 모델 가중치와 입력 데이터가 DML 장치에 있는지 매번 확인 및 강제 이동
        if next(self.parameters()).device.type != device.type:
            self.to(device)
        if x.device.type != device.type:
            x = x.to(device)
            
        x = self.package1(x)
        x = self.package2(x)
        x = self.package3(x)
        x = self.package4(x)
        x = self.package5(x)
        x = self.package6(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final_bn(x)
        x = self.final_dropout(x)
        return self.output(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        device = self.get_current_device()
        
        # 데이터를 DirectML 장치로 명시적 이송
        inputs, targets = inputs.to(device), targets.to(device)
        
        # forward 호출 시 내부에서 모델(self.to(device))도 함께 이동함
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        device = self.get_current_device()
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # 최적화기에도 DML 장치에 있는 파라미터가 전달됨
        return optim.Adam(self.parameters(), lr=self.hparams.lr)