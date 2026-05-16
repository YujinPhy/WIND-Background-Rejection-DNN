import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl

class resnet18(pl.LightningModule):
    def __init__(self, args, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(args)
        
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)