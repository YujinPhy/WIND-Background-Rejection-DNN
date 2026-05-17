import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size = 3, use_aux=True):
        super().__init__()
        pad = padding=k_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.use_aux = use_aux
        if self.use_aux:
            self.bn = nn.BatchNorm2d(out_ch)
            self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.use_aux:
            x = self.bn(x)
            x = self.dropout(x)
        return x

class DSNB_CNN(pl.LightningModule):
    def __init__(self, args, in_ch:int = 2, image_h:int = 256, image_w:int = 256, n_classes:int = 2,lr:float = 1e-3):
        super().__init__()
        self.save_hyperparameters(args)
        
        self.package1 = DoubleConvBlock(self.hparams.in_ch, 64, k_size=7, use_aux=True)
        self.package2 = DoubleConvBlock(64, 64, k_size=5, use_aux=True)
        self.package3 = DoubleConvBlock(64, 64, k_size=3, use_aux=True)
        self.package4 = DoubleConvBlock(64, 64, k_size=3, use_aux=True)
        self.package5 = DoubleConvBlock(64, 64, k_size=2, use_aux=False)
        self.package6 = DoubleConvBlock(64, 64, k_size=2, use_aux=False)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, self.hparams.image_h, self.hparams.image_w) 
            x = self.package1(dummy)
            x = self.package2(x)
            x = self.package3(x)
            x = self.package4(x)
            x = self.package5(x)
            x = self.package6(x)
            self.flatten_dim = x.view(1, -1).shape[1]
                
        self.fc1 = nn.Linear(self.flatten_dim, 384)
        self.fc2 = nn.Linear(384, 256)
        self.final_bn = nn.BatchNorm1d(256)
        self.final_dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(256, 2)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
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
        logits = self(inputs)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True, on_epoch=True)
        return {"loss": loss, "acc": acc, "logits": logits, "targets": targets}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)