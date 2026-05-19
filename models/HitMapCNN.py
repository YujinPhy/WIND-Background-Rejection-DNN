import torch
import torch.nn as nn
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
            nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)

class HitMapCNN(pl.LightningModule):
    def __init__(self, args, in_ch:int = 2,  image_h: int = 256, image_w: int = 256, n_classes: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(args)
        
        self.features = nn.Sequential(
            DoubleConvBlock(in_ch, 32, kernel_size=7, dropout=0.15),
            DoubleConvBlock(32, 32, kernel_size=5, dropout=0.20),
            DoubleConvBlock(32, 64, kernel_size=3, dropout=0.25),
            DoubleConvBlock(64, 64, kernel_size=3, dropout=0.25),
            DoubleConvBlock(64, 64, kernel_size=3, dropout=0.30),
        )

        # Dummy forward for a final Feature map size
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, image_h, image_w)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(128, n_classes),
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
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