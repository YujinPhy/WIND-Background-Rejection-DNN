import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl

class ResNet18(pl.LightningModule):
    def __init__(self, args, in_ch:int = 2, n_classes: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Not a pre-training model
        self.model = models.resnet18(weights=None)
        
        # Modify input layer
        self.model.conv1 = nn.Conv2d(self.hparams.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify output layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

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