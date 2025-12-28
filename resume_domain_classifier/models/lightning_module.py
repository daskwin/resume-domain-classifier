from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class TfidfLinearLightning(LightningModule):
    def __init__(self, *, input_dim: int, num_classes: int, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, targets = batch
        logits = self(features)
        loss = self.loss_fn(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        features, targets = batch
        logits = self(features)
        loss = self.loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1_macro", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
