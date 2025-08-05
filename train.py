from data import MNISTDataModule
from model import GAN
import pytorch_lightning as L
import torch

torch.set_float32_matmul_precision("high")

dm = MNISTDataModule()
model = GAN(1, 28, 28)

trainer = L.Trainer(accelerator="auto", devices=1, max_epochs=5)
trainer.fit(model, dm)
