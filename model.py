import os
from typing import List, Tuple

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
NUM_WORKERS = int(os.cpu_count() / 2)
PATH_DATASETS = os.path.join(os.path.dirname(__file__), "data")


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.img_shape = img_shape

        def block(in_chan, out_chan, normalize=True):
            layers = [nn.Linear(in_chan, out_chan)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_chan))
            layers.append(nn.LeakyReLU())
            return layers

        input_dim = latent_dim + 10

        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 256),
            *block(256, np.prod(img_shape)),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z, F.one_hot(labels, 10)], dim=1)
        img: torch.Tensor = self.model(z)
        img = img.reshape(z.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: List[int]) -> None:
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape) + 10, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        img = img.reshape(img.size(0), np.prod(self.img_shape))
        img = torch.cat([img, F.one_hot(labels, 10)], dim=1)
        validity = self.model(img)
        return validity  # (batch_size, 1)


class GAN(L.LightningModule):
    def __init__(
        self,
        channels: int,
        width: int,
        height: int,
        latent_dim: int = 100,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        data_shape = (channels, width, height)
        self.generator = Generator(self.hparams.latent_dim, data_shape)
        self.discriminator = Discriminator(data_shape)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            batch_size = z.size(0)
            labels = torch.zeros(batch_size, dtype=torch.int64, device=z.device)
        return self.generator(z, labels)

    def adversarial_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        imgs, labels = batch
        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(imgs.size(0), self.hparams.latent_dim)
        z = z.type_as(imgs)

        self.toggle_optimizer(optimizer_g)
        self.generated_img = self(z, labels)

        sample_imgs = self.generated_img[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        valid = torch.ones(imgs.size(0), 1).type_as(imgs)

        g_loss = self.adversarial_loss(
            self.discriminator(self.generated_img, labels), valid
        )
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(imgs.size(0), 1).type_as(imgs)
        fake = torch.zeros(imgs.size(0), 1).type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_img.detach(), labels), fake
        )

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        pass

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        return [optimizer_g, optimizer_d], []

    def on_validation_epoch_end(self) -> None:
        validation_z = torch.randn(8, self.hparams.latent_dim)
        z = validation_z.type_as(self.generator.model[0].weight)

        labels = torch.arange(8).to(z.device)
        sample_imgs = self(z, labels)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
        label_str = "Labels: " + ", ".join(str(label.item()) for label in labels)
        self.logger.experiment.add_text(
            "validation/generated_labels", label_str, self.current_epoch
        )
        self.logger.experiment.add_image(
            "validation/generated_images", grid, self.current_epoch
        )
