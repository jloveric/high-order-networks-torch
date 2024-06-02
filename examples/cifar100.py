import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import Metric
import torch_optimizer as alt_optim
from high_order_networks_torch.resnet import resnet_model
from high_order_networks_torch.simple_conv import SimpleConv
from high_order_layers_torch.layers import MaxAbsNormalizationND
import math
from torcheval.metrics.functional import multiclass_accuracy as accuracy

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torchvision.models as models
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor 
from lion_pytorch import Lion

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar100_mean = (0.5071, 0.4867, 0.4408)


cifar10_std = (0.2470, 0.2435, 0.2616)
cifar100_std = (0.2675, 0.2565, 0.2761)

# Since we are using polynomials we really want all
# the inputs between -1 and 1 and not just those within
# 1 std deviation away.
rescale = 2.0
cifar100_2std = (0.2675 * rescale, 0.2565 * rescale, 0.2761 * rescale)

cifar100_std = cifar100_2std


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):

        super().__init__()
        self.save_hyperparameters(cfg)

        # TODO: As of this writing, I have somethigng messed up with manual
        # optimization, so setting to True!
        self.automatic_optimization = True

        self._cfg = cfg
        self._data_dir = f"{hydra.utils.get_original_cwd()}/data"

        n = cfg.n
        self.n = cfg.n
        self._batch_size = cfg.batch_size
        self._layer_type = cfg.layer_type
        self._train_fraction = cfg.train_fraction
        self._num_workers = cfg.num_workers
        self._learning_rate = cfg.learning_rate
        self._scale = cfg.scale
        segments = cfg.segments

        self._epochs_per_layer = cfg.epochs_per_layer

        if cfg.loss == "cross_entropy":
            self._loss = F.cross_entropy  # nn.CrossEntropyLoss
        elif cfg.loss == "mse":
            self._loss = nn.MSELoss
        else:
            raise ValueError(f"loss must be cross_entropy or mse, got {cfg.loss}")

        if cfg.model_name != "simple":
            if cfg.layer_type == "standard":
                self.model = getattr(models, cfg.model_name)(num_classes=100)
            else:
                self.model = resnet_model(
                    model_name=cfg.model_name,
                    layer_type=self._layer_type,
                    n=self.n,
                    segments=segments,
                    num_classes=100,
                    periodicity=cfg.periodicity,
                    scale=cfg.scale,
                    rescale_planes=cfg.rescale_planes,
                    layer_by_layer=cfg.layer_by_layer,
                    #norm_layer=MaxAbsNormalizationND,
                    weight_magnitude=1.0e-2
                )
        else:
            self.model = SimpleConv(cfg)

    def forward(self, x):
        ans = self.model(x)
        return ans

    def setup(self, stage):

        self._transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)]
        )

        self._transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(cifar100_mean, cifar100_std),
            ]
        )

        num_train = int(self._train_fraction * 40000)
        num_val = 10000
        num_extra = 40000 - num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir,
            train=True,
            download=True,
            transform=self._transform_train,
        )

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train,
            [num_train, 10000, num_extra],
            generator=torch.Generator().manual_seed(1),
        )

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization is False :
            opt = self.optimizers()
            opt.zero_grad()

        loss = self.eval_step(batch, batch_idx, 'train')

        
        if self.automatic_optimization is False :
            if self._cfg.optimizer in ["adahessian"]:
                self.manual_backward(loss, create_graph=True)
            else:
                self.manual_backward(loss, create_graph=False)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self._cfg.gradient_clip_val
            )
            opt.step()

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        return trainloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR100(
            root=self._data_dir,
            train=False,
            download=True,
            transform=self._transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
        return testloader

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        # print('x', x)
        logits = self(x)
        # print('logits.shape', logits.shape)
        # This should b F.cross_entropy
        # loss = F.nll_loss(logits, y)
        #print('logits', logits)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        loss = self._loss(logits, y)
        # print('logits', logits)
        # exit()
        # print('loss', loss)

        diff = torch.argmax(logits, dim=1) - y.flatten()
        acc = torch.where(diff == 0, 1, 0).sum() / len(diff)

        top1 = accuracy(logits, y, k=1)
        top5 = accuracy(logits, y, k=5)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        self.log(f"{name}_acc1", top1, prog_bar=True)
        self.log(f"{name}_acc5", top5, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self._cfg.optimizer == "adahessian":
            return alt_optim.Adahessian(
                self.model.parameters(),
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self._cfg.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        elif self._cfg.optimizer == "lion" :
            optimizer = Lion(self.parameters(), lr=self._learning_rate)
        elif self._cfg.optimizer == "lbfgs":
            optimizer = optim.LBFGS(self.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise ValueError(f"Optimizer {self._cfg.optimizer} not recognized")
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._cfg.scheduler_patience,
            factor=self._cfg.scheduler_factor,
            verbose=True,
        )

        scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": True,
                "monitor": "train_loss",
            }

        return [optimizer], [scheduler]

    def on_before_zero_grad(self, *args, **kwargs):
        # clamp the weights here
        # This could be bad news for the linear layer.
        if self._cfg.clamp_weights is True:
            for param in self.model.parameters():
                w = param.data
                w = w.clamp(-1.0, 1.0)


@hydra.main(config_path="../config", config_name="cifar100_config")
def run_cifar100(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        divergence_threshold=1.0e4,
        verbose=False,
        check_finite=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=cfg.max_epochs, accelerator="gpu", callbacks=[lr_monitor]
    )
    model = Net(cfg)
    # clipper = WeightClipper()
    # model.apply(clipper)

    """
    for param in model.parameters():
        print(param.data)
    """

    trainer.fit(model)
    print("testing")
    result = trainer.test(model)

    print("result", result)
    print("finished testing")
    print("best check_point", trainer.checkpoint_callback.best_model_path)

    # We want to optimize on total error so that it is
    # independent of loss function etc...
    print("error fraction", 1 - result[0]["test_acc"])
    return 1.0 - result[0]["test_acc"]


if __name__ == "__main__":
    run_cifar100()
