import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from high_order_networks_torch.resnet import resnet10, resnet18
from pytorch_lightning.metrics.functional import accuracy
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torchvision.models as models


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar100_mean = (0.5071, 0.4867, 0.4408)


cifar10_std = (0.2470, 0.2435, 0.2616)
cifar100_std = (0.2675, 0.2565, 0.2761)

# Since we are using polynomials we really want all
# the inputs between -1 and 1 and not just those within
# 1 std deviation away.
rescale = 4.0
cifar100_2std = (0.2675*rescale, 0.2565*rescale, 0.2761*rescale)

cifar100_std = cifar100_2std

class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        self._data_dir = f"{hydra.utils.get_original_cwd()}/data"

        n = cfg.n
        self.n = cfg.n
        self._batch_size = cfg.batch_size
        self._layer_type = cfg.layer_type
        self._train_fraction = cfg.train_fraction
        self._num_workers = cfg.num_workers
        self._learning_rate = cfg.learning_rate
        segments = cfg.segments

        if cfg.layer_type=="standard" :
            self.model = models.resnet18(num_classes=100)
        else :
            self.model = resnet10(layer_type=self._layer_type,
                              n=self.n, segments=segments,num_classes=100, scale=4.0)

    def forward(self, x):
        ans = self.model(x)
        return ans

    def setup(self, stage):

        self._transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)])

        self._transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])

        num_train = int(self._train_fraction*40000)
        num_val = 10000
        num_extra = 40000-num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=True, download=True, transform=self._transform_train)

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train, [num_train, 10000, num_extra], generator=torch.Generator().manual_seed(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        return trainloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=False, download=True, transform=self._transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        return testloader

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f'{name}_loss', loss, prog_bar=True)
        self.log(f'{name}_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self._learning_rate)

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)
            module.weight.data = w


@hydra.main(config_name="./config/cifar100_config")
def run_cifar100(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    

    trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus, gradient_clip_val=cfg.gradient_clip_val)
    model = Net(cfg)
    #clipper = WeightClipper()
    #model.apply(clipper)

    trainer.fit(model)
    print('testing')
    trainer.test(model)
    print('finished testing')


if __name__ == "__main__":
    run_cifar100()
