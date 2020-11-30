import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from high_order_layers_torch.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewisePolynomialConvolution2d as PiecewisePolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d
from high_order_layers_torch.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from .high_order_layers import *
from torch import Tensor
from torch.nn import BatchNorm2d

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class SimpleConv(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)

        self._cfg = cfg
        self._data_dir = f"{hydra.utils.get_original_cwd()}/data"
        n = cfg.n
        self._layer_type = cfg.layer_type
        layer_type = cfg.layer_type
        segments = cfg.segments
        length = cfg.scale
        num_classes = cfg.num_classes
        self._layer_by_layer = cfg.layer_by_layer
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Sequential(high_order_convolution(
            layer_type=layer_type,
            n=n,
            segments=segments,
            length=length,
            in_channels=3,
            out_channels=16,
            rescale_output=2.0,
            kernel_size=5,
        ), BatchNorm2d(16),self.pool)

        self.conv2 = nn.Sequential(high_order_convolution(
            layer_type=layer_type,
            n=n,
            segments=segments,
            length=length,
            in_channels=16,
            out_channels=32,
            rescale_output=2.0,
            kernel_size=5,
        ), BatchNorm2d(32), self.pool)

        w1 = 28-4
        w2 = (w1//2)-4
        c1 = 6
        c2 = 16

        #self.pool = nn.MaxPool2d(2, 2)
        #self.avg_pool = nn.AdaptiveAvgPoofrom torch import Tensor

        self.layer0_intermediate = pool_linear(
            16, num_classes)
        self.layer_output = pool_linear(
            32, num_classes)

        self.model_layers = [
            self.conv1,
            self.conv2,
            self.layer_output
        ]

        self.intermediate_layers = [
            self.layer0_intermediate,
            self.layer_output
        ]

        self._training_layer = 0

    def set_training_layer(self, training_layer):
        self._training_layer = training_layer

    def forward(self, x):

        x = self.avg_pool(self.conv1(x))
        x = self.avg_pool(self.conv2(x))
        x = x.reshape(-1, 16)
        x = self.fc1(x)
        return x

    def _forward_layer_by_layer(self, x: Tensor) -> Tensor:
        #print('training layer by layer')
        # no back prop or gradients for the preceeding layers
        training_layer = min(self._training_layer,
                             len(self.intermediate_layers)-1)

        with torch.no_grad():
            for i in range(training_layer):
                x = self.model_layers[i](x)

        x = self.model_layers[training_layer](x)

        # and use a linear layer for backprop
        x = self.intermediate_layers[training_layer](x)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        for layer in self.module_layers:
            x = layer(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self._layer_by_layer is True:
            return self._forward_layer_by_layer(x)
        else:
            return self._forward_impl(x)
