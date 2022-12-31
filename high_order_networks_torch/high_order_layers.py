from high_order_layers_torch.FunctionalConvolution import (
    PolynomialConvolution2d as PolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PiecewisePolynomialConvolution2d as PiecewisePolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import FourierConvolution2d

from high_order_layers_torch.PolynomialLayers import (
    PiecewiseDiscontinuousPolynomial,
    PiecewisePolynomial,
    Polynomial,
)
import copy
import torch.nn as nn


def pool_linear(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=in_channels, out_features=out_channels),
    )
