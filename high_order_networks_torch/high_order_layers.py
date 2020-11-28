from high_order_layers_torch.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewisePolynomialConvolution2d as PiecewisePolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d
from high_order_layers_torch.FunctionalConvolution import FourierConvolution2d

from high_order_layers_torch.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial
import copy

def high_order_convolution(layer_type: str, **inkwargs):
    kwargs = copy.deepcopy(inkwargs)
    
    if layer_type == "polynomial":
        if "segments" in kwargs : 
            del kwargs['segments']
        return PolyConv2d(**kwargs)
    elif layer_type == "piecewise":
        return PiecewisePolyConv2d(**kwargs)
    elif layer_type == "discontinuous":
        return PiecewiseDiscontinuousPolyConv2d(**kwargs)
    elif layer_type == "fourier" :
        if "segments" in kwargs : 
            del kwargs['segments']
        return FourierConvolution2d(**kwargs)

def high_order_fully_connected_layer(layer_type: str, **kwargs):
    if layer_type == "polynomial":
        if "segments" in kwargs : 
            del kwargs['segments']
        return Polynomial(**kwargs)
    elif layer_type == "piecewise":
        return PiecewisePolynomial(**kwargs)
    elif layer_type == "discontinuous":
        return PiecewiseDiscontinuousPolynomial(**kwargs)