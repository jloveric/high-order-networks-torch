from high_order_layers_torch.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewisePolynomialConvolution2d as PiecewisePolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2dfrom high_order_layers_torch.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial

def high_order_convolution(layer_type: str, **kwargs) :
    if layer_type == "polynomial" :
        return PolyConv2d(**kwargs)
    elif layer_type == "piecewise" :
        return PiecewisePolyConv2d(**kwargs)
    elif layer_type == "discontinuous" :
        return PiecewiseDiscontinuousPolyConv2d(**kwargs)

def 