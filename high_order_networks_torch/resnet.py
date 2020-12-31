"""
This is a direct copy of torchvision resnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
transformed to high order layers and convolutions.
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import BatchNorm2d as SpecialNorm
#from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
#from .high_order_layers import high_order_convolution
#from .high_order_layers import high_order_fully_connected_layer as fully_connected
from high_order_layers_torch.layers import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

"""
class LayerNorm2d(x) :
    def __init__(self, channels: int)
        self._channels = channels
        self._norm = LayerNorm()

    def __call__(self, x)
        xin = x.view(x.shape[0],-1)
        LayerNorm(xin)
"""


class PassThrough():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x

#SpecialNorm = PassThrough


def conv3x3(layer_type: str, n: int, in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1, segments: int = 1, scale: float = 4.0, **kwargs) -> nn.Conv2d:
    """3x3 convolution with padding"""
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    """
    return high_order_convolution_layers(layer_type=layer_type, n=n, segments=segments, in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                                           padding=dilation, groups=groups, bias=False, dilation=dilation, length=scale, **kwargs)


def conv1x1(layer_type: str, n: int, in_planes: int, out_planes: int, stride: int = 1, segments: int = 1, scale: float = 4.0, **kwargs) -> nn.Conv2d:
    """1x1 convolution"""
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    """
    return high_order_convolution_layers(layer_type=layer_type, n=n, segments=segments, in_channels=in_planes,
                                           out_channels=out_planes, kernel_size=1, stride=stride, bias=False, length=scale, **kwargs)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        layer_type: str,
        n: str,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        segments: int = 1,
        scale: float = 4.0,
        rescale_output: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        self._norm_layer = norm_layer
        if norm_layer is None:
            norm_layer = SpecialNorm
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(layer_type=layer_type, n=n, segments=segments,
                             in_planes=inplanes, out_planes=planes, stride=stride, scale=scale, rescale_output=rescale_output)
        self.bn1 = norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(layer_type=layer_type, n=n,
                             segments=segments, in_planes=planes, out_planes=planes, scale=scale, rescale_output=rescale_output)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.layer_type = layer_type
        self.n = n
        self.segments = segments

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #print('out1', torch.max(out))
        #out = self.bn1(out)
        #out = self.relu(out)
        out = self.conv2(out)
        #print('out2', torch.max(out))
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)
        #print('out3', torch.max(out))
        # exit(0)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        layer_type: str,
        n: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        segments: int = 1,
        scale: float = 4.0,
        rescale_output: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = SpecialNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(layer_type=layer_type, n=n,
                             segments=segments, in_planes=inplanes, out_planes=width, scale=scale, rescale_output=rescale_output)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(layer_type=layer_type, n=n, segments=segments, in_planes=width, out_planes=width,
                             stride=stride, groups=groups, dilation=dilation, scale=scale, rescale_output=rescale_output)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(layer_type=layer_type, n=n, segments=segments,
                             in_planes=width, out_planes=planes * self.expansion, scale=scale, rescale_output=rescale_output)
        self.bn3 = norm_layer(planes * self.expansion)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        layer_type: str = "polynomial",
        n: int = 3,
        segments: int = 1,
        scale: float = 4.0,
        rescale_planes: int = 1,  # rescale the original planes based on number of nodes
        rescale_output: bool = False,
        layer_by_layer: bool = True,
        periodicity: float = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = SpecialNorm
        self._norm_layer = norm_layer
        self.layer_type = layer_type
        self.n = n
        self.segments = segments
        self.inplanes = 64//rescale_planes
        self.dilation = 1
        self._scale = scale
        self._rescale_output = rescale_output
        self._training_layer = 0
        self._layer_by_layer = layer_by_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer0 = nn.Sequential(
            high_order_convolution_layers(
                layer_type=layer_type, n=n, segments=segments, in_channels=3,
                out_channels=self.inplanes, kernel_size=7, stride=2,
                padding=3, bias=False, length=scale, rescale_output=rescale_output),
            norm_layer(self.inplanes),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64//rescale_planes, layers[0])
        self.layer2 = self._make_layer(block, 128//rescale_planes, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256//rescale_planes, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512//rescale_planes, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        def pool_linear(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=in_channels, out_features=out_channels)
            )

        self.layer5 = pool_linear(in_channels=(512//rescale_planes) *
                                  block.expansion, out_channels=num_classes)

        self.model_layers = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        ]

        self.softmax = nn.Softmax(dim=1)

        # Now create a buncth of intermediate Linear layers
        self.layer0_intermediate = pool_linear(
            64//rescale_planes, num_classes)
        self.layer1_intermediate = pool_linear(
            64//rescale_planes, num_classes)
        self.layer2_intermediate = pool_linear(
            128//rescale_planes, num_classes)
        self.layer3_intermediate = pool_linear(
            256//rescale_planes, num_classes)

        self.intermediate_layers = [
            self.layer0_intermediate,
            self.layer1_intermediate,
            self.layer2_intermediate,
            self.layer3_intermediate,
            self.layer5
        ]

        # TODO: this may need to be commented out
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (SpecialNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def set_training_layer(self, layer):
        self._training_layer = layer

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(layer_type=self.layer_type, n=self.n, segments=self.segments,
                        in_planes=self.inplanes, out_planes=planes * block.expansion,
                        stride=stride, scale=self._scale, rescale_output=self._rescale_output),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_type=self.layer_type, n=self.n, segments=self.segments, inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, scale=self._scale, rescale_output=self._rescale_output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(layer_type=self.layer_type, n=self.n, segments=self.segments, inplanes=self.inplanes, planes=planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, scale=self._scale, rescale_output=self._rescale_output))

        return nn.Sequential(*layers)

    def _forward_layer_by_layer(self, x: Tensor) -> Tensor:

        # no back prop or gradients for the preceeding layers
        training_layer = min(self._training_layer,
                             len(self.intermediate_layers)-1)
        #print('inside here')
        y = None
        with torch.no_grad():
            for i in range(training_layer):
                x = self.model_layers[i](x)
                """
                if i == 0 :
                    y=self.intermediate_layers[i](x)
                elif i>0: 
                    y+=self.intermediate_layers[i](x)
                """
            if training_layer-1 >= 0:
                y = self.softmax(self.intermediate_layers[training_layer-1](x))
                #y = self.intermediate_layers[training_layer-1](x)
        x = self.model_layers[training_layer](x)

        # and use a linear layer for backprop
        x = self.intermediate_layers[training_layer](x)
        if y is not None:
            x += y

        # for i in range(training_layer) :
        #    x=x+self.intermediate_layers[i](x)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self._layer_by_layer is True:
            return self._forward_layer_by_layer(x)
        else:
            return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    """
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    """
    return model


def resnet_model(model_name: str, pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    model_dict = {
        "resnet10": resnet10,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "resnext50_32x4d": resnext50_32x4d,
        "resnext101_32x8d": resnext101_32x8d,
        "wide_resnet50_2": wide_resnet50_2,
        "wide_resnet101_2": wide_resnet101_2
    }

    return model_dict[model_name](pretrained=pretrained, progress=progress, **kwargs)


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
