# High Order Networks in PyTorch
These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch)

I plan to convert a few standard networks to high order an experiment with them here.
## Implemented Networks

resnet converted from torchvision.

## Cifar 100
The following examples use ResNet18

Simple polynomial convolutional layers
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial segments=1 n=3 batch_size=128 gradient_clip_val=0.5
```
Piecewise polynomial convolutional layers
Simple polynomial convolutional layers
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=0.5
```
Discontinuous polynomial convolutional layers
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=0.5
```