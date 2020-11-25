# High Order Networks in PyTorch
These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch)

I plan to convert a few standard networks to high order an experiment with them here.
## Implemented Networks

resnet converted from torchvision.

## Cifar 10

```python
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=polynomial segments=1 n=3
```