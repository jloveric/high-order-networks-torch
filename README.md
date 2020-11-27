# High Order Networks in PyTorch
These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch)

Parameters that do work...
## Implemented Networks

resnet converted from torchvision.

## Cifar 100
The following examples use transformed torchvision resnet(s) using high order layers.

### Standard result for comparison - no high order layers RELU
After 100 epochs with command
```bash
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=standard segments=1 n=5 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy
```
```python
{'test_acc': tensor(0.4682, device='cuda:0'),
 'test_acc5': tensor(0.7399, device='cuda:0'),
 'test_loss': tensor(4.0737, device='cuda:0'),
 'train_acc': tensor(0.9531, device='cuda:0'),
 'train_acc5': tensor(1., device='cuda:0'),
 'train_loss': tensor(0.1139, device='cuda:0'),
 'val_acc': tensor(0.4421, device='cuda:0'),
 'val_acc5': tensor(0.7128, device='cuda:0'),
 'val_loss': tensor(4.0639, device='cuda:0')}

```

### Simple polynomial convolutional layers
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=6
```
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=4 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=8 model_name=resnet10
```
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=4 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=10 model_name=resnet18
```
## Fourier Series
resnet10
```python
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier segments=1 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet10 loss=cross_entropy
```
with accuracy (100 epochs)
```python
{'test_acc': tensor(0.4245, device='cuda:0'),
 'test_acc5': tensor(0.6917, device='cuda:0'),
 'test_loss': tensor(4.8819, device='cuda:0'),
 'train_acc': tensor(0.9688, device='cuda:0'),
 'train_acc5': tensor(1., device='cuda:0'),
 'train_loss': tensor(0.0680, device='cuda:0'),
 'val_acc': tensor(0.4036, device='cuda:0'),
 'val_acc5': tensor(0.6721, device='cuda:0'),
 'val_loss': tensor(5.0824, device='cuda:0')}

```
resnet18
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=fourier segments=1 n=5 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet10 loss=cross_entropy
```
### Piecewise polynomial convolutional layers - still working on stability here
Simple polynomial convolutional layers
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=0.5
```
### Discontinuous polynomial convolutional layers - still working on stability here
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=0.5
```