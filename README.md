# High Order Networks in PyTorch
These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch)

The following contain parameters that do work...  Weights represent actual function values and not slopes (as they do for RELU or linear) so some of the techniques used are slightly different.  For example, performing an average at the neuron is typically necessary for deep polynomial networks otherwise the you can get exploding values at the
edge outside the desired range.  In addition to averaging, the range of the function should be setm generally this is [-1, 1] so in these examples we choose scale=2.0 however to allow for 2 standard deviations scale=4.0 should be set.  On the other hand, deep fourier series networks are periodic so averaging is not necessary, but a scale should still be set, likely 2.0.
## Implemented Networks

resnet converted from torchvision.

## Cifar 100
The following examples use transformed torchvision resnet(s) using high order layers.

### Standard result for comparison - no high order layers RELU
After 100 epochs with command
```bash
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=standard segments=1 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy
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

```bash
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
```bash
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier segments=1 n=2 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy
```
```python
{'test_acc': tensor(0.4308, device='cuda:0'),
 'test_acc5': tensor(0.7070, device='cuda:0'),
 'test_loss': tensor(3.9088, device='cuda:0'),
 'train_acc': tensor(0.8906, device='cuda:0'),
 'train_acc5': tensor(1., device='cuda:0'),
 'train_loss': tensor(0.2658, device='cuda:0'),
 'val_acc': tensor(0.4050, device='cuda:0'),
 'val_acc5': tensor(0.6857, device='cuda:0'),
 'val_loss': tensor(4.0993, device='cuda:0')}

```


### Piecewise polynomial convolutional layers - still working on stability here
Simple polynomial convolutional layers
```python
python cifar100.py -m  max_epochs=100 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=1.0 learning_rate=1e-3 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=1 rescale_output=True
```
### Discontinuous polynomial convolutional layers - still working on stability here
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=0.5
```