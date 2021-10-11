# High Order Networks in PyTorch
SANDBOX Work in Progress for experimenting with High Order Layers...

These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch)

The following contain parameters that do work, though not necessarily very well...  Weights represent actual function values and not slopes (as they do for RELU or linear) so some of the techniques used are slightly different.  For example, performing an average at the neuron is typically necessary for deep polynomial networks otherwise the you can get exploding values at the
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

### Polynomial convolutional layers

```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=6 layer_by_layer=False
```
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=4 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=8 model_name=resnet10 layer_by_layer=False
```
```python
python cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial n=4 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=10 model_name=resnet18 layer_by_layer=False
```
## Fourier Series

```bash
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier segments=1 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet10 loss=cross_entropy layer_by_layer=False
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
python cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier segments=1 n=2 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy layer_by_layer=False
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


### Piecewise polynomial convolutional layers
Simple polynomial convolutional layers
```python
python cifar100.py -m  max_epochs=100 train_fraction=1.0 layer_type=piecewise segments=2 n=3 batch_size=128 gradient_clip_val=1.0 learning_rate=1e-3 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=1 rescale_output=True layer_by_layer=False
```
```python
{'test_acc': tensor(0.3771, device='cuda:0'),
 'test_acc5': tensor(0.6626, device='cuda:0'),
 'test_loss': tensor(3.8744, device='cuda:0'),
 'train_acc': tensor(0.6719, device='cuda:0'),
 'train_acc5': tensor(0.9219, device='cuda:0'),
 'train_loss': tensor(0.9543, device='cuda:0'),
 'val_acc': tensor(0.3647, device='cuda:0'),
 'val_acc5': tensor(0.6532, device='cuda:0'),
 'val_loss': tensor(3.4837, device='cuda:0')}
```
### Discontinuous polynomial convolutional layers
```python
python cifar100.py -m  max_epochs=100 train_fraction=1.0 layer_type=discontinuous segments=2 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=3 rescale_output=True layer_by_layer=False
```
```python
{'test_acc': tensor(0.3587, device='cuda:0'),
 'test_acc5': tensor(0.6414, device='cuda:0'),
 'test_loss': tensor(4.0509, device='cuda:0'),
 'train_acc': tensor(0.7031, device='cuda:0'),
 'train_acc5': tensor(0.9375, device='cuda:0'),
 'train_loss': tensor(1.0352, device='cuda:0'),
 'val_acc': tensor(0.3462, device='cuda:0'),
 'val_acc5': tensor(0.6307, device='cuda:0'),
 'val_loss': tensor(3.7229, device='cuda:0')}
```

### Running the small convolutional neural network for experimentation.
model_name is "simple"
```
python cifar100.py -m max_epochs=60 train_fraction=1.0 layer_type=polynomial segments=1 n=6 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=simple loss=cross_entropy rescale_planes=2 rescale_output=True layer_by_layer=True epochs_per_layer=20
```

## Running transformer example
Train
```
python wikitext_transformer.py fraction=1.0
```
predict
```
python wikitext_predict.py path=~/high-order-networks-pytorch/outputs/2021-02-14/15-09-33/lightning_logs/version_0 text="A bit of text"
```

## Year later
best parameters with resnet10
```
Best parameters: layer_type=continuous segments=2 n=2 optimizer=adahessian gradient_clip_val=1 max_epochs=20
```