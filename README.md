# High Order Networks in PyTorch

These are high order networks using the high order layers defined in the repo [here](https://github.com/jloveric/high-order-layers-torch).  This is an unfinished experiment where I'm trying to get the resnet performance using high order layers to at least match the standard approach.  So far fourier series layers do decent, I can also get results with piecewise polynomial layers - random constant initialization
seems to be key vs kaiming which introduces too much oscillation to begin with. However, I can rapidly get perfect accuracy on the training set with quadratic or higher (either piecewise or non-piecewsie polynomials), I believe I should actually be trying a much smaller model than resnet10 since the high order networks should require
far fewer parameters. Alternatively, try this with a much larger dataset.

## Notes

It seems the important thing is random constant initialization, normalization - I'm using infinity norm which probably isn't the best but I like because it keeps values in the correct range for polynomial problems and then lion. I'm getting overfitting except the linear case which is slow to converge. Non-linearity is introduced by the normalization so they do much better than I expected (but not well aware with cifar100 training). Piecewise polynomial training is much slower than polynomial so I need to speed up the algorithm.

## Implemented Networks

resnet converted from torchvision.

## Cifar 100
The following examples use transformed torchvision resnet(s) using high order layers.

### Standard result for comparison - no high order layers RELU
After 100 epochs with command
```bash
python examples/cifar100.py max_epochs=100 train_fraction=1.0 layer_type=standard segments=1 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy
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
python examples/cifar100.py max_epochs=1000 train_fraction=1.0 layer_type=polynomial2d n=3 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-4 scale=6 layer_by_layer=False optimizer=lion model_name=resnet10
```
with test results, the training top5 is basically 100% accurate. Highly overfit.
```
[{'test_loss': 33.931854248046875, 'test_acc': 0.28209999203681946, 'test_acc1': 0.28209999203681946, 'test_acc5': 0.49720001220703125}]
```


```python
python examples/cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial2d n=4 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-4 scale=8 model_name=resnet10 layer_by_layer=False optimizer=lion
```
```python
python examples/cifar100.py max_epochs=20 train_fraction=1.0 layer_type=polynomial2d n=4 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-4 scale=10 model_name=resnet18 layer_by_layer=False
```
Linear case with n=2 on resnet10
```
python examples/cifar100.py max_epochs=1000 train_fraction=1.0 layer_type=polynomial2d n=2 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-4 scale=6 layer_by_layer=False optimizer=lion
```
with results
```
[{'test_loss': 2.6638987064361572, 'test_acc': 0.3310000002384186, 'test_acc1': 0.3310000002384186, 'test_acc5': 0.6402000188827515}]
```
with resnet18 using linear
```
python examples/cifar100.py max_epochs=1000 train_fraction=1.0 layer_type=polynomial2d n=2 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-4 scale=2 layer_by_layer=False optimizer=lion model_name=resnet18
```
and results (all using maxabs normalization - infinity norm noramlization)
```
[{'test_loss': 2.2048048973083496, 'test_acc': 0.4456000030040741, 'test_acc1': 0.4456000030040741, 'test_acc5': 0.7317000031471252}]
```
## Fourier Series

```bash
python examples/cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier2d segments=1 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet10 loss=cross_entropy layer_by_layer=False
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
python examples/cifar100.py max_epochs=100 train_fraction=1.0 layer_type=fourier2d segments=1 n=2 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=resnet18 loss=cross_entropy layer_by_layer=False
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
python examples/cifar100.py -m  max_epochs=100 train_fraction=1.0 layer_type=continuous2d segments=2 n=3 batch_size=128 gradient_clip_val=1.0 learning_rate=1e-3 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=1 rescale_output=True layer_by_layer=False
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
Couple years later, no clipping using constant initialization 
```
python examples/cifar100.py max_epochs=100 train_fraction=1.0 layer_type=continuous2d segments=2 n=3 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-5 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=1 layer_by_layer=False optimizer=lion
```
results
```
result [{'test_loss': 4.571002006530762, 'test_acc': 0.34369999170303345, 'test_acc1': 0.34369999170303345, 'test_acc5': 0.6175000071525574}]
```
and with a resnet18
```
python examples/cifar100.py max_epochs=100 train_fraction=1.0 layer_type=continuous2d segments=2 n=3 batch_size=1024 gradient_clip_val=0.0 learning_rate=1e-5 scale=2.0 model_name=resnet18 loss=cross_entropy rescale_planes=1 layer_by_layer=False optimizer=lion
```
with result
```
[{'test_loss': 4.950081825256348, 'test_acc': 0.29280000925064087, 'test_acc1': 0.29280000925064087, 'test_acc5': 0.5515000224113464}]
```
the larger network resnet18 overfits faster than the smaller network (better training score with lower generalization), not surprising

### Discontinuous polynomial convolutional layers
```python
python cifar100.py -m  max_epochs=100 train_fraction=1.0 layer_type=discontinuous2d segments=2 n=3 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=2.0 model_name=resnet10 loss=cross_entropy rescale_planes=3 rescale_output=True layer_by_layer=False
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
python cifar100.py -m max_epochs=60 train_fraction=1.0 layer_type=polynomial2d segments=1 n=6 batch_size=128 gradient_clip_val=0.0 learning_rate=1e-3 scale=4.0 model_name=simple loss=cross_entropy rescale_planes=2 rescale_output=True layer_by_layer=True epochs_per_layer=20
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
