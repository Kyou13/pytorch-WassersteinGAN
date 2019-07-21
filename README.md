# pytorch-WassersteinGAN
## Description
WassersteinGANのpytorch実装

### WassersteinGAN
[papaer link](https://arxiv.org/abs/1701.07875)
- Wasserstein距離を最小化することで学習を行う
- 2つの確率密度関数の類似度を測る指標
  - KLDivergence
<img src="https://latex.codecogs.com/gif.latex?D_{K&space;L}(p&space;\|&space;q)=\int_{X}&space;p(x)&space;\log&space;\frac{p(x)}{q(x)}&space;d&space;x">
    - p,qに関して対称性がない
    - 確率密度分布に重なりがない場合∞

  - JSDivergence
<img src="https://latex.codecogs.com/gif.latex?D_{J&space;S}(p&space;\|&space;q)=\frac{1}{2}&space;D_{K&space;L}\left(p&space;\|&space;\frac{p&plus;q}{2}\right)&plus;\frac{1}{2}&space;D_{K&space;L}\left(q&space;\|&space;\frac{p&plus;q}{2}\right)">
    - p,qに関して対称性がある
    - 重なりが一致する場合に急に値が変動する

  - Wasserstein Distance
  <img src="https://latex.codecogs.com/gif.latex?W\left(p_{r},&space;p_{g}\right)=\inf&space;_{\gamma&space;\sim&space;\Pi\left(p_{r},&space;p_{g}\right)}&space;\mathbb{E}_{(x,&space;y)&space;\sim&space;\gamma}[|&space;|&space;x-y|&space;|]">
    - 値の変化がなめらか
    
- リプシッツ写像
<img src="https://latex.codecogs.com/gif.latex?d_{Y}\left(f(x),&space;f\left(x^{\prime}\right)\right)&space;\leq&space;\lambda&space;d_{X}\left(x,&space;x^{\prime}\right)&space;\quad\left(\forall&space;x,&space;\forall&space;x^{\prime}&space;\in&space;X\right)">
を満たすとき，<img src="https://latex.codecogs.com/gif.latex?f">は*リプシッツ***であるという

## Example
### loss
### Generated Image
- epochs: 200
  - batch size: 64

![generatedImage](https://github.com/Kyou13/pytorch-WassersteinGAN/blob/master/samples/mnist/fake_images-200.png)


## Requirement
- Python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Click

## Usage
### Training
```
$ pip install -r requirements.txt 
$ python main.py train [--dataset]
# training log saved at ./samples/fake_images-[epoch].png
```

### Generate
```
$ python main.py generate [--dataset]
# saved at ./samples/fake_images_%y%m%d%H%M%S.png
```
