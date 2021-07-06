# resnet: 2015, ILSVRC Winner


## resnet summary
### 1. Skip Connection
### 2. Bottleneck Design
### 3. Batch Normalization


## 1. Skip Connection

![image](https://user-images.githubusercontent.com/61177857/124564958-885c6680-de7c-11eb-8f12-f7e84a64b6aa.png)

Before Resnet had been invented, 
there is Gradient Vanising problem if layer is more deeper

Skip connection is solution of Gradient Vanishing

let ouput and input are H(x) and x for each, then the model is trained by F(x)=H(x)-x
this is reason why resnet is called Residual Networks

## 2. Bottleneck Design : more deeper but less calculation

![image](https://user-images.githubusercontent.com/61177857/124567193-be9ae580-de7e-11eb-9aa1-201d8c374cbc.png)

![image](https://user-images.githubusercontent.com/61177857/124567035-901d0a80-de7e-11eb-9c40-486f749dcd3f.png)

original residual block is consisted of two Conv(3x3)
but as layer is more deeper, there are many weight for calculate,
because of this reason, it spend long time
so new improved residual block is consisted of three Conv:
1x1 (256 =>64) => 3x3 (64 =>64) => 1x1 (64 =>256)

this spend less time than original (because of less calc)
but layer is more deeper than original 
(so more activation ftn avilable=>good)

## 3. Batch Normalization

why gradient vanishing occur?,
By some paper, internal covarrience shift cause gradient vanishing problem!

so we need to normalize Conv's output
this is reason why batch normalization is needed





