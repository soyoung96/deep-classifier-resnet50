# resnet: 2015, ILSVRC Winner


## resnet summary
### 1. Skip Connection
### 2. Bottleneck Design
### 3. Batch Normalization


## 1. Skip Connection

![image](https://user-images.githubusercontent.com/61177857/124564958-885c6680-de7c-11eb-8f12-f7e84a64b6aa.png)

Before Resnet had been invented, 
there is Gradient Vanishing problem if layer is more deeper

Skip connection is solution of Gradient Vanishing problem

let output and input are H(x) and x for each, then the model is trained by F(x)=H(x)-x
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

why does gradient vanishing occurr?
By some paper, internal covarrience shift cause gradient vanishing problem!

so we need to normalize Conv's output
this is reason why batch normalization is needed

![image](https://user-images.githubusercontent.com/61177857/124691304-98288900-df16-11eb-8464-4647e16f898a.png)

then...
why do we need to add additionalγ,β standard Normalization?
because of two problem!

1. bias is disappear
2. standard Normalization lose non-linear feature from activation funtion

### reason why 1. bias is disappear
y = Wx+b (W:weight,x:input,b:bias)

E(y) = WE(x)+b
y-E(y) = W(x-E(x)) => bias out!

### reason why 2. standard Normalization lose non-linear feature from activation funtion

about sigmoid ftn,

![image](https://user-images.githubusercontent.com/61177857/124692319-5d275500-df18-11eb-8727-8d9731f9873a.png)

if we apply standard Normalization then output will pass by sigmoid ftn.
but standard Normalization's result is focused on middle by 0
==> this cause to lose non-linear problem

https://eehoeskrap.tistory.com/430






