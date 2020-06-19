# Neural Network Raw Python

*Build neural network myself with python*

*#by-myself #MNIST #recognition #project-2 #back-propagation #stochastic-gradient-descent*

## **MNIST**

### Download MNIST data
* Visit this website: http://yann.lecun.com/exdb/mnist/
* Download 4 gzip files
* Extract files
* Rename files
> Example: Rename **t10k-images.idx3-ubyte** to **t10k-images-idx3-ubyte**
### Import MNIST
> ```python
> from mnist import MNIST
> mnist = MNIST('./samples/')
> images_train, labels_train = mnist.load_training()
> images_test, labels_test = mnist.load_testing()
> ```
### Data normalization
*Resize input to vector (784x1)*\
*Data value: [0,255] to [0,1]*
> ```python
> x_train = np.array(images_train)
> x_test = np.array(images_test)
> x_train = np.reshape(x_train, (x_train.shape[0],784,1))
> x_test = np.reshape(x_test, (x_test.shape[0],784,1))
> x_train, x_test = x_train/255.0, x_test/255.0
>```
### One-hot Encoding
*5 to [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]*\
*Resize label to vector (10x1)*
> ```python
> y_train = []
> y_test = []
> for label in labels_train:
>     arr = np.zeros((10,1))
>     arr[label] = 1
>     y_train.append(arr)
> for label in labels_test:
>     arr = np.zeros((10,1))
>     arr[label] = 1
>     y_test.append(arr)
> y_train = np.array(y_train)
> y_test = np.array(y_test)
> ```
## **Model**
### I build a fully-connected neural network with:
* 784 neural in input layer because image's size is 28x28
* 256 neural in 1st hidden layer
* 128 neural in 2nd hidden layer
* 10 neural in output layer because we have 10 digits to classify

### This neural network use **ReLU** and **Softmax** functions:
> **ReLU**
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=max(0,x)" title="ReLU function"/> <br/>
> 
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;f'(x)=\left\{\begin{matrix}1&x>0\\0&x\leq0\end{matrix}\right." title="ReLU derivative"/>

>**Softmax** (*stable*)
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;f_i(x)=\frac{e^{x_i-x_{max}}}{\sum_{j}^{}e^{x_j-x_{max}}}" title="Softmax function"/> <br/>

### Use categorical cross-entropy to evaluate loss value:
>**Categorical Cross-Entropy**
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;Loss=-\sum_{i=1}^{C}y_i\cdot\hspace{0mm}log(\widehat{y_i})" title="Categorical Cross-Entropy"/> <br/>
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;with\hspace{2mm}C\hspace{2mm}is\hspace{2mm}number\hspace{2mm}of\hspace{2mm}classes" title="with-C-is-number-of-classes"/>

### Weights initialization techniques
>**Reference: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78**
>
> ```python
> random_rate = np.sqrt(2/(layer[i+1]+layer[i]))
> w_i = np.random.randn(layer[i+1], layer[i])*random_rate
> ```
> ```python np.random.randn``` use "standard normal distribution" with 
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu=0\hspace{2mm}and\hspace{2mm}\sigma=1" title="mean=0-and-variance=1"/>
>
>
> Weights initialization
>
> <img src="https://latex.codecogs.com/svg.latex?\Large&space;W^{[l\hspace{0.5mm}]}=np.random.randn(size^{[l+1\hspace{0.5mm}]},size^{[l]})*\sqrt{\frac{2}{size^{[l+1\hspace{0.5mm}]}+size^{[l]}}}" title="weights-initialization-technique"/>

## How to use
"Run all" train.ipynb, this takes about 25 minutes (if you use Ubuntu)
