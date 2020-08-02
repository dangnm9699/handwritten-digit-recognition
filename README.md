# Neural Network Raw Python

*Build neural network in Python from scratch*
<br/>
*Demo from [Katz Sasaki](https://github.com/nai-kon/CNN-Digit-Recognition)*

### Python version
```
Python 3.7.6
```

## Install required packages
```git
pip install -r requirements.txt
```
## How to use
1. "Run all" **model.ipynb** to get **model.db**
2. Start server
```git
python server.py
```
3. Open browser, access to **0.0.0.0:5000**

## Model details
1. Fully-connected neural network
2. 
* Input layer: 784 neurons
* Hidden layer: 256 neurons and 128 neurons, use ReLU function
* Output layer: 10 neurons, use Softmax and Cross-Entropy to evaluate loss
3. Use gradient descent and back-propagation