# Hand-written digits recognition

## How to run

### Prerequisites

**Python: 3.7**

**Install requirements**

```bash
$ pip install -r requirements.txt
```

### Train model

Run all `notebook.ipynb`

### Run app

```bash
$ FLASK_ENV=development FLASK_APP=app flask run
```

## Model descriptions

1. Fully-connected neural network
2. Architecture

- Input layer: 784 neurons ~ 28x28 image
- Hidden layer: 256 neurons and 128 neurons, use ReLU function
- Output layer: 10 neurons, use Softmax and Cross-Entropy to evaluate loss

3. Use gradient descent and back-propagation
