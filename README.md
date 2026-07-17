# The goal of building the library

I wanted to experiment with building a low-level framework so bad — to the point where I literally started without any planning. I ported the code from my existing GitHub repository (BeyondersTensorflow) from Python to C++, and to work more closely with low-level code, I implemented a tensor type.

In the future, **<font size="+1">If Allah wills</font>**, I will attempt to build a comprehensive, professional-grade framework using C++ and, most likely, Rust. <font size="+2">🫡</font>

### So, if you explore this framework and find it useful, please wish me good luck. <font size="+2">🥹</font>

# libAI

C++ neural network library with Python bindings. Parallel execution via OpenMP.

#### the framework is not available in pip now, so to try it you can run:

```
python -m build

pip install dist/libai-1.0.0-cp312-cp312-linux_x86_64.whl
```

## Quick start

```python
import libAI as la

x = la.Tensor.random_normal([300, 2])
y = la.Tensor.zeros([300])
for i in range(300):
    y[i] = float(i % 3)

model = la.Sequential()
model.add(la.Dense(2, 64))
model.add(la.ReLU())
model.add(la.Dense(64, 64))
model.add(la.ReLU())
model.add(la.Dense(64, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x, y, epochs=100, batch_size=32, verbose=True, print_every=20)
print(f"Accuracy: {model.accuracy(x, y)}")
```

## Features

- Dense, ReLU, Sigmoid, Tanh, Linear, Dropout, BatchNormalization
- SGD, Momentum, AdaGrad, Adam optimizers
- MSE, SoftmaxCrossEntropy losses
- Sequential API (Keras-like)
- OpenMP parallelism for all compute-intensive operations
