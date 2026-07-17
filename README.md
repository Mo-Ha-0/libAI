# libAI

C++ neural network library with Python bindings. Parallel execution via OpenMP.

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
