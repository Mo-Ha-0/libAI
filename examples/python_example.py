"""
libAI Python API - quick example
Usage: python3.14 examples/python_example.py
"""

import libAI as la
import math
import random

# ── Generate spiral data ──
n = 300
x = la.Tensor.zeros([n, 2])
y = la.Tensor.zeros([n])
random.seed(42)
for i in range(n):
    angle = (i % 3) * 2 * math.pi / 3
    r = 0.5 + random.uniform(-1, 1) * 0.3
    x[i, 0] = r * math.cos(angle) + random.uniform(-1, 1) * 0.1
    x[i, 1] = r * math.sin(angle) + random.uniform(-1, 1) * 0.1
    y[i] = float(i % 3)

print(f"Data: {x.shape} {y.shape}")

# ── Build model (Sequential API) ──
model = la.Sequential()
model.add(la.Dense(2, 64))
model.add(la.ReLU())
model.add(la.Dense(64, 64))
model.add(la.ReLU())
model.add(la.Dense(64, 3))
model.compile(optimizer=la.Adam(0.001), loss='categorical_crossentropy')

# ── Train ──
model.fit(x, y, epochs=100, batch_size=32,
          validation_data=(x, y), verbose=True, print_every=20)

# ── Evaluate ──
acc = model.accuracy(x, y)
print(f"\nFinal accuracy: {acc:.4f}")

preds = model.predict(x)
print(f"Predictions shape: {preds.shape}")
