"""
libAI - C++ neural network library with direct Python bindings
"""

from . import _libAI_core as _C

Tensor = _C.Tensor
Dense = _C.Dense
Linear = _C.Linear
ReLU = _C.ReLU
Sigmoid = _C.Sigmoid
Tanh = _C.Tanh
Dropout = _C.Dropout
BatchNormalization = _C.BatchNormalization
MeanSquaredError = _C.MeanSquaredError
SoftmaxCrossEntropy = _C.SoftmaxCrossEntropy

SGD = _C.SGD
Momentum = _C.Momentum
AdaGrad = _C.AdaGrad
Adam = _C.Adam

NeuralNetwork = _C.NeuralNetwork
Trainer = _C.Trainer
HyperparameterTuner = _C.HyperparameterTuner

__version__ = "1.0.0"


class Sequential:
    """A simple sequential model for building neural networks."""
    def __init__(self):
        """Initialize a new Sequential model.
        This model allows you to add layers sequentially and compile the model with an optimizer and loss function.
        """
        self._net = _C.NeuralNetwork()
        self._layers = []
        self._optimizer = None
        self._loss = None
        self._trainer = None

    def add(self, layer):
        """Add a layer to the model.
        Args:
            layer: An instance of a layer (e.g., Dense, ReLU, etc.) to be added to the model.
        """
        self._net.add_layer(layer)
        self._layers.append(layer)

    def compile(self, optimizer=None, loss=None):
        """Compile the model with the specified optimizer and loss function.
        Args:
            optimizer: An instance of an optimizer (e.g., SGD, Adam) or a string representing the optimizer name.
            loss: An instance of a loss function (e.g., MeanSquaredError, SoftmaxCrossEntropy) or a string representing the loss function name.
        """
        if isinstance(optimizer, str):
            opt_map = {'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad, 'adam': Adam}
            opt_lower = optimizer.lower()
            if opt_lower in opt_map:
                self._optimizer = opt_map[opt_lower]()
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self._optimizer = optimizer

        if isinstance(loss, str):
            loss_map = {
                'mse': MeanSquaredError,
                'mean_squared_error': MeanSquaredError,
                'categorical_crossentropy': SoftmaxCrossEntropy,
                'crossentropy': SoftmaxCrossEntropy,
            }
            loss_lower = loss.lower()
            if loss_lower in loss_map:
                self._loss = loss_map[loss_lower]()
                self._net.set_loss(self._loss)
            else:
                raise ValueError(f"Unknown loss: {loss}")
        elif loss is not None:
            self._loss = loss
            self._net.set_loss(self._loss)

    def fit(self, x, y, epochs=100, batch_size=32,
            validation_data=None, verbose=True, print_every=10):
        """Train the model on the provided data.
        Args:
            x: Input data as a Tensor.
            y: Target data as a Tensor.
            epochs: Number of epochs to train the model.
            batch_size: Size of the batches for training.
            validation_data: Optional tuple (x_val, y_val) for validation.
            verbose: Whether to print training progress.
            print_every: Frequency of printing training progress.
        Returns:
            A dictionary containing training history, including loss and accuracy.
        """
        if self._optimizer is None or self._loss is None:
            raise RuntimeError("Compile the model first with .compile()")

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            x_val = Tensor.zeros([0])
            y_val = Tensor.zeros([0])

        self._trainer = _C.Trainer(self._net, self._optimizer)
        result = self._trainer.fit(x, y, x_val, y_val,
                                    epochs, batch_size, verbose, print_every)
        return result

    def predict(self, x):
        """Make predictions using the trained model.
        Args:
            x: Input data as a Tensor.
        Returns:
            Predictions as a Tensor.
        """
        return self._net.predict(x, False)

    def evaluate(self, x, y):
        """Evaluate the model on the provided data.
        Args:
            x: Input data as a Tensor.
            y: Target data as a Tensor.
        Returns:
            A tuple (loss, accuracy) representing the evaluation metrics.
        """
        loss = self._net.compute_loss(x, y, False)
        acc = self._net.accuracy(x, y)
        return loss, acc

    def accuracy(self, x, y):
        """Compute the accuracy of the model on the provided data.
        Args:
            x: Input data as a Tensor.
            y: Target data as a Tensor.
        Returns:
            Accuracy as a float.
        """
        return self._net.accuracy(x, y)

    def save(self, path):
        """Save the model to the specified path.
        Args:
            path: The path where the model will be saved.
        """
        self._net.save(path)

    def load(self, path):
        """Load the model from the specified path.
        Args:
            path: The path from which the model will be loaded.
        """
        self._net.load(path)
