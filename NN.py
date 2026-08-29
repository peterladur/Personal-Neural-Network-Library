import numpy as np
from typing import List, Tuple, Dict, Callable, Union, Optional


class Activations:
    """Activation functions and their analytical derivatives."""

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(z.dtype)

    @staticmethod
    def leaky_relu(z: np.ndarray, k: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, k * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, k: float = 0.01) -> np.ndarray:
        return np.where(z > 0, 1.0, k).astype(z.dtype)

    @staticmethod
    def logistic(z: np.ndarray, k: float = 1.0) -> np.ndarray:
        z_clipped = np.clip(-k * z, -500, 500)
        return 1.0 / (1.0 + np.exp(z_clipped))

    @staticmethod
    def logistic_derivative(z: np.ndarray, k: float = 1.0) -> np.ndarray:
        s = Activations.logistic(z, k)
        return k * s * (1.0 - s)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(z: np.ndarray) -> np.ndarray:
        s = Activations.softmax(z)
        return s * (1.0 - s)


class NeuralNetwork:
    """
    A modular, vectorized Feedforward Neural Network implemented in NumPy.
    """

    ACTIVATION_REGISTRY: Dict[str, Tuple[Callable, Callable]] = {
        'ReLU': (Activations.relu, Activations.relu_derivative),
        'LeakyReLU': (Activations.leaky_relu, Activations.leaky_relu_derivative),
        'Logistic': (Activations.logistic, Activations.logistic_derivative),
        'Sigmoid': (Activations.logistic, Activations.logistic_derivative),
        'Tanh': (Activations.tanh, Activations.tanh_derivative),
        'Linear': (Activations.linear, Activations.linear_derivative),
        'Softmax': (Activations.softmax, Activations.softmax_derivative),
    }

    def __init__(
        self,
        layer_sizes: List[int],
        activation_functions: List[str],
        data_type=np.float32,
        seed: Optional[int] = None
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least an input layer and an output layer.")
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError(
                f"Number of activation functions ({len(activation_functions)}) "
                f"must match number of layer transitions ({len(layer_sizes) - 1})."
            )

        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = list(layer_sizes)
        self.activation_functions_names = list(activation_functions)
        self.data_type = data_type
        self.num_layers = len(layer_sizes)

        self._bind_activations()
        self._initialize_weights()

    def _bind_activations(self):
        """Resolves activation function string names to callable forward and derivative functions."""
        self.activation_functions = []
        self.activation_functions_derivatives = []
        for name in self.activation_functions_names:
            if name not in self.ACTIVATION_REGISTRY:
                raise ValueError(
                    f"Unsupported activation function '{name}'. "
                    f"Available options: {list(self.ACTIVATION_REGISTRY.keys())}"
                )
            forward_fn, deriv_fn = self.ACTIVATION_REGISTRY[name]
            self.activation_functions.append(forward_fn)
            self.activation_functions_derivatives.append(deriv_fn)

    def _initialize_weights(self):
        """Initializes weights using He (Kaiming) or Xavier initialization based on activation type."""
        self.layer_weights = []
        self.layer_biases = []

        for i in range(self.num_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            act_name = self.activation_functions_names[i]

            # He initialization for ReLU/LeakyReLU, Xavier initialization for Sigmoid/Tanh/Linear
            if act_name in ('ReLU', 'LeakyReLU'):
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(1.0 / n_in)

            weights = (np.random.randn(n_out, n_in) * std).astype(self.data_type)
            biases = np.zeros((n_out, 1), dtype=self.data_type)

            self.layer_weights.append(weights)
            self.layer_biases.append(biases)

    def __repr__(self) -> str:
        lines = ["NeuralNetwork Architecture:"]
        lines.append(f"  Input Layer Size:  {self.layer_sizes[0]}")
        for i in range(self.num_layers - 1):
            lines.append(
                f"  Layer {i+1}: {self.layer_sizes[i]} -> {self.layer_sizes[i+1]} "
                f"| Activation: {self.activation_functions_names[i]}"
            )
        lines.append(f"  Output Layer Size: {self.layer_sizes[-1]}")
        return "\n".join(lines)

    def forward_propagate(self, input_data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Runs forward propagation through all layers.

        Parameters
        ----------
        input_data : np.ndarray
            Input matrix of shape (n_features, batch_size) or (batch_size, n_features).

        Returns
        -------
        z_layers : List[np.ndarray]
            List of pre-activation matrices [z1, z2, ..., zL]
        a_layers : List[np.ndarray]
            List of post-activation matrices [a0, a1, ..., aL] where a0 is input.
        """
        x = np.asarray(input_data, dtype=self.data_type)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.shape[0] != self.layer_sizes[0] and x.shape[1] == self.layer_sizes[0]:
            x = x.T

        a_layers = [x]
        z_layers = []

        for i in range(self.num_layers - 1):
            z = self.layer_weights[i] @ a_layers[-1] + self.layer_biases[i]
            a = self.activation_functions[i](z)
            z_layers.append(z)
            a_layers.append(a)

        return z_layers, a_layers

    def backward_propagate(
        self,
        z_layers: List[np.ndarray],
        a_layers: List[np.ndarray],
        target_output: np.ndarray,
        learning_rate: float
    ):
        """
        Backpropagates error through the network and updates weights and biases.
        """
        target = np.asarray(target_output, dtype=self.data_type)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        elif target.shape[0] != self.layer_sizes[-1] and target.shape[1] == self.layer_sizes[-1]:
            target = target.T

        batch_size = a_layers[-1].shape[1]

        # Compute output error (MSE loss gradient with respect to network predictions)
        output_error = a_layers[-1] - target

        for i in reversed(range(self.num_layers - 1)):
            z = z_layers[i]
            a_prev = a_layers[i]

            activation_grad = self.activation_functions_derivatives[i](z)
            delta = output_error * activation_grad

            # Compute gradients
            dW = (delta @ a_prev.T) / batch_size
            db = np.sum(delta, axis=1, keepdims=True) / batch_size

            # Save next output error BEFORE updating current layer weights
            if i > 0:
                output_error = self.layer_weights[i].T @ delta

            # Update weights and biases
            self.layer_weights[i] -= learning_rate * dW
            self.layer_biases[i] -= learning_rate * db

    def train(
        self,
        input_data: np.ndarray,
        target_output: np.ndarray,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """
        Trains the neural network using Mini-batch Gradient Descent.
        """
        x = np.asarray(input_data, dtype=self.data_type)
        y = np.asarray(target_output, dtype=self.data_type)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Standardize samples to (n_samples, features) layout for batch splitting
        if x.shape[0] == self.layer_sizes[0] and x.shape[1] != self.layer_sizes[0]:
            x = x.T
        if y.shape[0] == self.layer_sizes[-1] and y.shape[1] != self.layer_sizes[-1]:
            y = y.T

        num_samples = x.shape[0]
        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_samples, batch_size):
                batch_x = x_shuffled[i : i + batch_size].T  # Shape: (n_in, batch_size)
                batch_y = y_shuffled[i : i + batch_size].T  # Shape: (n_out, batch_size)

                z_layers, a_layers = self.forward_propagate(batch_x)
                batch_loss = float(np.mean((a_layers[-1] - batch_y) ** 2))
                epoch_loss += batch_loss
                num_batches += 1

                self.backward_propagate(z_layers, a_layers, batch_y, learning_rate)

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)

            if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        return loss_history

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Runs forward propagation and returns predictions of shape (n_samples, n_outputs).
        """
        _, a_layers = self.forward_propagate(input_data)
        return a_layers[-1].T

    def save_NN_data(
        self,
        file_weights: str = "weights.npz",
        file_biases: str = "biases.npz",
        other_data: str = "data.txt"
    ):
        """Saves weights, biases, and network architecture metadata."""
        np.savez(file_weights, *self.layer_weights)
        np.savez(file_biases, *[b.squeeze() for b in self.layer_biases])
        with open(other_data, "w") as f:
            for i, (w, b) in enumerate(zip(self.layer_weights, self.layer_biases)):
                f.write(f"Layer {i+1}:\n")
                f.write(f"  Weights: {w.shape}\n")
                f.write(f"  Biases: {b.shape}\n")
                f.write(f"  Activation Function: {self.activation_functions_names[i]}\n")

    def load_NN_data(
        self,
        file_weights: str = "weights.npz",
        file_biases: str = "biases.npz",
        other_data: str = "data.txt"
    ):
        """Loads weights, biases, and metadata into the neural network instance."""
        weights_data = np.load(file_weights)
        biases_data = np.load(file_biases)

        activation_names = []
        with open(other_data, "r") as f:
            for line in f:
                if line.startswith("  Activation Function:"):
                    activation_names.append(line.split(":")[1].strip())

        self.activation_functions_names = activation_names
        self._bind_activations()

        self.layer_weights = [weights_data[f'arr_{i}'] for i in range(len(weights_data.files))]
        self.layer_biases = [
            biases_data[f'arr_{i}'].reshape(-1, 1) for i in range(len(biases_data.files))
        ]
        self.layer_sizes = [self.layer_weights[0].shape[1]] + [b.shape[0] for b in self.layer_biases]
        self.num_layers = len(self.layer_sizes)
