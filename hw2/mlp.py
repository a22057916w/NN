import numpy as np

# Activation Function & Derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x)
    exp_x = np.maximum(exp_x, 1e-10) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Nomralization
def min_max_norm(array, epsilon=1e-10):
    return (array - np.min(array)) / (np.max(array) - np.min(array) + epsilon)

def mean_norm(array, epsilon=1e-10):
    return (array - np.mean(array)) / (np.max(array) - np.min(array) + epsilon)

def standardize(array, epsilon=1e-10):
    return (array - np.mean(array)) / (np.std(array) + epsilon)


# Loss Function
def categorical_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # avoid log(0) 
    y_pred = np.maximum(y_pred, epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # avoid log(0) 
    y_pred = np.maximum(y_pred, epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.1  # init randomly
        self.biases = np.zeros((1, output_size))  # init to 0
        self.activation = activation

        # init Momentums
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)


    def forward(self, x):
        self.input = x
        self.input = standardize(self.input)   # Do normalization

        self.Z = np.dot(self.input, self.weights) + self.biases     # Z : potential values
        self.Y = self.activation(self.Z)                            # Y : activation values
        return self.Y


    def backward(self, dY, learning_rate, momentum):
        # Compute gradients
        if self.activation == relu:
            dZ = dY * relu_derivative(self.Z)
        elif self.activation == sigmoid:
            dZ = dY * sigmoid_derivative(self.Z)
        else:
            dZ = dY

        dW = np.dot(self.input.T, dZ) / self.input.shape[0]     # average each weight by sample amount
        dB = np.sum(dZ, axis=0, keepdims=True) / self.input.shape[0]
        dY_prev = np.dot(dZ, self.weights.T)

        # Update with "Momentum"
        self.velocity_w = momentum * self.velocity_w - learning_rate * dW
        self.velocity_b = momentum * self.velocity_b - learning_rate * dB
        self.weights += self.velocity_w
        self.biases += self.velocity_b

        # return the gradients to the previous layer
        return dY_prev


class Network:
    def __init__(self):
        self.layers = []
        self.history = {"train_accuracy": [], "train_loss": [], "val_accuracy": [], "val_loss": []}


    def add_layer(self, layer):
        self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, y_pred, y_true, loss_type):
        if loss_type == "categorical_cross_entropy":
            return categorical_cross_entropy(y_pred, y_true)
        elif loss_type == "binary_cross_entropy":
            return binary_cross_entropy(y_pred, y_true)
        else:
            raise ValueError("Unsupported loss type")


    def backward(self, y_pred, y_true, learning_rate, momentum):
        dJ = y_pred - y_true        # Gradients of Output Layer
        for layer in reversed(self.layers):
            dJ = layer.backward(dJ, learning_rate, momentum)


    def train(self, x, y_true, val_x, val_y, learning_rate=0.05, epochs=100, momentum=0.9, loss_type="categorical_cross_entropy"):
        # Train process
        for epoch in range(epochs):
            # Training pass
            y_pred = self.forward(x)
            train_loss = self.compute_loss(y_pred, y_true, loss_type)
            self.backward(y_pred, y_true, learning_rate, momentum)
            train_accuracy = self.evaluate(x, y_true)

            # Validation pass
            val_pred = self.forward(val_x)
            val_loss = self.compute_loss(val_pred, val_y, loss_type)
            val_accuracy = self.evaluate(val_x, val_y)

            # Record training history
            self.history["train_accuracy"].append(train_accuracy)
            self.history["train_loss"].append(train_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    def predict(self, x):
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)


    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred == true_labels)
        return accuracy
