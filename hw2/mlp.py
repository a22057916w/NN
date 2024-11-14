import numpy as np

# 激活函數及其微分
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 損失函數
def categorical_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # 避免 log(0) 錯誤
    y_pred = np.maximum(y_pred, epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Layer 類別，包含動量
class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative=None):
        self.weights = np.random.randn(input_size, output_size) * 0.1  # 隨機初始化權重
        self.biases = np.zeros((1, output_size))  # 初始化偏置為 0
        self.activation = activation
        self.activation_derivative = activation_derivative

        # 初始化動量項
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

    def forward(self, x):
        # 前向傳播
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, dJ, learning_rate, momentum):
        # 計算梯度
        if self.activation_derivative:
            dZ = dJ * self.activation_derivative(self.z)
        else:
            dZ = dJ

        dW = np.dot(self.input.T, dZ) / self.input.shape[0]
        dB = np.sum(dZ, axis=0, keepdims=True) / self.input.shape[0]

        # 使用動量進行更新
        self.velocity_w = momentum * self.velocity_w - learning_rate * dW
        self.velocity_b = momentum * self.velocity_b - learning_rate * dB
        self.weights += self.velocity_w
        self.biases += self.velocity_b

        # 返回前一層的梯度
        dJ_prev = np.dot(dZ, self.weights.T)
        return dJ_prev

# Network 類別
class Network:
    def __init__(self):
        self.layers = []
        self.history = {"train_accuracy": [], "train_loss": [], "val_accuracy": [], "val_loss": []}

    def add_layer(self, layer):
        # 添加層到網路
        self.layers.append(layer)

    def forward(self, x):
        # 前向傳播
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, y_pred, y_true):
        # 計算損失
        return categorical_cross_entropy(y_pred, y_true)

    def backward(self, y_pred, y_true, learning_rate, momentum):
        # 反向傳播
        dJ = y_pred - y_true  # 輸出層的梯度
        for layer in reversed(self.layers):
            dJ = layer.backward(dJ, learning_rate, momentum)

    def train(self, x, y_true, val_x, val_y, learning_rate=0.05, epochs=100, momentum=0.9):
        # 訓練過程
        for epoch in range(epochs):
            # Training pass
            y_pred = self.forward(x)
            train_loss = self.compute_loss(y_pred, y_true)
            self.backward(y_pred, y_true, learning_rate, momentum)
            train_accuracy = self.evaluate(x, y_true)

            # Validation pass
            val_pred = self.forward(val_x)
            val_loss = self.compute_loss(val_pred, val_y)
            val_accuracy = self.evaluate(val_x, val_y)

            # 記錄訓練和驗證數據
            self.history["train_accuracy"].append(train_accuracy)
            self.history["train_loss"].append(train_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    def predict(self, x):
        # 使用前向傳播進行預測
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)

    def evaluate(self, x, y_true):
        # 計算準確率
        y_pred = self.predict(x)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred == true_labels)
        return accuracy
