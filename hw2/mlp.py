import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        # 初始化權重和偏置項
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, x):
        # 線性變換
        self.z = np.dot(x, self.weights) + self.biases
        # 應用激活函數
        self.a = self.activation(self.z)
        return self.a
    

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, y_pred, y_true):
        # 使用均方誤差 (Mean Squared Error)
        return np.mean((y_pred - y_true) ** 2)
    

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# 建立網路
network = Network()
network.add_layer(Layer(input_size=3, output_size=5, activation=relu))   # 隱藏層 1
network.add_layer(Layer(input_size=5, output_size=4, activation=relu))   # 隱藏層 2
network.add_layer(Layer(input_size=4, output_size=1, activation=sigmoid)) # 輸出層

# 隨機生成數據並進行前向傳播
x = np.random.randn(10, 3)  # 10個樣本，每個樣本有3個特徵
y_true = np.random.randn(10, 1)

y_pred = network.forward(x)
loss = network.compute_loss(y_pred, y_true)
print("Loss:", loss)