import numpy as np
import matplotlib.pyplot as plt

class LVQ:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
        # Select the first sample of each class
        unique_labels = np.unique(labels)
        self.weights = np.array([data[labels == cls][0] for cls in unique_labels])
        self.weights_cls = unique_labels

        self.convergence_curve = []

    def train(self, X, y, epoch, learning_rate):
        for _ in range(epoch):
            for x, x_cls in zip(X, y):
                # Find the index of the closest weight vector (winner)
                winner_idx = self.find_winner(x)

                # Update the winner weight vector
                if self.weights_cls[winner_idx] == x_cls:
                    self.weights[winner_idx] += learning_rate * (x - self.weights[winner_idx])
                else:
                    self.weights[winner_idx] -= learning_rate * (x - self.weights[winner_idx])
                
            # Calculate the errors of convergence curve
            error = self.calculate_error(X)
            self.convergence_curve.append(error)

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = [euclidean_distance(x, w) for w in self.weights]
            winner_idx = np.argmin(distances)
            y_pred.append(self.weights_cls[winner_idx])
        return y_pred
    
    def find_winner(self, x):
        distances = [euclidean_distance(x, w) for w in self.weights]
        return np.argmin(distances)
    
    # Calculate the error by the distance between nodes and winners
    def calculate_error(self, X):
        error = 0
        for x in X:
            winner_idx = self.find_winner(x)
            error += np.linalg.norm(self.weights[winner_idx] - x)
        return error / len(X)

    def plot_convergence(self):
        plt.plot(self.convergence_curve)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Convergence Curve')
        plt.show()

    # plot the training/test data by every 2 features with trained winners
    def plot_result(self, data, labels, features, title):
        for i in range(len(features)-1):
            # Plot data points by labels
            for label in np.unique(labels):
                cls_indices = np.where(np.array(labels) == label)[0]    # e.g. X([0, 1, 2, 3], dtype=int64)[0]
                plt.scatter(
                    data[cls_indices, i], data[cls_indices, i+1],
                    label=f"Class {int(label)}",
                    alpha=0.7
                )
            # Plot weights
            plt.scatter(self.weights[:, i], self.weights[:, i+1], c='red', marker='x', label='Weights')
            plt.xlabel(f"{features[i]}")
            plt.ylabel(f"{features[i+1]}")
            plt.title(title)
            plt.legend()
            plt.grid()
            plt.show()


# Helper
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Nomralization
def min_max_norm(X, X_min, X_max, epsilon=1e-10):
    return (X - X_min) / (X_max - X_min + epsilon)


if __name__ == "__main__":
    # Load training data
    data = np.genfromtxt("data/power_consumption.csv", delimiter=",", dtype=str)
    features = data[0, 1:-1].astype(str)
    X = data[1:, 1:-1].astype(float)
    y = data[1:, -1].astype(int)
    
    # Load test data
    test_data = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1)
    X_test = test_data[:, 1:]

    
    # Normalize training and test data
    X_min = np.min(X)
    X_max = np.max(X)
    X = min_max_norm(X, X_min, X_max)
    X_test = min_max_norm(X_test, X_min, X_max)

    # Train
    lvq = LVQ(X, y)
    lvq.train(X, y, epoch=100, learning_rate=0.1)
    lvq.plot_convergence()
    lvq.plot_result(X, y, features, title="Training Data and Weights")

    # Predict
    y_pred = lvq.predict(X_test)
    lvq.plot_result(X_test, y_pred, features, title="Test Data and Weights")
  