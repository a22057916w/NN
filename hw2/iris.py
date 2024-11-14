
import numpy as np
from mlp import Layer, Network, relu, relu_derivative, softmax
import matplotlib.pyplot as plt

def load_iris_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    features = data[:, 1:5].astype(float)   # convert the first 4 column to features
    species, labels = np.unique(data[:, 5], return_inverse=True)    # convert last column(species) to labels
    return features, labels, dict(enumerate(species))


def split_data(features, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    np.random.seed(0)  # for reproducibility
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    
    # Split indices based on the given ratios
    train_end = int(train_ratio * len(indices))
    val_end = train_end + int(val_ratio * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Index the features and labels
    return (features[train_indices], labels[train_indices],
            features[val_indices], labels[val_indices],
            features[test_indices], labels[test_indices])


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


# Load and preprocess the iris dataset, including splitting it into train, validation, and test sets
features, labels, species_to_label = load_iris_data('Iris.csv')
num_classes = len(species_to_label)
labels_one_hot = one_hot_encode(labels, num_classes)
train_features, train_labels, val_features, val_labels, test_features, test_labels = split_data(features, labels_one_hot)


# Initialize and train the MLP network
network = Network()
network.add_layer(Layer(input_size=4, output_size=5, activation=relu, activation_derivative=relu_derivative))
network.add_layer(Layer(input_size=5, output_size=5, activation=relu, activation_derivative=relu_derivative))
network.add_layer(Layer(input_size=5, output_size=3, activation=softmax))


network.train(train_features, train_labels, val_features, val_labels, learning_rate=0.01, epochs=1000)


# Plotting accuracy vs epoch and loss vs epoch
epochs = range(1, 1001)

plt.figure(figsize=(14, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, network.history["train_accuracy"], label="Training Accuracy")
plt.plot(epochs, network.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, network.history["train_loss"], label="Training Loss")
plt.plot(epochs, network.history["val_loss"], label="Validation Loss", linestyle="--", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()