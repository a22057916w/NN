import numpy as np
import matplotlib.pyplot as plt
from mlp import Layer, Network, relu, relu_derivative, softmax
import numpy as np

def load_cancer_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    features = data[:, 2:].astype(float)  # Extract all columns except the first (ID) and second one (label) as features
    diagnosis, labels = np.unique(data[:, 1], return_inverse=True)  # Convert the second column (diagnosis) to labels
    return features, labels, dict(enumerate(diagnosis))

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

# Load and preprocess the breast cancer dataset, including splitting it into train, validation, and test sets
features, labels, diagnosis_to_label = load_cancer_data('breast_cancer.csv')
num_classes = len(diagnosis_to_label)
labels_one_hot = one_hot_encode(labels, num_classes)

train_features, train_labels, val_features, val_labels, test_features, test_labels = split_data(features, labels_one_hot)

# Initialize and train the MLP network
network = Network()
network.add_layer(Layer(input_size=features.shape[1], output_size=10, activation=relu, activation_derivative=relu_derivative))
network.add_layer(Layer(input_size=10, output_size=10, activation=relu, activation_derivative=relu_derivative))
network.add_layer(Layer(input_size=10, output_size=num_classes, activation=softmax))

network.train(train_features, train_labels, val_features, val_labels, learning_rate=0.01, epochs=1000, momentum=0.9)

# Plotting accuracy vs epoch and loss vs epoch
epochs = range(1, 801)

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
