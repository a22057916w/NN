import numpy as np
import matplotlib.pyplot as plt
from mlp import Layer, Network, relu, relu_derivative, softmax
from util import split_data, one_hot_encoding, plot_feature_scatter

def load_cancer_data(fp):
    with open(fp, 'r') as f:
        feature_names = f.readline().strip().split(',')[2:]
    data = np.genfromtxt(fp, delimiter=',', dtype=str, skip_header=1)
    features = data[:, 2:].astype(float)  # Extract all columns except the first (ID) and second one (label) as features
    diagnosis, labels = np.unique(data[:, 1], return_inverse=True)  # Convert the second column (diagnosis) to labels
    return features, labels, dict(enumerate(diagnosis)), feature_names


if __name__ == "__main__":
    features, labels, diagnosis_to_label, feature_names = load_cancer_data('breast_cancer.csv')

    # plot_feature_scatter(features, labels, diagnosis_to_label, feature_names)

    num_classes = len(diagnosis_to_label)
    labels_one_hot = one_hot_encoding(labels, num_classes)


    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels_one_hot) 


    network = Network()
    network.add_layer(Layer(input_size=features.shape[1], output_size=10, activation=relu, activation_derivative=relu_derivative))
    network.add_layer(Layer(input_size=10, output_size=10, activation=relu, activation_derivative=relu_derivative))
    network.add_layer(Layer(input_size=10, output_size=num_classes, activation=softmax))

    network.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=1000, momentum=0.9)

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
