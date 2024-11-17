
import numpy as np
import matplotlib.pyplot as plt
from mlp import Layer, Network, relu, relu_derivative, softmax
from util import split_data, one_hot_encoding, plot_feature_scatter

def load_iris_data(fp):
    data = np.genfromtxt(fp, delimiter=',', dtype=str)
    feature_names = data[0, 1:5].tolist()       # record the feature (column) names
    features = data[1:, 1:5].astype(float)      # convert the first 4 column to features
    species, labels = np.unique(data[1:, 5], return_inverse=True)    # convert last column(species) to labels
    return features, labels, len(species), dict(enumerate(species)), feature_names


if __name__ == "__main__":
    features, labels, num_cls, species_to_label, feature_names = load_iris_data('data/Iris.csv')

    # plot feature-to-feature figures
    plot_feature_scatter(features, labels, species_to_label, feature_names, save_dir="result/iris")

    # preprocess data
    labels_one_hot = one_hot_encoding(labels, num_cls)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels_one_hot) 


    # Initialize and train the MLP network
    network = Network()
    network.add_layer(Layer(input_size=4, output_size=5, activation=relu))
    network.add_layer(Layer(input_size=5, output_size=5, activation=relu))
    network.add_layer(Layer(input_size=5, output_size=num_cls, activation=softmax))


    # network.train(train_features, train_labels, val_features, val_labels, learning_rate=0.01, epochs=1000)
    network.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=1000, momentum=0.9, loss_type="categorical_cross_entropy")


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

