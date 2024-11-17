import numpy as np
from mlp import Layer, Network, relu, relu_derivative, softmax
from util import split_data, one_hot_encoding, plot_feature_scatter, plot_training_results

def load_wine_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)
    feature_names = data[0, :-1]
    features = data[1:, :-1].astype(float)  # get features
    qualities, labels = np.unique(data[1:, -1], return_inverse=True)  # convert the last column (quality) to labels
    return features, labels, len(qualities), dict(enumerate(qualities)), feature_names


if __name__ == "__main__":
    features, labels, num_cls, quality_to_label, feature_names = load_wine_data('data/winequality-red.csv')
    
    # plot feature-to-feature figures
    plot_feature_scatter(features, labels, quality_to_label, feature_names, save_dir="result/wine/scatter")

    # preprocess data
    labels_one_hot = one_hot_encoding(labels, num_cls)  
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels_one_hot, 0.6, 0.2, 0.2)    # train, val, test

    # initialize and train the MLP network
    network = Network()
    network.add_layer(Layer(input_size=features.shape[1], output_size=10, activation=relu))
    network.add_layer(Layer(input_size=10, output_size=10, activation=relu))
    network.add_layer(Layer(input_size=10, output_size=num_cls, activation=softmax))

    eps = 2000
    network.train(X_train, y_train, X_val, y_val, learning_rate=0.1, epochs=eps, momentum=0.9, loss_type="categorical_cross_entropy")

    # plot accuracy and loss
    plot_training_results(network.history, eps, save_dir="result/wine/metric")
    print(f"Test Accuracy: {network.evaluate(X_test, y_test)}")