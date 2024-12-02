import numpy as np
from mlp import Layer, Network, relu, softmax
from util import split_data, one_hot_encoding, plot_feature_scatter, plot_training_results

def load_cancer_data(fp):
    with open(fp, 'r') as f:
        feature_names = f.readline().strip().split(',')[2:]     # record feature names
    
    data = np.genfromtxt(fp, delimiter=',', dtype=str, skip_header=1)
    features = data[:, 2:].astype(float)  # get features
    diagnosis, labels = np.unique(data[:, 1], return_inverse=True)  # convert the second column (diagnosis) to labels
    return features, labels, len(diagnosis), dict(enumerate(diagnosis)), feature_names


if __name__ == "__main__":
    features, labels, num_cls, diagnosis_to_label, feature_names = load_cancer_data('data/breast_cancer.csv')

    # plot feature-to-feature figures
    plot_feature_scatter(features, labels, diagnosis_to_label, feature_names, save_dir="result/breast_cancer/scatter")

    # preprocess data
    labels_one_hot = one_hot_encoding(labels, num_cls)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels_one_hot, 0.6, 0.2, 0.2)    # train, val, test

    # initialize and train the MLP network
    network = Network()
    network.add_layer(Layer(input_size=features.shape[1], output_size=40, activation=relu))
    network.add_layer(Layer(input_size=40, output_size=20, activation=relu))
    network.add_layer(Layer(input_size=20, output_size=10, activation=relu))
    network.add_layer(Layer(input_size=10, output_size=num_cls, activation=softmax))

    eps = 2000
    network.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=eps, momentum=0.3, loss_type="categorical_cross_entropy")
    
    # plot accuracy and loss
    plot_training_results(network.history, eps, save_dir="result/breast_cancer/metric")
    print(f"Test Accuracy: {network.evaluate(X_test, y_test)}")
