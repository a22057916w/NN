
import numpy as np
import matplotlib.pyplot as plt
from mlp import Layer, Network, relu, softmax
from util import split_data, one_hot_encoding, plot_feature_scatter, plot_training_results

def load_iris_data(fp):
    data = np.genfromtxt(fp, delimiter=',', dtype=str)
    feature_names = data[0, 1:5].tolist()       # record the feature (column) names
    features = data[1:, 1:5].astype(float)      # convert the first 4 column to features
    species, labels = np.unique(data[1:, 5], return_inverse=True)    # convert last column(species) to labels
    return features, labels, len(species), dict(enumerate(species)), feature_names


if __name__ == "__main__":
    features, labels, num_cls, species_to_label, feature_names = load_iris_data('data/Iris.csv')

    # plot feature-to-feature figures
    plot_feature_scatter(features, labels, species_to_label, feature_names, save_dir="result/iris/scatter")

    # preprocess data
    labels_one_hot = one_hot_encoding(labels, num_cls)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels_one_hot) 


    # Initialize and train the MLP network
    network = Network()
    network.add_layer(Layer(input_size=4, output_size=5, activation=relu))
    network.add_layer(Layer(input_size=5, output_size=5, activation=relu))
    network.add_layer(Layer(input_size=5, output_size=num_cls, activation=softmax))

    eps = 1000
    network.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=eps, momentum=0.9, loss_type="categorical_cross_entropy")
    
    # plot accuracy and loss
    plot_training_results(network.history, eps, save_dir="result/iris/metric")



