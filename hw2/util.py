from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os


def split_data(features, labels, train_rt=0.6, val_rt=0.2, test_rt=0.2, random_state=42, shuffle=True):
    # assign indices to feature samples
    indices = np.arange(features.shape[0])     

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    # split indices
    train_end = int(train_rt * len(indices))
    val_end = train_end + int(val_rt * len(indices))

    # assign dataset
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (features[train_indices], features[val_indices], features[test_indices], 
            labels[train_indices], labels[val_indices], labels[test_indices])


def one_hot_encoding(labels, num_cls):
    one_hot = np.zeros((len(labels), num_cls))
    for i in range(len(labels)):
        one_hot[i, labels[i]] = 1
    return one_hot


def plot_feature_scatter(features, labels, feature_to_label, feature_names, save_dir):
    """
    Plot scatter plots of features against each other.

    Parameters:
    features (ndarray): 2D array containing feature values of shape (n_samples, n_features).
    labels (ndarray): 1D array containing the labels for each sample.
    feature_to_label (dict): A dictionary mapping label indices to label names.
    feature_names (list): A list of feature names corresponding to the features array.
    save_dir (str): Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving scatter plots to {save_dir}...")
    
    feature_indices = list(combinations(range(features.shape[1]), 2))  # Get all combinations of two features
    
    count = 1
    for i in range(0, len(feature_indices), 6):
        plt.figure(figsize=(18, 12))
        for idx, (feature_x_idx, feature_y_idx) in enumerate(feature_indices[i:i+6]):
            # Extracting selected features
            x_values = features[:, feature_x_idx]
            y_values = features[:, feature_y_idx]

            plt.subplot(2, 3, idx + 1)
            for label in np.unique(labels):
                plt.scatter(x_values[labels == label], y_values[labels == label],
                            label=feature_to_label[label], edgecolor='black')

            plt.xlabel(f'{feature_names[feature_x_idx]}')
            plt.ylabel(f'{feature_names[feature_y_idx]}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feat_to_feat({count}).png'))
        count += 1
        plt.close()


def plot_training_results(history, epochs, save_dir, figsize=(14, 6)):
    """
    Plot training "accuracy vs epoch" and "loss vs epoch".

    Parameters:
    history (dict): A dictionary containing 'train_accuracy', 'val_accuracy', 'train_loss', and 'val_loss'.
    epochs (int): Number of epochs used during training.
    save_path (str): Directory to save the plot.
    figsize (tuple): Size of the plot figure. Default is (14, 6).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epoch_range = range(1, epochs + 1)
    
    plt.figure(figsize=figsize)

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epoch_range, history["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, history["train_loss"], label="Training Loss")
    plt.plot(epoch_range, history["val_loss"], label="Validation Loss", linestyle="--", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics.png"))
    plt.show()
    
