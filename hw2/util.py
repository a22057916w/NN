from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os


def split_data(features, labels, train_rt=0.6, val_rt=0.2, test_rt=0.2, random_state=42, shuffle=True):
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


def one_hot_encoding(labels, cls_num):
    one_hot = np.zeros((len(labels), cls_num))
    for i in range(len(labels)):
        one_hot[i, labels[i]] = 1
    return one_hot


def plot_feature_scatter(features, labels, feature_to_label, feature_names):
    feature_indices = list(combinations(range(features.shape[1]), 2))  # Get all combinations of two features
    save_dir = 'feature_scatter_plots'
    os.makedirs(save_dir, exist_ok=True)
    
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
        plt.savefig(os.path.join(save_dir, f'feature_scatter_{count}.png'))
        count += 1
        plt.close()