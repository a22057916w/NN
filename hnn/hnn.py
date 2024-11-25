import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    def __init__(self):
        self.W = None

    def activation(self, X):
        return np.sign(X)

    def train(self, patterns):
        """
        Train the weight matrix of the Hopfield network.

        Parameters:
        patterns (ndarray): 2D array of Training samples, each row is a pattern.
        """
        num_neurons = patterns.shape[1]
        self.W = np.zeros((num_neurons, num_neurons))
        
        # Update by "Outer Product"
        for p in patterns:
            self.W += np.outer(p, p)
        
        # Remove self-correlation
        np.fill_diagonal(self.W, 0)

        self.W /= patterns.shape[0]


    def recall(self, pattern, max_iters=1000, tol=1e-5):
        """
        Recall process to recover the pattern until it reaches a stable state.

        Parameters:
        pattern (ndarray): 1D array of image pixels.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criteria.
        """
        for _ in range(max_iters):
            new_pattern = self.activation(np.dot(self.W, pattern))
            
            # If the new pattern is equal to previous one, it is reached the stable state
            if np.allclose(new_pattern, pattern, atol=tol):
                break
            pattern = new_pattern

        return pattern

def plot_pattern(pattern, title):
    fig = pattern.reshape((9, 5))
    plt.imshow(fig, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def add_noise_to_pattern(pattern, noise_level):
    """
    Randomly modify a given percentage of pixel values.

    Parameters:
    pattern (ndarray): 1D array of original pattern.
    noise_level (int): Percentage of pixels to be randomly changed (0-100).
    """
    noisy_pattern = pattern.copy()
    num_pixels_to_change = int(len(pattern) * noise_level / 100)
    indices = random.sample(range(len(pattern)), num_pixels_to_change)
    for idx in indices:
        noisy_pattern[idx] = random.choice([-1, 1]) 
    return noisy_pattern


if __name__ == "__main__":
    # Image 1
    img1 = np.array([
        [1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    # Image 2
    img2 = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    # Image 3
    img3 = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    # Image 4
    img4 = np.array([
        [1, 1, 1, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1]
    ])
    # Preprocess the patterns (images)
    patterns = np.array([img1.flatten(), img2.flatten(), img3.flatten(), img4.flatten()])
    patterns = np.where(patterns == 0, -1, 1)   # Convert 0 to -1

    # Init and train the HNN
    hnn = HopfieldNetwork()
    hnn.train(patterns)
    
    # Plot the original, corrupted, recovered images
    noise_level = 60  
    for img in patterns:
        plot_pattern(img, title="Original Pattern")
        test_pattern = add_noise_to_pattern(img, noise_level)   # Randomly add noise to the original image
        plot_pattern(test_pattern, title=f"Test Pattern ({noise_level}% Distorted)")
        recovered_pattern = hnn.recall(test_pattern)
        plot_pattern(recovered_pattern, title=f"Recovered Pattern")
