import numpy as np
import matplotlib.pyplot as plt
import os

class SOM:
    def __init__(self, n, m, dim):
        """
        Initialize the SOM (Self-Organizing Map) with a given grid size and input dimension.
        
        Parameters:
        n (int): Number of rows in the SOM grid.
        m (int): Number of columns in the SOM grid.
        dim (int): Dimension of the input vectors.
        """
        self.input_dim = dim
        self.weights = np.random.rand(n, m, dim)
        self.convergence_curve = []

    def train(self, data, sigma=5, learning_rate=0.1, max_iter=1000, early_stop=True, patience=20, min_iter=100):
        """
        Train the SOM with sigma decay and optional early stopping.
        
        Parameters:
        sigma (float): Initial standard deviation(radius) for Gassuain Function.
        """
        no_improvement_count = 0
        early_stop_th = 1e-4
        
        for epoch in range(max_iter):
            for input_vector in data:
                # Find the winner neuron (weight)
                winner_pos = self.find_winner(input_vector)

                # Update the weights for the winner and its neighbors
                self.update(input_vector, winner_pos, sigma, learning_rate)

            # Decay by iteration
            sigma = max(sigma * np.exp(-epoch / max_iter), 1)            

            # Calculate and store the quantization error
            error = self.calculate_error(data)
            self.convergence_curve.append(error)

            # Early stopping condition based on error difference between epochs
            if epoch > min_iter and early_stop:
                error_diff = abs(self.convergence_curve[-2] - error)
                if error_diff < early_stop_th:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch} with quantization error {error}")
                    break

    def find_winner(self, input_vector):
        """
        Returns:
        tuple: Coordinates of the winning neuron in the SOM grid (x, y).
        """
        distances = euclidean_distance(self.weights, input_vector)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update(self, input_vector, winner_pos, sigma, learning_rate):
        """   
        Parameters:
        winner_pos (tuple): Coordinates of the winning neuron in the SOM grid.
        sigma (float): Current neighborhood radius for updating.
        """
        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                dist_to_winner = euclidean_distance(np.array([x, y]), np.array(winner_pos))
                # Gaussian neighborhood function
                alpha = max(np.exp(-dist_to_winner**2 / (2 * (sigma**2))), 1e-9)
                # Update winner and neighbor 
                self.weights[x, y] += alpha * learning_rate * (input_vector - self.weights[x, y])

    
    def calculate_error(self, data):
        """
        Returns:
        float: Average error over all input vectors.
        """
        error = 0
        for input_vector in data:
            winner_pos = self.find_winner(input_vector)
            error += np.linalg.norm(self.weights[winner_pos] - input_vector)
        return error / data.shape[0]

    def plot_weight_mesh(self, save_name):
        ig, ax = plt.subplots()
        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                # draw points
                ax.scatter(self.weights[x, y, 0], self.weights[x, y, 1], c='b')
                # draw lines
                if x > 0:
                    ax.plot([self.weights[x, y, 0], self.weights[x - 1, y, 0]],
                            [self.weights[x, y, 1], self.weights[x - 1, y, 1]], c='k')
                if y > 0:
                    ax.plot([self.weights[x, y, 0], self.weights[x, y - 1, 0]],
                            [self.weights[x, y, 1], self.weights[x, y - 1, 1]], c='k')
        # draw data
        ax.scatter(data[:, 0], data[:, 1], c='lightgreen', marker='o', label='Data Points')
        ax.set_title('Weight Mesh')
        plt.legend()
        plt.savefig(os.path.join("result", save_name))
        # plt.show()
        plt.close()
        
    def plot_convergence_curve(self, save_name):
        plt.plot(self.convergence_curve)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Convergence Curve')
        plt.savefig(os.path.join("result", save_name))
        # plt.show()
        plt.close()
        


# helper functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))



if __name__ == "__main__":
    # ========================= ThreeGropus.txt ================================
    # data = np.loadtxt('ThreeGroups.txt')

    # # Min-Max Normalization to [0, 1]
    # data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    # # Initialize and train SOM
    # som = SOM(20, 20, data.shape[1])
    # som.train(data, sigma=1, learning_rate=0.1, max_iter=1000)

    # # Plot the weight mesh and the convergence curve
    # som.plot_weight_mesh(f"mesh_ThreeGroups_1_400.png")
    # som.plot_convergence_curve("curve_ThereeGroups_1_400.png")

    # # ========================= TwoCircles.txt ================================
    # data = np.loadtxt('TwoCircles.txt')
    # data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    # # Initialize and train SOM
    # som = SOM(5, 5, data.shape[1])
    # som.train(data, sigma=1, learning_rate=0.1, max_iter=1000)

    # # Plot the weight mesh and the convergence curve
    # som.plot_weight_mesh("mesh_TwoCircles_1_25.png")
    # som.plot_convergence_curve("curve_TwoCricles_1_25.png")

    # # ========================= TwoRings.txt ================================
    data = np.loadtxt('TwoRings.txt')
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    # Initialize and train SOM
    som = SOM(10, 10, data.shape[1])
    som.train(data, sigma=1, learning_rate=0.1, max_iter=1000)

    # Plot the weight mesh and the convergence curve
    som.plot_weight_mesh("mesh_TwoRings_1_100_2.png")
    som.plot_convergence_curve("curve_TwoRings_1_100_2.png")