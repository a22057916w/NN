import numpy as np

class Neuron:
    def __init__(self, num_features):
        self.forward_weight = np.ones(num_features) / (1 + num_features)  # 1-D vector of 1/(1+n)
        self.feedback_weight = np.ones(num_features)  # 1-D vector of 1
        self.status = "enable"  
        self.activation_value = 0  

    def update_weights(self, sample):
        self.forward_weight = (self.feedback_weight * sample) / (0.5 + np.dot(self.feedback_weight, sample))
        self.feedback_weight = self.feedback_weight * sample 
        
    def calculate_activation(self, sample):
        self.activation_value = np.dot(self.forward_weight, sample)


class ART:
    def __init__(self, num_features, vigilance):
        self.num_features = num_features
        self.vigilance = vigilance
        self.neurons = [Neuron(num_features)]  # initialize only one neuron at begining

    def calculate_similarity(self, neuron, sample):
        return np.dot(neuron.feedback_weight, sample) / np.sum(sample)

    def find_winner(self, sample):
        max_activation = -1e9
        winner_idx = None

        # find winner neurons with max activation value
        for idx, neuron in enumerate(self.neurons):
            neuron.calculate_activation(sample)
            if neuron.status == "enable" and neuron.activation_value > max_activation:
                max_activation = neuron.activation_value
                winner_idx = idx

        # return neuron index
        return winner_idx

    def train(self, data):
        history = []    # store the training process
        for sample in data:
            # store the result of each sample
            result = {
                "status": "Update", "k_first": self.find_winner(sample), "k_final": None
            }

            # Begin training
            matched = False
            while not matched:
                winner_idx = self.find_winner(sample)

                if winner_idx is not None:
                    winner_neuron = self.neurons[winner_idx]
                    
                    similarity = self.calculate_similarity(winner_neuron, sample)

                    # Update weights
                    if similarity >= self.vigilance:
                        winner_neuron.update_weights(sample)
                        matched = True
                        result["k_final"] = winner_idx
                    else:
                        # disable the failed candidate
                        winner_neuron.status = "disable"
                else:
                    new_neuron = Neuron(self.num_features)
                    new_neuron.feedback_weight = sample.copy()
                    self.neurons.append(new_neuron)

                    result["k_final"] = len(self.neurons) -1
                    result["status"] = "Add"
                    matched = True
                    
            history.append(result)

            # Reset all neuron as enable
            for neuron in self.neurons:
                neuron.status = "enable"

        return history


def dispaly_output(history):
    # Initialize groups and output vector
    size = max(res["k_final"] for res in history) + 1
    cls = [[] for _ in range(size)]
    output_vector = []

    # Printing sample status
    for idx, res in enumerate(history):
        output_vector = [0] * max(len(output_vector), res["k_final"] + 1)
        output_vector[res["k_final"]] = 1
        cls[res["k_final"]].append(idx)
        print(f'input idx: {idx}, k_first: {res["k_first"]}, k_final: {res["k_final"]}, output_vector: {output_vector} ({res["status"]})')

    # Printing grouping result  
    for idx, c in enumerate(cls):
        print(f"class {idx}: {c}")


if __name__ == "__main__":
    # Load data
    data = np.genfromtxt("data/situations_data.csv", delimiter=",", skip_header=1)
    data = data[:, 1:-1]

    # Run the ART
    art = ART(num_features=data.shape[1], vigilance=0.7)
    history = art.train(data)

    dispaly_output(history)
