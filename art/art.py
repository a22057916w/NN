import numpy as np

class Neuron:
    def __init__(self, num_features):
        self.forward_weight = np.ones(num_features) / (1 + num_features)  # 權重初始化為 1/(1+n)
        # print(f"forwrad_weight: {self.forward_weight}")
        self.feedback_weight = np.ones(num_features)  # 額外權重初始化為 1
        # print(f"feedback_weight: {self.feedback_weight}")
        self.status = "enable"  # 神經元狀態初始化為 enable
        self.activation_value = 0  # 激活值初始化為 0

    def update_weights(self, sample, winner_idx):
        self.forward_weight = (self.feedback_weight[winner_idx] * sample[winner_idx]) / (0.5 + np.dot(self.feedback_weight, sample))
        # print(f"sample: {sample}")
        # print(f"feedback_weight: {self.feedback_weight}")
        self.feedback_weight = self.feedback_weight * sample 
        # print(f"result: {self.feedback_weight}")
        

    def calculate_activation(self, sample):
        self.activation_value = np.sum(np.dot(self.forward_weight, sample))


class ART:
    def __init__(self, num_features, vigilance=0.8):
        self.num_features = num_features
        self.vigilance = vigilance
        self.neurons = [Neuron(num_features)]  # 初始化一個神經元
        self.cls = [[] * len(self.neurons)]

    def calculate_similarity(self, neuron, sample):
        return np.dot(neuron.feedback_weight, sample) / np.sum(sample)

    def find_winner(self, sample):
        max_activation = -float('inf')
        winner_idx = None

        # 計算所有神經元的激活值
        for idx, neuron in enumerate(self.neurons):
            neuron.calculate_activation(sample)

        # 找到最大激活值且啟用的神經元
        for idx, neuron in enumerate(self.neurons):
            # print(f"neuron.activation_value: {neuron.activation_value}")
            if neuron.status == "enable" and neuron.activation_value > max_activation:
                max_activation = neuron.activation_value
                winner_idx = idx

        return winner_idx

    def train(self, data):
        """
        訓練 ART-1 模型

        :param data: 輸入數據，形狀為 (樣本數, 特徵數)
        """
        for idx, sample in enumerate(data):
            print(f"============== Sample_{idx} ================")

            update_status = "Update"
            matched = False
            k_first = self.find_winner(sample)
            k_final = None
            while not matched:
                winner_idx = self.find_winner(sample)

                if winner_idx is None:
                    # 創建新分類
                    print("!!! new !!!")
                    new_neuron = Neuron(self.num_features)
                    new_neuron.forward_weight = sample.copy()
                    self.neurons.append(new_neuron)
                    self.cls.append([])
                    self.cls[-1].append(idx)
                    matched = True
                else:
                    winner_neuron = self.neurons[winner_idx]

                    # 計算相似度
                    similarity = self.calculate_similarity(winner_neuron, sample)
                    print(f"similarty: {similarity}")

                    if similarity >= self.vigilance:
                        # 更新權重
                        print(f"!!! UPDATE !!!")
                        winner_neuron.update_weights(sample, winner_idx)
                        matched = True
                        k_final = winner_idx
                        self.cls[winner_idx].append(idx)
                    else:
                        # 禁用該神經元
                        winner_neuron.status = "disable"
            print(f"len(neuron): {len(self.neurons)}")
            # 重置所有神經元狀態為 enable
            for neuron in self.neurons:
                neuron.status = "enable"

            print(f"input idx: {idx}, k_first: {k_first}, k_final: {k_final}")
            
        for idx, cls in enumerate(self.cls):
            print(f"class {idx}: {cls}")

    def predict(self, sample):
        """
        預測輸入樣本的分類

        :param sample: 輸入樣本，形狀為 (特徵數, )
        :return: 分類索引（若無匹配則返回 "New Category"）
        """
        if sample.shape[0] != self.num_features:
            raise ValueError("Sample feature size does not match the number of model features.")

        winner_idx = self.find_winner(sample)

        return winner_idx if winner_idx is not None else "New Category"


if __name__ == "__main__":
    # 初始化數據
    data = np.genfromtxt("data/situations_data.csv", delimiter=",", skip_header=1)
    data = data[:, 1:-1]

    # 初始化 ART-1 模型
    art = ART(num_features=data.shape[1], vigilance=0.6)

    # 訓練模型
    art.train(data)

    # 打印分類結果
    # print("Weights after training:")
    # for i, neuron in enumerate(art.neurons):
    #     print(f"Category {i}: {neuron.forward_weight}, Feedback Weight: {neuron.feedback_weight}, Activation Value: {neuron.activation_value}, Status: {neuron.status}")

    # # 測試新樣本
    # test_sample = np.array([1, 0, 1, 0, 1])
    # category = art.predict(test_sample)
    # print(f"Test sample belongs to category: {category}")
