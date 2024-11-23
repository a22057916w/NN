import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    def __init__(self, beta=1.0):
        self.W = None
        self.beta = beta

    def activation(self, u):
        return (1 - np.exp(-self.beta * u)) / (1 + np.exp(-self.beta * u))

    def train(self, patterns):
        """
        訓練 Hopfield 網路的權重矩陣
        :param patterns: 訓練樣本，每一行是一個 pattern
        :return: None
        """
        num_neurons = patterns.shape[1]
        self.W = np.zeros((num_neurons, num_neurons))
        
        # 使用外積法更新權重
        for p in patterns:
            self.W += np.outer(p, p)
        
        # 移除自連接
        np.fill_diagonal(self.W, 0)
        
        self.W /= patterns.shape[0]

    def recall(self, pattern, max_iters=10, tol=1e-5):
        """
        召回過程，使用 Hopfield 網路進行模式復原，直到穩定狀態
        :param pattern: 輸入模式
        :param max_iters: 最大迭代次數
        :param tol: 停止條件的容忍度
        :return: 復原的模式
        """
        num_neurons = pattern.shape[0]
        for _ in range(max_iters):
            new_pattern = self.activation(np.dot(self.W, pattern))
            # 將激活後的輸出二值化，符合 Hopfield 網路特徵
            new_pattern = np.sign(new_pattern)
            new_pattern[new_pattern == 0] = 1
            
            # 如果新模式與之前的模式相同，則認為已達到穩定狀態
            if np.allclose(new_pattern, pattern, atol=tol):
                break
            pattern = new_pattern
        
        return pattern

def plot_pattern(pattern, title):
    """
    繪製給定的二元模式圖像
    :param pattern: 要繪製的模式 (1D 數組)
    :param title: 圖像的標題
    """
    pattern_2d = pattern.reshape((9, 5))
    plt.imshow(pattern_2d, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def add_noise_to_pattern(pattern, noise_level):
    """
    隨機修改給定百分比的像素值
    :param pattern: 原始模式
    :param noise_level: 隨機改變的像素比例 (0-100)
    :return: 含噪聲的模式
    """
    noisy_pattern = pattern.copy()
    num_pixels_to_change = int(len(pattern) * noise_level / 100)
    indices = random.sample(range(len(pattern)), num_pixels_to_change)
    for idx in indices:
        noisy_pattern[idx] = -noisy_pattern[idx]  # 改變像素值
    return noisy_pattern

def main():
    # 設定 4 個 9x5 的輸入圖片，並轉換成 1 維向量
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
    
    patterns = np.array([img1.flatten(), img2.flatten(), img3.flatten(), img4.flatten()])
    patterns = np.where(patterns == 0, -1, 1)  # 將 0 轉換為 -1，1 保持不變
    
    # 創建 Hopfield 網路實例並訓練權重矩陣
    hnn = HopfieldNetwork(beta=1.0)
    hnn.train(patterns)
    
    # 繪製原始模式
    for i, pattern in enumerate(patterns):
        plot_pattern(pattern, title=f"Original Pattern {i+1}")
    
    # 測試網路對扭曲的 pattern 進行復原
    noise_level = 20  # 隨機改變 20% 的像素
    for i, original_pattern in enumerate(patterns):
        for j in range(3):  # 每個數字生成 3 個測試圖片
            test_pattern = add_noise_to_pattern(original_pattern, noise_level)
            plot_pattern(test_pattern, title=f"Test Pattern {i+1}-{j+1} (Distorted)")
            recovered_pattern = hnn.recall(test_pattern)
            plot_pattern(recovered_pattern, title=f"Recovered Pattern {i+1}-{j+1}")

if __name__ == "__main__":
    main()
