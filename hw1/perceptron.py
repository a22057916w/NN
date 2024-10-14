import numpy as np
import matplotlib.pyplot as plt


def train(X, d):
    w = [1, 0.1, 0.2]
    n = 1
    epoch = 0
    errors = []

    while True:
        total_error = 0

        for i in range(len(X)):
            u = np.dot(X[i], w)
            y = signal(u)
            err = d[i] - y
            total_error += abs(err)
            
            if err != 0:
                w = w + n * np.dot(err, X[i])     
            
            errors.append(err)
            epoch += 1

        if total_error == 0:
            break

    plot_epoch_error(epoch, errors)
    plot_decision_boundary(X, d, w)


def signal(x):
    return 1 if x >= 0 else -1

def plot_epoch_error(epoch, errors):
    plt.plot(range(epoch), errors)
    plt.xlabel("epoch")
    plt.ylabel("Error")
    plt.show()

def plot_decision_boundary(X, d, w):
    plt.figure()
    # ignore the bias term -1
    for i in range(len(X)):
        if d[i] == 1:
            plt.scatter(X[i][1], X[i][2], color='blue', marker='o')
        else:
            plt.scatter(X[i][1], X[i][2], color='red', marker='x')

    x0 = -1
    x1 = np.linspace(-1, 1, 100)
    x2 =  (-(w[1] * x1) - (w[0] * x0)) / w[2]
    plt.plot(x1, x2, 'g--')

    plt.ylim(-2, 2)
    plt.xlabel("Feature 1 (x1)")
    plt.ylabel("Feature 2 (x2)")
    plt.title("Vectors to be Classified")
    plt.show()


if __name__ == "__main__":
    samples = [[-1, -0.5, -0.5], [-1, -0.5, 0.5], [-1, 0.3, -0.5], [-1, -0.1, 1.0]]
    d = [1, 1, -1, -1]
    train(samples, d)