import os
import numpy as np
import matplotlib.pyplot as plt


def save_learning_curves(path):
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
        plt.close()


def get_result(path):
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names