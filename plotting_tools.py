from torch import np
from matplotlib import pyplot as plt
from scipy.io import loadmat


def plot_validation_accuracy(acc, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    plt.plot(acc)
    plt.xlabel("Epoch")
    plt.ylabel("F2 Score")
    return fig

def plot_loss(loss, num_epochs, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    t = np.linspace(0, num_epochs, len(loss))
    plt.plot(t, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Multi Label Soft Margin Loss")
    return fig

def accuracy_and_loss_plots_from_mat(mat_file_path, save=True):
    data = loadmat(mat_file_path)
