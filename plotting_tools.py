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

def accuracy_and_loss_plots_from_mat(mat_file_path, save_name=""):
    data = loadmat(mat_file_path)
    acc = data["acc"].flatten()
    loss = data["loss"].flatten()
    num_epochs = data["num_epochs"]
    fig_acc = plot_validation_accuracy(acc)
    fig_loss = plot_loss(loss, num_epochs)
    if save_name:
        fig_acc.savefig("plot_validation_accuracy_{}.png".format(save_name), dpi=300)
        fig_loss.savefig("plot_loss_{}.png".format(save_name), dpi=300)
    return fig_acc, fig_loss
