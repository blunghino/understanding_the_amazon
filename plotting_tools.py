from torch import np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.io import savemat, loadmat


mpl.rcParams['font.size'] = 20

def save_accuracy_and_loss_mat(save_mat_path, train_acc, val_acc, loss, num_epochs):
    dic = {"train_acc": train_acc, "val_acc": val_acc,
           "loss": loss, "num_epochs": num_epochs}
    savemat(save_mat_path, dic)

def plot_accuracy(train_acc, val_acc, num_epochs, figsize=(10,7)):
    fig = plt.figure(figsize=figsize)
    t = np.linspace(0, num_epochs, len(train_acc))
    plt.plot(t, train_acc, alpha=0.5, color='#1f77b4', label="Training")
    t = np.arange(num_epochs) + 1
    plt.plot(t, val_acc, color='#1f77b4', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("F2 Score")
    plt.legend()
    return fig

def plot_loss(loss, num_epochs, figsize=(10,7)):
    fig = plt.figure(figsize=figsize)
    t = np.linspace(0, num_epochs, len(loss))
    plt.plot(t, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Multi Label Soft Margin Loss")
    return fig

def plots_from_mat(mat_file_path, save_name=""):
    """
    load from matfile with vars "acc", "loss", and "num_epochs"
    """
    data = loadmat(mat_file_path)
    train_acc = data["train_acc"].flatten()
    val_acc = data["val_acc"].flatten()
    loss = data["loss"].flatten()
    num_epochs = data["num_epochs"].flatten()

    fig_acc = plot_accuracy(train_acc, val_acc, num_epochs)
    fig_loss = plot_loss(loss, num_epochs)
    if save_name:
        fig_acc.savefig("fig/plot_validation_accuracy_{}.png".format(save_name), dpi=300)
        fig_loss.savefig("fig/plot_loss_{}.png".format(save_name), dpi=300)
    return fig_acc, fig_loss

def triple_plots_from_mat(mat_file_path_root, model_names, save_name=""):
    """
    load from matfile with vars "acc", "loss", and "num_epochs"
    """
    fig_acc = plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.xlabel("Epoch")
    plt.ylabel("F2 Score")
    fig_loss = plt.figure(figsize=(9,8))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.xlabel('Epoch')
    axs = [ax1, ax2, ax3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, name in enumerate(model_names):
        data = loadmat(mat_file_path_root.format(name))
        train_acc = data["train_acc"].flatten()
        val_acc = data["val_acc"].flatten()
        loss = data["loss"].flatten()
        num_epochs = data["num_epochs"].flatten()
        ax.plot(np.linspace(0, num_epochs, len(train_acc)), train_acc,
                color=colors[i], alpha=0.35, label="Train {}".format(name))
        ax.plot(np.arange(num_epochs) + 1, val_acc, color=colors[i], lw=2,
                label="Val {}".format(name), zorder=100-i)
        axs[i].plot(np.linspace(0, num_epochs, len(loss)), loss, label=name,
                    color=colors[i])
        axs[i].legend(loc='upper right')
    leg = ax.legend(ncol=3)
    plt.setp(leg.get_texts(), fontsize='16')
    axs[1].set_ylabel("Multi-Label Soft Margin Loss")
    if save_name:
        fig_acc.savefig("fig/plot_validation_accuracy_{}.png".format(save_name), dpi=300)
        fig_loss.savefig("fig/plot_loss_{}.png".format(save_name), dpi=300)
    return fig_acc, fig_loss
