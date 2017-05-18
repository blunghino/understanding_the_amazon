from torch import np
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat


def save_accuracy_and_loss_mat(save_mat_path, train_acc, val_acc, loss, num_epochs):
    dic = {"train_acc": train_acc, "val_acc": val_acc,
           "loss": loss, "num_epochs": num_epochs}
    savemat(save_mat_path, dic)

def plot_accuracy(train_acc, val_acc, num_epochs, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    t = np.arange(num_epochs) + 1
    plt.plot(t, val_acc, label="Validation")
    t = np.linspace(0, num_epochs, len(train_acc))
    plt.plot(t, train_acc, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("F2 Score")
    plt.legend()
    return fig

def plot_loss(loss, num_epochs, figsize=(8,6)):
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
