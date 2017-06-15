import os

import torch
from torch import np
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.io import savemat, loadmat
from PIL import Image



mpl.rcParams['font.size'] = 20

def save_accuracy_and_loss_mat(save_mat_path, train_acc, val_acc, loss,
                               num_epochs, lr=0):
    dic = {"train_acc": train_acc, "val_acc": val_acc,
           "loss": loss, "num_epochs": num_epochs, "lr": lr}
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

def image_and_labels(model, loader, dtype,
                             sigmoid_threshold=0.25, correct_labels='all'):
    """
    pass in a model and dataloader, get out an image and its labels
    choose whether you want all labels correct (`correct_labels`='all'),
    all labels incorrect (`correct_labels`='none'),
    or some correct and some incorrect labels (`correct_labels`='some')

    only runs on cpu
    """
    ## Put the model in test mode
    model.eval()
    for i, (x, y) in enumerate(loader):
        print(loader.dataset.X_train[i])
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)

        y_pred = torch.sigmoid(scores).data.numpy() > sigmoid_threshold
        y = y.numpy()

        sum_prod = np.sum(y * y_pred)
        if correct_labels == 'all' and np.sum(y == y_pred) == 17:
            break
        elif correct_labels == 'none' and sum_prod == 0:
            break
        elif correct_labels == 'some' and sum_prod < np.sum(y) and sum_prod:
            break
        else:
            continue
    ## get labels
    mlb = loader.dataset.mlb
    pred_labels = mlb.inverse_transform(y_pred)
    target_labels = mlb.inverse_transform(y)
    ## load in image
    img_str = loader.dataset.X_train[i] + loader.dataset.img_ext
    load_path = os.path.join(loader.dataset.img_path, img_str)
    img = Image.open(load_path)
    ## plot
    fig = plt.figure()
    plt.imshow(img)
    plt.title(correct_labels)
    print(correct_labels)
    print("predicted", pred_labels)
    print("target", target_labels)
    print(load_path)
    print()
    return fig

