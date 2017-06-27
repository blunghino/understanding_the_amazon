from matplotlib import pyplot as plt

from plotting_tools import plots_from_mat


if __name__ == "__main__":
    root = "resnet18_7channel"
    matfile = "{}_training_data.mat".format(root)
    plots_from_mat(matfile, root)
    plt.show()
