from matplotlib import pyplot as plt

from plotting_tools import triple_plots_from_mat


if __name__ == "__main__":
    names = ['rgb', 'inf', 'grad']
    matfile = "triple_resnet18_{}_data_fc.mat"
    matfile2 = "triple_resnet18_{}_data_tune.mat"
    triple_plots_from_mat(matfile2, names, "triple_resnet18_tune")
    plt.show()
