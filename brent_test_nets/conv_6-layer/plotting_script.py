from matplotlib import pyplot as plt

from plotting_tools import plots_from_mat


if __name__ == "__main__":
  matfile = "conv_6-layer_loss_and_acc.mat"
  plots_from_mat(matfile, "conv_6-layer")
  plt.show()
