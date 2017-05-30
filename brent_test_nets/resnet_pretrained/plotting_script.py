from matplotlib import pyplot as plt

from plotting_tools import plots_from_mat


if __name__ == "__main__":
  matfile = "resnet101_pretrained_loss_and_acc_fc.mat"
  plots_from_mat(matfile, "resnet101_pretrained_fc")
  matfile2 = "resnet101_pretrained_loss_and_acc_tune.mat"
  plots_from_mat(matfile2, "resnet101_pretrained_tune")
  plt.show()
