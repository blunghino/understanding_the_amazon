from matplotlib import pyplot as plt

from plotting_tools import plots_from_mat


if __name__ == "__main__":
  root = "resnet18"
  matfile = "{}_loss_and_acc_fc.mat".format(root)
  plots_from_mat(matfile, "{}_fc".format(root))
  matfile2 = "{}_loss_and_acc_tune.mat".format(root)
  plots_from_mat(matfile2, "{}_tune".format(root))
  plt.show()
