import torch
from torch import np
from torch.autograd import Variable
import csv
from loss import f2_score

def get_scores(models, loaders, dtype,n_classes=17):
    s = np.zeros((3, len(loaders[0].sampler), n_classes))
    y_array = np.zeros(s.shape[1:])
    bs = loaders[0].batch_size
    ## Put the model in test mode
    models[0].eval()
    models[1].eval()
    models[2].eval()
    ## loop over the three models
    for i, (model, loader) in enumerate(zip(models, loaders)):
        for j, (x, y) in enumerate(loader):
            x_var = Variable(x.type(dtype), volatile=True)
            if i == 0:
                y_array[j*bs:(j+1)*bs,:] = y.numpy()
            score_var = model(x_var)
            ## store each set of scores
            s[i,j*bs:(j+1)*bs,:] = score_var.data.numpy()
    return s,y_array
