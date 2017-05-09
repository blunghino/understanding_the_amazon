"""
boiler plate code to run simple training examples.
Also analysis code to be reused (eg plotting functions)
some code from cs231n assignment 2 pytorch ipynb
"""
from torch.autograd import Variable 
from torch import nn


def train(loader_train, model, loss_fn, optimizer, dtype, 
          num_epochs=1, print_every=1e6):
    """
    train `model` on data from `loader_train`

    inputs:
    `loader_train` object subclassed from torch.data.DataLoader 
    `model` neural net, subclassed from torch.nn.Module 
    `loss_fn` loss function see torch.nn for examples
    `optimizer` subclassed from torch.optim.Optimizer
    `dtype` data type for variables 
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)

    from cs231n assignment 2
    """
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader, dtype):
    """
    function to check performance of trained model against validation dataset
    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data 
    `dtype` data type for variables 
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)

    from cs231n assignment 2
    """
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

class Flatten(nn.Module):
    """
    flatten a 4d tensor into a 2d tensor

    from cs231n assignment 2
    """
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
