import numpy as np
import torch

# generate current, previous and next set
def set_generate(x):  # input Nx1xM signal
    set_all = []
    N, M = x.shape # M is the number of data points of each epoch, N is the number of epoch
    x_0 = x[0, :] # data of the first epoch 
    x_f = x[-1, :] # data of the last epoch
    x_current = x # current set
    x_previous = np.roll(x, 1, axis=0)
    x_previous[0, :] = x_0  # ''previous'' set
    x_next = np.roll(x, -1, axis=0)
    x_next[0, :] = x_f # ''next'' set
    x_current, x_previous, x_next = x_current.reshape(-1,1,1,3000), x_previous.reshape(-1,1,1,3000), x_next.reshape(-1,1,1,3000)
    return x_current, x_previous, x_next