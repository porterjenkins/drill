import numpy as np
import torch

class RunningAvgQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data=[]

    def __str__(self):
        return str(self.data)

    def add(self, x):
        self.data.append(x)
        if len(self.data) > self.maxsize:
            self.data.pop(0)

    def mean(self):
        return np.mean(self.data)


def get_sine_from_theta(theta, seq_len):

    """

    :param h: (batch size x num params)
    :return:
    """
    bs, n_seq, n_param = theta.shape

    x = torch.arange(0, seq_len)
    x = x.repeat((bs, n_seq, 1))

    a = theta[:, :, 0].unsqueeze(-1) # amplitude
    b = theta[:, :, 1].unsqueeze(-1) # frequency
    c = theta[:, :, 2].unsqueeze(-1) # horizontal offset
    d = theta[:, :, 3].unsqueeze(-1) # vertical offset

    y = a * torch.sin(b*x + c) + d

    return y