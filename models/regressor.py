import torch
import torch.nn as nn

from collections import OrderedDict

class SineOutput(nn.Module):
    def __init__(self, seq_len, use_cuda:bool = False):
        super(SineOutput, self).__init__()
        self.seq_len = seq_len

        if use_cuda:
            self = self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, h):
        """

        :param h: (batch size x num params)
        :return:
        """
        bs, n_seq, n_param = h.shape

        x = torch.arange(0, self.seq_len)
        x = x.repeat((bs, n_seq, 1)).to(self.device)

        a = h[:, :, 0].unsqueeze(-1) # amplitude
        b = h[:, :, 1].unsqueeze(-1) # frequency
        c = h[:, :, 2].unsqueeze(-1) # horizontal offset
        d = h[:, :, 3].unsqueeze(-1) # vertical offset

        y = a * torch.sin(b*x + c) + d

        return y


class RegressorHead(nn.Module):
    """
        Regressor with Fully Connected Layers
    """

    def __init__(self, input_size, reg_layers, n_classes, dropout_prob, use_cuda):
        super(RegressorHead, self).__init__()
        self.reg_layers = reg_layers
        self.input_size = input_size
        #self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob
        self.weights = self._build_mlp(input_size, reg_layers)
        self.output = nn.Linear(self.reg_layers[-1], 4)
        #self.sine = SineOutput(n_classes, use_cuda)

    def _build_mlp(self, input_size, reg_layers):
        W = []
        for i, dim in enumerate(reg_layers):
            layer_name = f"layer_{i}"
            W.append((f"dropout_{i}", nn.Dropout(p=self.dropout_prob)))
            if i == 0:
                w = nn.Linear(input_size, dim)
            else:
                w = nn.Linear(reg_layers[i - 1], dim)
            W.append((layer_name, w))
            W.append((f"relu_{i}", nn.ReLU()))

        weights = nn.Sequential(
            OrderedDict(W)
        )
        return weights


    def forward(self, x):
        h = self.weights(x)
        y_hat = self.output(h)
        #y_hat = self.sine(torch.clamp(h, min=-6, max=6))
        return y_hat


if __name__ == "__main__":
    s = SineOutput(32)
    bs = 2
    n_chunk = 10
    a = torch.randint(1, 4, size=(bs, n_chunk, 1))
    b = torch.randint(1, 4, size=(bs, n_chunk, 1))
    c = torch.randint(1, 4, size=(bs, n_chunk, 1))
    d = torch.randint(1, 4, size=(bs, n_chunk, 1))
    h = torch.cat([a, b, c, d], dim=-1)


    y = s(h)
    print(y)

    """import matplotlib.pyplot as plt

    x = torch.arange(0, 32)
    x = x.repeat((bs, n_chunk, 1))

    for i in range(bs):
        for j in range(n_chunk):
            plt.plot(x[i, j, :].data.numpy(), y[i, j, :].data.numpy())

    plt.show()"""
