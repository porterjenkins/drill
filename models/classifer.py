import torch
import torch.nn as nn
from collections import OrderedDict
#from models.seq_transformer import SeqTransformer, build
from typing import Optional


class ClassifierHead(nn.Module):
    """
        Regressor with Fully Connected Layers
    """

    def __init__(self, input_size, reg_layers, n_classes, dropout_prob):
        super(ClassifierHead, self).__init__()
        self.reg_layers = reg_layers
        self.input_size = input_size
        #self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob
        self.weights = self._build_mlp(input_size, reg_layers)
        self.output = nn.Linear(self.reg_layers[-1], n_classes)
        self.softmax = nn.Softmax()

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
        y_hat = self.softmax(y_hat)

        return y_hat


