import torch
import torch.nn as nn
from models.classifer import ClassifierHead

class LstmClassifier(nn.Module):

    def __init__(
            self,
            head: ClassifierHead,
            dim: int, # hidden dimension size
            use_cuda: bool = False,

    ):
        super(LstmClassifier, self).__init__()
        self.rnn = nn.LSTM(1, dim, 1, bidirectional=False, batch_first=True)
        self.head = head

        if use_cuda:
            self = self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def forward(self, seq):
        output, (hn, cn) = self.rnn(seq)
        y_hat = self.head(hn.squeeze(0))
        return y_hat



def build_cls(cfg: dict, use_cuda: bool):


    head = ClassifierHead(
        input_size=cfg["meta"]["feat_size"],
        reg_layers=cfg["classifier"]["reg_layers"],
        n_classes=cfg["classifier"]["n_cls"],
        dropout_prob=cfg["classifier"]["dropout"]
    )

    model = LstmClassifier(

        head=head,
        dim=cfg["meta"]["feat_size"],
        seq_len=cfg["meta"]["seq_len"],
        use_cuda=use_cuda
    )
    return model
