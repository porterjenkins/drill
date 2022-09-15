import torch
import torch.nn as nn

class ConvEncoderNetwork(nn.Module):
    def __init__(self, n_channels: int, feat_size: int):
        super(ConvEncoderNetwork, self).__init__()

        output = 3  # number of options for classification for using Cross Entropy Loss
        #     print(dataset.data_out.size())
        A = n_channels  # number of channels of input data
        B = 32
        C = B * 2  # 64
        D = C * 2  # 128
        E = D * 2  # 256
        F = E * 2  # 512
        ks1 = 4  # was 4 # these need to be greater than 1 to take advantage of the convolutions; otherwise it is just like doing a linear layer
        ks2 = 4  # was 4
        ks3 = 4
        # Activation functions: nn.ReLU()  nn.SELU()  nn.Tanh()  nn.LeakyReLU(negative_slope=neg_slope)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.convolutions = nn.Sequential(
        self.c1 = nn.Conv1d(A, B, kernel_size=ks1)
        #nn.ReLU(),
        # nn.Dropout(p=0.25),
        #self.c2 = nn.Conv1d(B, C, kernel_size=ks2)
        #nn.ReLU(),
        # nn.Dropout(p=0.25),
        #self.c3 = nn.Conv1d(C, D, kernel_size=ks3, stride=1, padding=1)  # was stride = 2
        #nn.ReLU(),
        # nn.Dropout(p=0.25),
        #self.c4 = nn.Conv1d(D, E, kernel_size=ks3, stride=1, padding=1)  # was stride = 2
        #nn.ReLU(),
        # nn.Dropout(p=0.25),
        #self.c5 = nn.Conv1d(E, F, kernel_size=ks3, stride=1, padding=1)  # was stride = 2
        #)
        self.linear = nn.Linear(C, feat_size)
        """self.linear = nn.Sequential(
            # nn.Linear(size_in, 100),
            # nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(size_in, output),
        )"""

    def forward(self, x):
        bs, n, c, l = x.shape
        x = x.view(bs*n, c, l)
        h = self.relu(self.c1(x))
        h = self.maxpool(h)
        #h = self.relu(self.c2(h))

        #h = self.maxpool(h)
        #h = self.relu(self.c3(h))
        #h = self.maxpool(h)
        #h = self.relu(self.c4(h))
        #h = self.maxpool(h)
        #h = self.relu(self.c5(h))
        #h = self.maxpool(h)
        h = h.flatten(start_dim=1)
        h = self.relu(self.linear(h))
        return h.view(bs, n, -1)


if __name__ == "__main__":

    bs = 4
    chunks = 12
    chunk_size = 32
    channels = 3
    feat_size = 512

    model = ConvEncoderNetwork(n_channels=channels, feat_size=feat_size)
    x = torch.randn(chunks, channels, chunk_size)
    y_hat = model(x)
