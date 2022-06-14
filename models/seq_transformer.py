import torch
import torch.nn as nn

from models.transformer import Transformer
from models.convnet import ConvEncoderNetwork
from models.regressor import RegressorHead

class SeqTransformer(nn.Module):

    def __init__(self, conv_encoder: ConvEncoderNetwork, transformer: Transformer, head: RegressorHead):
        super(SeqTransformer, self).__init__()
        self.encoder = conv_encoder
        self.transformer = transformer
        self.head = head

    def forward(self, x):
        h = self.encoder(x)
        h = self.transformer(h)
        output = self.head(h)
        return output




def build():

    encoder = ConvEncoderNetwork(
        n_channels=3,
        feat_size=512,
    )
    transformer = Transformer(
        dim=512,
        attn_heads=4,
        n_enc_blocks=1,
        dropout_prob=0.0,
        use_global=False
    )
    head = RegressorHead(
        input_size=512,
        reg_layers=[256, 128],
        n_classes=32*3,
        dropout_prob=0.0
    )

    model = SeqTransformer(
        conv_encoder=encoder,
        transformer=transformer,
        head=head
    )
    return model



if __name__ == "__main__":
    from utils import get_n_params
    bs = 4
    chunks = 12
    chunk_size = 32
    channels = 3
    x = torch.randn(bs, chunks, channels, chunk_size)

    model = build()
    y = model(x)
    y = y.view(bs, chunks, channels, chunk_size)
    print(y.shape)
    print(get_n_params(model))

    d = torch.norm(x - y, p=2)
    print(d)