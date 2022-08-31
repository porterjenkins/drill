import torch
import torch.nn as nn
from models.transformer import Transformer
from models.convnet import ConvEncoderNetwork
from models.regressor import RegressorHead
import numpy as np
from typing import Optional

class SeqTransformer(nn.Module):

    def __init__(
            self,
            conv_encoder: ConvEncoderNetwork,
            transformer: Transformer,
            head: RegressorHead,
            dim: int,
            seq_len: int = 32,
            use_cuda: bool = False

    ):
        super(SeqTransformer, self).__init__()
        self.encoder = conv_encoder
        self.transformer = transformer
        self.head = head
        self.mask_token = nn.Embedding(1, seq_len)
        self.cls_token = nn.Embedding(1, dim)

        if use_cuda:
            self = self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        stop = 0

    def forward(self, seq: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """

        :param seq: (torch.Tensor: float64) dims (bs, chunks, channels, chunk_size)
        :param pos: (torch.Tensor: float64) positional encoding (bs, chunks, channels, chunk_size)
        :param mask: torch.Tensor: int64): dims (bs, chunks)
        :return:
        """
        if mask is not None:
            h_mask = self.mask_token(torch.zeros(1).long())
            mask_idx = torch.where(mask)
            seq[mask_idx[0], mask_idx[1], :, :] = h_mask
        # CNN encoding
        h = self.encoder(seq)
        if pos is not None:
            h = h + pos
        # append cls token
        bs = seq.shape[0]

        #the next two lines add the extra cls token to the end of each sequence (changing the dim from (bs, chunks, channels, chunk_size) to (bs, chunks, channels + 1, chunk_size)
        cls = self.cls_token(torch.zeros(bs).long().to(self.device))
        h = torch.cat([cls.unsqueeze(1), h], dim=1)

        h = self.transformer(h)
        output = self.head(h)
        return output

def build(cfg: dict, use_cuda: bool):

    output_dim = cfg["encoder"]["n_channels"]*cfg["meta"]["seq_len"]

    encoder = ConvEncoderNetwork(
        n_channels=cfg["encoder"]["n_channels"],
        feat_size=cfg["meta"]["feat_size"],
    )
    transformer = Transformer(
        dim=cfg["meta"]["feat_size"],
        attn_heads=cfg["transformer"]["attn_heads"],
        n_enc_blocks=cfg["transformer"]["n_encoder_blocks"],
        dropout_prob=cfg["transformer"]["dropout"],
    )
    head = RegressorHead(
        input_size=cfg["meta"]["feat_size"],
        reg_layers=cfg["regressor"]["reg_layers"],
        n_classes=output_dim,
        dropout_prob=cfg["regressor"]["dropout"]
    )

    model = SeqTransformer(
        conv_encoder=encoder,
        transformer=transformer,
        head=head,
        dim=512,
        seq_len=32,
        use_cuda=use_cuda
    )
    return model

def get_masked_tensor(mask, src):
    idx = torch.where(mask)
    output = src[idx[0], idx[1]]
    return output



if __name__ == "__main__":
    from utils import get_n_params, get_yaml_cfg
    bs = 4
    chunks = 12
    chunk_size = 32
    channels = 1
    # following 2 lines are dataloaded
    x = torch.randn(bs, chunks, channels, chunk_size)

    mask = torch.randint(0, 2, size=(4, 12))
    cfg = get_yaml_cfg("../models/cfg_seq_transformer.yaml")
    model = build(cfg, use_cuda=False)
    y = model(x, mask=mask)


    #use the following 3 lines as loss
    y_for_loss = get_masked_tensor(mask, x.flatten(2))
    y_hat_for_loss = get_masked_tensor(mask, y[:, 1:, :])
    d = torch.norm(y_for_loss - y_hat_for_loss, p=2)
    print(d)

    print(get_n_params(model))
