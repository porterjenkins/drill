from typing import List
import torch

def get_zero_padded_batch(input: List[torch.Tensor], flatten_last: bool = False):
    max_dim = 0
    batch_size = len(input)

    # make sure all inputs are at least 2D
    batch = []
    for x in input:
        if x.ndim == 1:
            batch.append(
                x.unsqueeze(-1)
            )
        else:
            batch.append(x)


    shape = list(batch[0].shape)
    shape.pop(0)
    dtype = batch[0].dtype

    for x in batch:
        n = x.shape[0]
        if n > max_dim:
            max_dim = n
    shape = [batch_size, max_dim] + shape
    padded = torch.zeros(shape,dtype=dtype)

    for i, x in enumerate(batch):
        n = x.shape[0]
        padded[i,:n, ::] = x

    if flatten_last:
        padded = padded.squeeze(-1)

    return padded