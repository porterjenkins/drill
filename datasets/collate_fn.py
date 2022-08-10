from typing import List, Dict
import torch

from datasets.dataset_utils import get_zero_padded_batch

def collate_padded(batch: List[Dict]):
    _keys_to_stack = set([])
    _list_keys_flatten = set()
    _list_keys = set()
    _zero_pad_keys = set(["signal", "mask", "targets"])

    if batch is None or len(batch) == 0:
        return None
    x = {}

    collated_dict = {
        "targets": []
    }
    for b in batch:
        inputs, target = b
        collated_dict["targets"].append(target)
        for k, v in inputs.items():
            if k not in collated_dict:
                collated_dict[k] = []
            collated_dict[k].append(v)

    for k in _zero_pad_keys:
        #if k == "targets":
        #    x[k] = get_zero_padded_batch(collated_dict[k], flatten_last=True)
        x[k] = get_zero_padded_batch(collated_dict[k])

    for k, v in collated_dict.items():
        if k in _zero_pad_keys:
            continue
        if k in _keys_to_stack:
            x[k] = torch.stack(v, dim=0)
        elif k in _list_keys_flatten:
            x[k] = []
            for l in v:
                x[k] += l
        elif k in _list_keys:
            x[k] = []
            for l in v:
                x[k].append(l)
        else:
            x[k] = torch.cat(v)
    return x, x['targets']