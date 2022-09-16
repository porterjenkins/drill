import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import torch
import os
import numpy as np

from sklearn.metrics import classification_report


from datasets.pump_dataset import SelfSupervisedPumpDataset
from datasets.collate_fn import collate_padded
from models.seq_transformer import build_cls
from models.model_utils import load_model_chkp
from utils import get_yaml_cfg, get_n_params


NAMES = list(SelfSupervisedPumpDataset.idx_to_cls.values())

def eval(test_cfg_path: str, model_cfg_path: str):
    test_cfg = get_yaml_cfg(test_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)
    device = torch.device('cuda' if test_cfg["optimization"]["cuda"] else 'cpu')


    run = wandb.init(
        project=test_cfg["wandb"]["project"],
        entity=test_cfg["wandb"]["entity"],
        group=test_cfg["wandb"]["group"],
        job_type='TEST',
        name=test_cfg["wandb"]["name"],
        config=test_cfg
    )

    model = build_cls(model_cfg, use_cuda=test_cfg["optimization"]["cuda"])

    if test_cfg["model"]["weights"] is not None:
        model = load_model_chkp(
            model=model,
            chkp_path=test_cfg["model"]["weights"],
            use_cuda=test_cfg["optimization"]["cuda"],
            strict=False
        )


    val_fpath = test_cfg["dataset"]["ctrl_file"]
    val_data = SelfSupervisedPumpDataset(
        ctrl_fpath=val_fpath,
        data_dir=test_cfg["dataset"]["data_dir"],
        chunk_length=model_cfg["meta"]["seq_len"],
        mask_prob=model_cfg["meta"]["mask_prob"],
        max_seq_len=model_cfg["meta"]["max_len"]
    )
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=1,
        collate_fn=collate_padded,
        drop_last=False
    )



    # validation loop
    val_pbar = tqdm(val_loader, total=len(val_loader))
    model.eval()
    print("\n----> Validation inference <----")

    val_targets = np.zeros(len(val_loader))
    val_pred = np.zeros(len(val_loader))
    val_probs = np.zeros((len(val_loader), 3))

    i = 0
    with torch.no_grad():
        for x, cls in val_pbar:

            signal = x["signal"].to(device)
            cls = cls.to(device)
            pos = x["pos"].to(device).squeeze(-1)
            sig_cls = x["sig_label"].to(device)

            # need [batch size, chunks, channels, chunk size]
            signal = torch.permute(signal, [0, 1, 3, 2])

            y_hat = model(signal, pos=pos)

            val_pred[i] = torch.argmax(y_hat).cpu().data.numpy()
            val_targets[i] = cls.flatten().cpu().data.numpy()

            val_probs[i] = y_hat.cpu().data.numpy()


            i += 1




    report = classification_report(val_targets, val_pred, target_names=NAMES)
    print(report)
    run.log({"val/loss": 0, "val/acc": 0})
    print(val_probs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--test-cfg', type=str, default="./eval_cfg_cls.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_classifier.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    eval(args.test_cfg, args.model_cfg)
