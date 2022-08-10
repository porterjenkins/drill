import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from datasets.pump_dataset import SelfSupervisedPumpDataset
from models.seq_transformer import build
from utils import get_yaml_cfg



def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)

    model = build(model_cfg)

    trn_data = SelfSupervisedPumpDataset(
        ctrl_fpath=trn_cfg["dataset"]["ctrl_file"],
        data_dir=trn_cfg["dataset"]["data_dir"],
        chunk_length=model_cfg["meta"]["seq_len"],
        rand_chunk_rate=trn_cfg["dataset"]["rand_chunk_prob"],
        mask_prob=model_cfg["meta"]["mask_prob"]
    )
    trn_loader = DataLoader(
        trn_data,
        shuffle=True,
        batch_size=trn_cfg["optimization"]["batch_size"]
    )

    for i in range(trn_cfg["optimization"]["n_epochs"]):
        print(f"\nStarting epoch {i+1}/{trn_cfg['optimization']['n_epochs']}\n")
        for inputs, aug, mask, cls in tqdm(trn_loader, total=len(trn_data)):
            # need [batch size, chunks, channels, chunk size]
            inputs = torch.permute(inputs,[0, 1, 3, 2])
            y = model(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
