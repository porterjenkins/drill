import argparse
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
        drill_data='../data/2022-03-01_to_2022-03-03/sensor-data.csv',
        label_data='../data/2022-03-01_to_2022-03-03/label-data.csv',
        chunk_length=32,
        rand_chunk_rate=0.0
    )
    trn_loader = DataLoader(
        trn_data,
        shuffle=True,
        batch_size=trn_cfg["optimization"]["batch_size"]
    )

    for i in range(trn_cfg["optimization"]["n_epochs"]):
        for inputs, aug, targets in trn_loader:
            inputs = torch.permute(inputs,[0, 1, 3, 2])
            y = model(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
