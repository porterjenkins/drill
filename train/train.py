import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import torch

from datasets.pump_dataset import SelfSupervisedPumpDataset
from datasets.collate_fn import collate_padded
from models.seq_transformer import build
from utils import get_yaml_cfg




def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)

    run = wandb.init(
        project=trn_cfg["wandb"]["project"],
        entity=trn_cfg["wandb"]["entity"],
        group=trn_cfg["wandb"]["group"],
        job_type='TRAIN',
        name=trn_cfg["wandb"]["name"],
        config=trn_cfg
    )

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
        batch_size=trn_cfg["optimization"]["batch_size"],
        collate_fn=collate_padded
    )

    for i in range(trn_cfg["optimization"]["n_epochs"]):
        print(f"\nStarting epoch {i+1}/{trn_cfg['optimization']['n_epochs']}\n")

        j = 0
        for x, y in tqdm(trn_loader, total=len(trn_loader)):
            signal = x["signal"]
            mask = x["mask"]
            if i == 0 and j == 0:
                for k in range(signal.shape[0]):
                    plot = trn_data.plot_batch(signal[k], y[k], mask[k], drop_zero=True)
                    run.log({"signal": wandb.Image(plot)})


            # need [batch size, chunks, channels, chunk size]
            signal = torch.permute(signal,[0, 1, 3, 2])
            y_hat = model(signal, mask)

            j += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
