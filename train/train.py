import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import torch

from datasets.pump_dataset import SelfSupervisedPumpDataset
from datasets.collate_fn import collate_padded
from models.seq_transformer import build
from utils import get_yaml_cfg


def calculate_loss(y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)

    wandb.login(key="12e86495ba1196625be91161799a65f1f56aa345")

    # run = wandb.init(
    #     project=trn_cfg["wandb"]["project"],
    #     entity=trn_cfg["wandb"]["entity"],
    #     group=trn_cfg["wandb"]["group"],
    #     job_type='TRAIN',
    #     name=trn_cfg["wandb"]["name"],
    #     config=trn_cfg
    # )

    model = build(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(trn_cfg["optimization"]["lr"]))

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

    best_loss = float("inf")
    #wouldn't this just be pre training?
    for i in range(1):
    # for i in range(trn_cfg["optimization"]["n_epochs"]):
        print(f"\nStarting epoch {i+1}/{trn_cfg['optimization']['n_epochs']}\n")

        j = 0
        for x, y in tqdm(trn_loader, total=len(trn_loader)):
            signal = x["signal"]
            mask = x["mask"]
            # if i == 0 and j == 0:
                # for k in range(signal.shape[0]):
                #     plot = trn_data.plot_batch(signal[k], y[k], mask[k], drop_zero=True)
                #     run.log({"signal": wandb.Image(plot)})


            # need [batch size, chunks, channels, chunk size]
            og_signal = signal
            signal = torch.permute(signal,[0, 1, 3, 2])

            y_hat = model(signal, mask)

            #y is shape (4, 299, 1), y_hat is shape (4, 300, 96)... why?



            loss = calculate_loss(y_hat, y)

            # Implement backward pass, zero gradient etc...
            # Implement the optimizer and loss function
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trn_cfg["optimization"]["clip_grad_norm"])
            model.optimizer.step()

            j += 1

        # Checkpoint model after each epoch:
        #     Log the best model "best.pt" (best model to this point)
        #     Log the last model "last.pt"
        # torch.save(model.state_dict(), f"{run.dir}/last.pt")
        # if model(signal, mask).item() < best_loss:
        #     torch.save(model.state_dict(), f"{run.dir}/best.pt")
        #     best_loss = model(signal, mask).item()




    # Implement a validation data loader and validation loop.
    # Run experiments, verify that model is able to learn

    # Calculate loss over the full validation set; not an average over validation batches. Log to wandb.
def val(model, val_cfg_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
