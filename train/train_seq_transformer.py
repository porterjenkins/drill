import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import torch

from datasets.pump_dataset import SelfSupervisedPumpDataset
from datasets.collate_fn import collate_padded
from models.seq_transformer import build, get_masked_tensor
from utils import get_yaml_cfg
from train.train_utils import RunningAvgQueue


def calculate_masked_loss(y_hat, y, mask):
    y_for_loss = get_masked_tensor(mask, y.flatten(2))
    y_hat_for_loss = get_masked_tensor(mask, y_hat[:, 1:, :])
    objective = torch.mean(torch.norm(y_for_loss - y_hat_for_loss, p=2, dim=-1))
    return objective


def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)
    device = torch.device('cuda' if trn_cfg["optimization"]["cuda"] else 'cpu')


    run = wandb.init(
        project=trn_cfg["wandb"]["project"],
        entity=trn_cfg["wandb"]["entity"],
        group=trn_cfg["wandb"]["group"],
        job_type='TRAIN',
        name=trn_cfg["wandb"]["name"],
        config=trn_cfg
    )

    model = build(model_cfg, use_cuda=trn_cfg["optimization"]["cuda"])
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

    queue_loss = RunningAvgQueue(trn_cfg["optimization"]["ma_lookback"])

    best_loss = float("inf")
    model.train()


    for i in range(trn_cfg["optimization"]["n_epochs"]):
        print(f"\nStarting epoch {i+1}/{trn_cfg['optimization']['n_epochs']}\n")
        epoch_loss = 0
        j = 0
        pbar = tqdm(trn_loader, total=len(trn_loader))
        for x, cls in pbar:
            signal = x["signal"].to(device)
            mask = x["mask"].to(device)
            if i == 0 and j == 0:
                for k in range(signal.shape[0]):
                    plot = trn_data.plot_batch(
                        signal[k].cpu(),
                        cls[k].cpu(),
                        mask[k].cpu(),
                        drop_zero=True
                    )
                    run.log({"signal": wandb.Image(plot)})


            # need [batch size, chunks, channels, chunk size]
            signal = torch.permute(signal,[0, 1, 3, 2])

            optimizer.zero_grad()
            y_hat = model(signal, mask)


            loss = calculate_masked_loss(y_hat, signal, mask)

            # Implement backward pass, zero gradient etc...
            # Implement the optimizer and loss function
            loss.backward()
            optimizer.step()

            pbar.set_description(
                "train loss: {:.5f}".format(loss.detach())
            )
            run.log({"train/batch-loss": loss.detach()})

            epoch_loss += loss.detach()

            j += 1

        avg_epoch_loss = epoch_loss / len(trn_loader)
        run.log({"train/loss":avg_epoch_loss})
        print(
            "\ntrain/loss {:.5f} ".format(avg_epoch_loss)
        )


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
