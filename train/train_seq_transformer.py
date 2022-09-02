import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import torch
import os

from datasets.pump_dataset import SelfSupervisedPumpDataset
from datasets.collate_fn import collate_padded
from models.seq_transformer import build_reg, get_masked_tensor
from utils import get_yaml_cfg, get_n_params
from train.train_utils import RunningAvgQueue


def calculate_masked_loss(y_hat, y, mask):
    y_for_loss = get_masked_tensor(mask, y.flatten(2))
    y_hat_for_loss = get_masked_tensor(mask, y_hat[:, 1:, :])
    objective = torch.mean(torch.norm(y_for_loss - y_hat_for_loss, p=2, dim=-1))
    return objective

def init_chkp_dir(chkp, run_name):
    if not os.path.exists(chkp):
        os.mkdir(chkp)
    run_dir = os.path.join(chkp, run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    return run_dir

def plot_batch_pred(signal, pred, mask, k=10, figsize=(12, 5)):
    bs = signal.shape[0]
    fig, ax = plt.subplots(nrows=bs, ncols=1, figsize=figsize)
    signal = torch.permute(signal, [0, 1, 3, 2]).squeeze(-1)


    for b in range(bs):
        batch_gt = signal[b, :k]
        batch_mask = mask[b, :k].squeeze(-1)
        batch_pred = pred[b, :k]

        assert (batch_gt.shape[0] == batch_mask.shape[0]) and (batch_gt.shape[0] == batch_pred.shape[0])
        n_chunk, chunk_size = batch_gt.shape


        start_idx = 0
        for i in range(batch_gt.shape[0]):
            sig_chunk = batch_gt[i]
            pred_chunk = batch_pred[i]

            x = torch.arange(start_idx, start_idx+chunk_size)
            if batch_mask[i] == 0.0:
                ax[b].plot(x, sig_chunk, c='g')
            else:
                ax[b].plot(x, sig_chunk, c='gray', linestyle='-.', alpha=0.7)
                ax[b].plot(x, pred_chunk, c='r', linestyle='--')

            start_idx += chunk_size
    return fig

def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)
    device = torch.device('cuda' if trn_cfg["optimization"]["cuda"] else 'cpu')

    run_dir = init_chkp_dir(
        trn_cfg["optimization"]["chkp_dir"],
        trn_cfg["wandb"]["name"]
    )

    run = wandb.init(
        project=trn_cfg["wandb"]["project"],
        entity=trn_cfg["wandb"]["entity"],
        group=trn_cfg["wandb"]["group"],
        job_type='TRAIN',
        name=trn_cfg["wandb"]["name"],
        config=trn_cfg
    )

    model = build_reg(model_cfg, use_cuda=trn_cfg["optimization"]["cuda"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(trn_cfg["optimization"]["lr"]))
    n_params = get_n_params(model)
    print(f"Parameters: {n_params}")

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

    val_fpath = trn_cfg["dataset"]["ctrl_file"].replace("train", "val")
    val_data = SelfSupervisedPumpDataset(
        ctrl_fpath=val_fpath,
        data_dir=trn_cfg["dataset"]["data_dir"],
        chunk_length=model_cfg["meta"]["seq_len"],
        rand_chunk_rate=trn_cfg["dataset"]["rand_chunk_prob"],
        mask_prob=model_cfg["meta"]["mask_prob"]
    )
    val_loader = DataLoader(
        trn_data,
        shuffle=False,
        batch_size=trn_cfg["optimization"]["val_batch_size"],
        collate_fn=collate_padded,
        drop_last=False
    )

    best_loss = float("inf")

    for i in range(trn_cfg["optimization"]["n_epochs"]):
        print(f"\nStarting epoch {i+1}/{trn_cfg['optimization']['n_epochs']}\n")
        model.train()
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
                        drop_zero=False
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



        # validation loop
        val_pbar = tqdm(val_loader, total=len(val_loader))
        model.eval()
        print("\n----> Validation inference <----")

        val_targets = []
        val_pred = []
        val_masks = []
        avg_val_loss = 0
        val_cntr = 0
        for x, cls in val_pbar:

            signal = x["signal"].to(device)
            mask = x["mask"].to(device)

            # need [batch size, chunks, channels, chunk size]
            signal = torch.permute(signal, [0, 1, 3, 2])

            y_hat = model(signal, mask)
            val_loss = calculate_masked_loss(y_hat, signal, mask).detach()

            avg_val_loss += val_loss

            if val_cntr == 0:
                # plot first batch only
                fig = plot_batch_pred(
                    signal.detach().cpu(),
                    y_hat[:, 1:, ::].detach().cpu(),
                    mask.detach().cpu(),
                    k=50
                )
                run.log({"prediction": wandb.Image(fig)})

            #val_targets.append(signal.detach())
            #val_pred.append(y_hat.detach())
            #val_masks.append(mask.detach())

            val_cntr += 1

        avg_val_loss = avg_val_loss / len(val_loader)
        run.log({"val/loss": avg_val_loss})
        print(
            "\nval/loss {:.5f} ".format(avg_val_loss)
        )

        #val_targets = torch.cat(val_targets, dim=0)
        #val_pred = torch.cat(val_pred, dim=0)
        #val_masks = torch.cat(val_masks, dim=0)


        # Checkpoint model after each epoch:
        #     Log the best model "best.pt" (best model to this point)
        #     Log the last model "last.pt"

        torch.save(model.state_dict(), f"{run_dir}/last.pt")
        if avg_val_loss < best_loss:
            torch.save(model.state_dict(), f"{run_dir}/best.pt")
            #print("New best loss: {:.4f} --> {:.4f}".format(best_loss, avg_val_loss))
            best_loss = avg_val_loss


def val(model, val_cfg_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
