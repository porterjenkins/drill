import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class PumpDataset(Dataset):
    """Pump dataset."""

    idx_to_cls = {
        1: "child",
        2: "woman",
        3: "man"
    }

    def __init__(self, ctrl_fpath: str, data_dir: str, transform=None):
        """

        :param ctrl_fpath: filepath to control file
        :param data_dir: data directory of segment files
        :param transform: (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        ctrl_file = pd.read_csv(ctrl_fpath, header=None)
        self.examples = ctrl_file[0].values
        self.label_data = ctrl_file[1].values
        self.transform = transform

    def get_label(self, label):
        if label == 1:
            string = "C"
        elif label == 2:
            string = "W"
        elif label == 3:
            string = "M"
        else:
            string = "Wut"
        return string

    def display_data(self, sample):
        signal = sample['signal']
        label = sample['label']

        plt.plot(signal)
        plt.title(label)

        plt.figure()
        plt.title("fft")
        for i in range(3):
            fft = np.fft.fft(signal[:, i])
            freq = np.fft.fftfreq(signal[:, i].shape[-1])
            plt.plot(freq, np.abs(fft))
        plt.xlim([0, .5])
        plt.ylim([0, 2e5])

        plt.figure()
        plt.title("copy pasta regrssion model")
        plt.plot(angles(signal))


    @staticmethod
    def calc_encoded_positions(bs, chunks, channels):
        """
        :param bs: batch size
        :param chunks: number of chunks
        :return:
        """
        # create empty tensor to be filled with encoded positions
        pe = torch.zeros(chunks, channels)
        # populate tensor with encoded positions
        for pos in range(chunks):
            for i in range(0, channels, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / channels)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / channels)))

        # convert tensor to shape of h (bs, chunks, channels) and return
        return pe.unsqueeze(0).expand(bs, chunks, channels)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx: int):

        #start_time = self.label_data[idx, 9]
        #end_time = self.label_data[idx, 10]
        #sensor_data = self.drill_data[
        #    np.where((self.drill_data[:, 0] > start_time) * (self.drill_data[:, 0] < end_time))]
        #signal = sensor_data[:, 2:5]
        #label = self.get_label(self.label_data[idx, 1])


        example_id = self.examples[idx]
        example_fpath = os.path.join(self.data_dir, example_id + ".csv")
        if example_fpath is None:
            raise FileNotFoundError(f"No segment filepath was found: {example_fpath}")

        signal = pd.read_csv(
            os.path.join(
                self.data_dir,
                example_fpath
            ),
            index_col=None,
            header=None,
            names=["time", "dt", "h1", "h2", "h3", "angle", "angle_2", "matlab_ts", "user", "cls", "pred"]
        )
        signal = signal[["angle"]].values

        label = self.label_data[idx] - 1 # 0 index labels
        sample = {'signal': signal, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return signal, label



class SelfSupervisedPumpDataset(PumpDataset):
    """Pump dataset."""

    def __init__(
            self,
            ctrl_fpath: str,
            data_dir: str,
            chunk_length: int = 16,
            rand_chunk_rate: float = 0.2,
            mask_prob: float = 0.15,
            transform: Optional[Callable] = None,
            max_seq_len: int = 512
    ):

        super(SelfSupervisedPumpDataset, self).__init__(
            ctrl_fpath,
            data_dir,
            transform
        )
        self.chunk_length = chunk_length
        self.rand_chunk_rate = rand_chunk_rate
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len

    @staticmethod
    def get_chunks(signal, chunk_length, rand_pct=0.1):

        # TODO:
        #if len_user <= chunk_length:  # skip users whose total lengths are less than the chunk length
        #    continue  # go to the next iteration

        duration_ind, n_channels = signal.shape

        num_chunks = duration_ind // chunk_length  # Starting from beginning of array, get chunks of size chunk_length

        # Create and combine chunks
        #new_chunks = np.zeros([1, chunk_length, 2], dtype=float)  # initialize array for concatenation in the loop
        new_chunks = np.zeros((num_chunks, chunk_length, n_channels), dtype=float)
        for j in range(num_chunks):
            new_chunk = signal[np.newaxis, 0 + j * chunk_length:chunk_length + j * chunk_length, :]
            new_chunks[j] = new_chunk



        # Also augment by getting random chunks
        num_rand_chunks = int(rand_pct*num_chunks)  # changed from 3 to 10 for only Day 1 data; improved acc from 50% to 100%
        rand_chunks = np.zeros((num_rand_chunks, chunk_length, n_channels), dtype=float)

        #for j in range(num_rand_chunks):
        #    start_ind = np.random.randint(0, duration_ind - chunk_length)
        #    new_chunk = signal[np.newaxis, start_ind:(start_ind + chunk_length), : ]
        #    rand_chunks[j] = new_chunk - new_chunk.mean(axis=-1, keepdims=True)  # Normalizing each chunk to have a mean of 0

        return new_chunks, num_chunks, rand_chunks


    @staticmethod
    def plot_batch(x, y, mask, drop_zero=False, figsize=(12, 6)):

        if isinstance(x, torch.Tensor):
            x = x.data.numpy()

        start_idx = 0
        _, chunk_size, n_channel = x.shape



        fig = plt.figure(figsize=figsize)

        cls = int(y) + 1 # undo zero-index (file is 1-indexed)
        plt.title(f"{SelfSupervisedPumpDataset.idx_to_cls[cls]}")
        for j in range(x.shape[0]):
            chunk_color = np.random.rand(3, )
            is_mask = bool(mask[j])
            signal = x[j, :, :]

            if is_mask:
                style = (0, (1, 10))
                c = 'gray'
            else:
                c = chunk_color
                style = "dotted"

            if drop_zero and np.all(signal) == 0.0:
                continue

            plt.plot(
                np.arange(start_idx, start_idx + chunk_size),
                signal,
                linestyle=style,
                c=c,
                alpha=0.5
            )

            start_idx += chunk_size
        return fig




    def __getitem__(self, idx):
        signal, label = PumpDataset.__getitem__(self, idx)
        signal, n_chunks, rand = self.get_chunks(signal, self.chunk_length, self.rand_chunk_rate)

        if n_chunks > self.max_seq_len:
            # random start
            start_idx = np.random.randint(0, n_chunks - self.max_seq_len)
            end_idx = start_idx + self.max_seq_len
            signal = signal[start_idx:end_idx, :, :]
            n_chunks = self.max_seq_len

        sig_label = torch.Tensor(np.repeat(label, n_chunks)).long()


        # TODO: concat tensors
        signal = torch.Tensor(signal).float()

        # generate mask tokens
        mask = torch.Tensor(np.random.binomial(1, p=self.mask_prob, size=n_chunks)).long()

        pos = torch.arange(1, n_chunks+2).long()
        #pos = self.calc_encoded_positions(1, n_chunks, 512)
        # denoiuse
        #n, s, c = signal.shape
        #eps =  torch.randn((n, s, c)) / 10
        #signal += eps

        x = {
                "signal": signal,
                "mask": mask,
                "out_signal": signal.clone().detach(),
                "pos": pos,
                "sig_label": sig_label
        }

        return x, torch.Tensor([label]).long()


if __name__ == "__main__":
    """pump_dataset = PumpDataset(
        ctrl_fpath="/home/porter/code/drill/Preprocessing/key_train.csv",
        data_dir='/home/porter/code/drill/Preprocessing/Segments/'
    )
"""
    pump_dataset = SelfSupervisedPumpDataset(
        ctrl_fpath="/home/porter/code/drill/Preprocessing/key_train.csv",
        data_dir='/home/porter/code/drill/Preprocessing/Segments/',
        chunk_length=32,
        rand_chunk_rate=0.1,
        mask_prob=0.15
    )

    from tqdm import tqdm
    for i in tqdm(range(len(pump_dataset))):
        x, y = pump_dataset[i]
        if i == 0:
            signal = x['signal']
            plt.plot(signal.reshape(-1, 3), alpha=0.5)
            plt.show()
            #for j in range(rand.shape[0]):
            #    plt.plot(rand[j,:, 0], linestyle='--')

            break