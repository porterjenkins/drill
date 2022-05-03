import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def angles(raw_hall):
  adj_hall = np.where(raw_hall[:,2] < h32, raw_hall[:,2], np.where(raw_hall[:,2] < h21, raw_hall[:,1]+h32b, raw_hall[:,0]+h21b))
  return poly(adj_hall)


class PumpDataset(Dataset):
    """Pump dataset."""

    def __init__(self, drill_data, label_data, transform=None):
        """
        Args:
            drill_data (string): Path to the csv file with raw sensor data.
                schema:
                 - timestamp, ms elapsed, sensor, sensor, sensor
            label_data (string): Path to the csv file with labels for the sensor data.
                schema:
                -  User#, Label, Start time, End time
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.drill_data = pd.read_csv(drill_data, header=None).values
        df = pd.read_csv(label_data)
        df = df.replace(['C', 'W', 'M'], [1, 2, 3])
        self.label_data = df.to_numpy()
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

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_time = self.label_data[idx, 9]
        end_time = self.label_data[idx, 10]
        sensor_data = self.drill_data[
            np.where((self.drill_data[:, 0] > start_time) * (self.drill_data[:, 0] < end_time))]
        signal = sensor_data[:, 2:5]
        #label = self.get_label(self.label_data[idx, 1])
        label = self.label_data[idx, 5]
        sample = {'signal': signal, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return signal, label





class SelfSupervisedPumpDataset(PumpDataset):
    """Pump dataset."""

    def __init__(self, drill_data: str, label_data: str, transform=None, chunk_length: int = 16):
        super(SelfSupervisedPumpDataset, self).__init__(
            drill_data,
            label_data,
            transform
        )
        self.chunk_length = chunk_length

    @staticmethod
    def get_chunks(signal, chunk_length):

        # TODO:
        #if len_user <= chunk_length:  # skip users whose total lengths are less than the chunk length
        #    continue  # go to the next iteration

        duration_ind, n_channels = signal.shape

        num_chunks = duration_ind // chunk_length  # Starting from beginning of array, get chunks of size chunk_length

        # Create and combine chunks
        #new_chunks = np.zeros([1, chunk_length, 2], dtype=float)  # initialize array for concatenation in the loop
        new_chunks = np.zeros((num_chunks, chunk_length, 3), dtype=float)
        for j in range(num_chunks):
            new_chunk = signal[np.newaxis, 0 + j * chunk_length:chunk_length + j * chunk_length, :]
            new_chunks[j] = new_chunk / new_chunk.mean(axis=-1)


        # normalize chunks
        #new_chunks[]

        return new_chunks





    def __getitem__(self, idx):
        signal, label = PumpDataset.__getitem__(self, idx)
        signal = self.get_chunks(signal, self.chunk_length)

        return signal, label


if __name__ == "__main__":
    """pump_dataset = PumpDataset(drill_data='../data/2022-03-01_to_2022-03-03/sensor-data.csv',
                               label_data='../data/2022-03-01_to_2022-03-03/label-data.csv')"""

    pump_dataset = SelfSupervisedPumpDataset(drill_data='../data/2022-03-01_to_2022-03-03/sensor-data.csv',
                               label_data='../data/2022-03-01_to_2022-03-03/label-data.csv')

    for i in range(len(pump_dataset)):
        x, y = pump_dataset[i]


        if i == 3:
            plt.show()
            break