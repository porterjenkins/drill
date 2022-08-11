import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

# from datasets.pump_dataset import SelfSupervisedPumpDataset
# from models.seq_transformer import build
# from utils import get_yaml_cfg
import yaml
import pandas as pd
import pdb

#################### Delete from here ######################################
def print_yaml(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"\t - {k2}: {v2}")
        else:
            print(f"{k}: {v}")


def get_yaml_cfg(fpath):
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("CONFIG SETTINGS:")
    print_yaml(cfg)
    return cfg


class SeqTransformer(nn.Module):

    def __init__(
            self,
            conv_encoder: ConvEncoderNetwork,
            transformer: Transformer,
            head: RegressorHead,
            dim: int,
            seq_len: int = 32

    ):
        super(SeqTransformer, self).__init__()
        self.encoder = conv_encoder
        self.transformer = transformer
        self.head = head
        self.mask_token = nn.Embedding(1, seq_len)
        self.cls_token = nn.Embedding(1, dim)

    def forward(self, seq: torch.Tensor,  mask: Optional[torch.Tensor] = None):
        """

        :param seq: (torch.Tensor: float64) dims (bs, chunks, channels, chunk_size)
        :param mask: torch.Tensor: int64): dims (bs, chunks)
        :return:
        """
        if mask is not None:
            h_mask = self.mask_token(torch.zeros(1).long())
            mask_idx = torch.where(mask)
            seq[mask_idx[0], mask_idx[1], :, :] = h_mask
        # CNN encoding
        h = self.encoder(seq)

        pos = calc_encoded_positions(h.shape[0], h.shape[1], h.shape[2])
        h = h + pos
        # append cls token
        bs = seq.shape[0]
        cls = self.cls_token(torch.zeros(bs).long())
        h = torch.cat([cls.unsqueeze(1), h], dim=1)

        h = self.transformer(h)
        output = self.head(h)
        return output




def build(cfg: dict):

    output_dim = cfg["encoder"]["n_channels"]*cfg["meta"]["seq_len"]

    encoder = ConvEncoderNetwork(
        n_channels=cfg["encoder"]["n_channels"],
        feat_size=cfg["meta"]["feat_size"],
    )
    transformer = Transformer(
        dim=cfg["meta"]["feat_size"],
        attn_heads=cfg["transformer"]["attn_heads"],
        n_enc_blocks=cfg["transformer"]["n_encoder_blocks"],
        dropout_prob=cfg["transformer"]["dropout"],
    )
    head = RegressorHead(
        input_size=cfg["meta"]["feat_size"],
        reg_layers=cfg["regressor"]["reg_layers"],
        n_classes=output_dim,
        dropout_prob=cfg["regressor"]["dropout"]
    )

    model = SeqTransformer(
        conv_encoder=encoder,
        transformer=transformer,
        head=head,
        dim=512,
        seq_len=32
    )
    return model

def get_masked_tensor(mask, src):
    idx = torch.where(mask)
    output = src[idx[0], idx[1]]
    return output

def calc_encoded_positions(bs,chunks, channels):
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
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/channels)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/channels)))

    # convert tensor to shape of h (bs, chunks, channels) and return
    return pe.unsqueeze(0).expand(bs, chunks, channels)


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

    def __init__(
            self,
            drill_data: str,
            label_data: str,
            transform=None,
            chunk_length: int = 16,
            rand_chunk_rate: float = 0.2
    ):

        super(SelfSupervisedPumpDataset, self).__init__(
            drill_data,
            label_data,
            transform
        )
        self.chunk_length = chunk_length
        self.rand_chunk_rate = rand_chunk_rate

    @staticmethod
    def get_chunks(signal, chunk_length, rand_pct=0.1):

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
            new_chunks[j] = new_chunk / new_chunk.mean(axis=-1, keepdims=True)



        # Also augment by getting random chunks
        num_rand_chunks = int(rand_pct*num_chunks)  # changed from 3 to 10 for only Day 1 data; improved acc from 50% to 100%
        rand_chunks = np.zeros((num_rand_chunks, chunk_length, 3), dtype=float)
        for j in range(num_rand_chunks):
            start_ind = np.random.randint(0, duration_ind - chunk_length)
            new_chunk = signal[np.newaxis, start_ind:(start_ind + chunk_length), : ]
            rand_chunks[j] = new_chunk - new_chunk.mean(axis=-1, keepdims=True)  # Normalizing each chunk to have a mean of 0

            # this loop removes any chunks with large (>65 degree) jumps
            """for k in range(new_chunk.shape[0]):
                #           counter_chunks += 1
                if max(new_chunk[k, :, 0]) - min(new_chunk[k, :, 0]) < 65:
                    #             counter_recorded += 1
                    #             big_jump.append(new_chunk[k, :, 0]) # each chunk, all the angle data
                    new_chunks = np.vstack((new_chunks, new_chunk))"""

        return new_chunks, num_chunks, rand_chunks



    def __getitem__(self, idx):
        signal, label = PumpDataset.__getitem__(self, idx)
        signal, n_chunks, rand = self.get_chunks(signal, self.chunk_length, self.rand_chunk_rate)
        label = np.repeat(label, n_chunks)

        # TODO: add time differencing, concat tensors

        signal = torch.Tensor(signal).float()
        rand = torch.Tensor(rand).float()

        pdb.set_trace()

        return signal, rand, label


################ To here #############################################################
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

            loss = torch.nn.MSELoss()(y, targets)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            print(loss)
        break
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./trn_cfg.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default="../models/cfg_seq_transformer.yaml", help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cfg, args.model_cfg)
