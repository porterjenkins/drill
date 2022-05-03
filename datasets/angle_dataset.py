import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

################################################################################
################# Create Angle Dataset class ###################################
################################################################################

class AngleDataset(Dataset):
    def __init__(self, data_in, indices, chunk_len):
        super(AngleDataset, self).__init__()

        self.data_out, self.label_out = self.process_data(data_in, indices, chunk_len)

    def __getitem__(self, i):
        x = self.data_out[i]
        y = self.label_out[i]
        #     return x, y.float() # Use for MSELoss
        return x, y

    def __len__(self):
        return len(self.label_out)

    def process_data(self, data, index, chunk_length):
        # chunk_length is a class input; Arbitrary, but there are between about 15-25 samples per second
        big_jump = []

        # data augmentation options: Start with 1 horiz_sampling and then increase from there if needed
        horiz_samplings = 1  # number of times to sample horizontally from the same dataset
        # vert_samplings = 0 # number of vertical samplings; No longer needed due to normalization
        # vert_shift = 3 # degrees of shift when augmenting the data; No longer needed due to normalization

        data_out_m = np.zeros((1, chunk_length, 3), dtype=float)  # Initialize master data out (x)
        labels_m = np.zeros((1, 1), dtype=int)  # Initialize labels out (y)

        # Create a for loop that gets the data for one user then does all the magic for it:
        for i in index:
            #   print('User:',i)
            single_user = data[np.where(data[:, 0] == i)]  # get the data for one user

            len_user, cc = single_user.shape
            #   print(len_user)
            if len_user <= chunk_length:  # skip users whose total lengths are less than the chunk length
                continue  # go to the next iteration

            # SKIP FOR NOW: Clean up data by removing leading and ending portions with no change
            #       single_user = single_user # Will assign the cleaned up data here

            # Calculate total pump time of user for use later as second channel
            duration_seconds = single_user[-1, 1] - single_user[0, 1]
            duration_ind, skip = single_user.shape
            #   print(duration)
            #   print(duration_ind)

            num_chunks = int(
                duration_ind / chunk_length)  # Starting from beginning of array, get chunks of size chunk_length
            #   print(num_chunks)

            # Create and combine chunks
            new_chunks = np.zeros([1, chunk_length, 2], dtype=float)  # initialize array for concatenation in the loop
            for j in range(num_chunks):
                new_chunk = single_user[np.newaxis, 0 + j * chunk_length:chunk_length + j * chunk_length, 2:4]
                new_chunk[0, :, 0] = new_chunk[0, :, 0] - np.mean(
                    new_chunk[0, :, 0])  # Normalizing each chunk to have a mean of 0
                new_chunks = np.vstack((new_chunks, new_chunk))
            #     print(new_chunk.shape)
            new_chunks = np.delete(new_chunks, 0, 0)  # delete initialization row
            #   print('new',new_chunk.shape)

            # Also augment by getting random chunks
            random_sets = 6  # changed from 3 to 10 for only Day 1 data; improved acc from 50% to 100%
            for j in range(random_sets * num_chunks):
                start_ind = np.random.randint(0, duration_ind - chunk_length)
                new_chunk = single_user[np.newaxis, start_ind:start_ind + chunk_length, 2:4]
                new_chunk[0, :, 0] = new_chunk[0, :, 0] - np.mean(
                    new_chunk[0, :, 0])  # Normalizing each chunk to have a mean of 0

                # this loop removes any chunks with large (>65 degree) jumps
                for k in range(new_chunk.shape[0]):
                    #           counter_chunks += 1
                    if max(new_chunk[k, :, 0]) - min(new_chunk[k, :, 0]) < 65:
                        #             counter_recorded += 1
                        #             big_jump.append(new_chunk[k, :, 0]) # each chunk, all the angle data
                        new_chunks = np.vstack((new_chunks, new_chunk))

            #       for i in range(min(len(big_jump), 10)):
            #         plt.plot(big_jump[i])
            #         plt.show()

            # Augment by shifting vertically (DO NOT DO THIS; REPLACED WITH NORMALIZING above)

            # Augment by adding drift based on some condition to make sure we don't add drift to segments that already have drift
            #       slope_samplings = 0 # number of slope augmentations
            #       slopes = [-1, 1]

            # Combine data
            current_num_chunks, a, b = new_chunks.shape
            dur_array = np.full((current_num_chunks, chunk_length, 1),
                                duration_seconds)  # creating channel for the duration
            new_chunks = np.concatenate((new_chunks, dur_array), axis=2)  # Concatenate on the duration channel
            data_out_m = np.vstack((data_out_m, new_chunks))  # concatenate on all chunks from that user
            #   print(data_out_m.shape)
            #   Decide which users to compare
            #       for t in range(single_user.shape[0]):
            #         if single_user[t,4] == 3:
            #           single_user[t,4] = 2

            # create labels
            labels_m = np.vstack((labels_m, np.full((current_num_chunks, 1), single_user[0, 4])))

        data_out_m = np.delete(data_out_m, 0, 0)  # delete initialization row
        labels_out_m = np.delete(labels_m, 0, 0)  # delete initialization row
        labels_out_m = labels_out_m - min(labels_out_m)  # reassign M/W/C to [0, 1, 2] so I can use cross entropy loss

        # Normalize data for better gradients
        # Normalizing params: We want to choose these such that after dividing the original values by this, we get a value between -1 and 1 or 0 and 1 (on the 1 scale)
        amp_norm = 30  # 30 was chosen because we want the max value of this to be approx. 1 and the max amplitude is 60 degrees and half of that is 30;
        dt_norm = 80  # The max dt is about 80 ms
        dur_norm = 600  # the max duration is about 600 seconds

        data_out_m[:, :, 0] = data_out_m[:, :, 0] / amp_norm
        data_out_m[:, :, 1] = data_out_m[:, :, 1] / dt_norm
        data_out_m[:, :, 2] = data_out_m[:, :, 2] / dur_norm
        #     print('Data size = ',data_out_m.shape)
        #     print('Classification size = ',labels_out_m.shape)
        #     print('Mean of all amplitudes (should be ~0 if normalization worked) = ', np.mean(data_out_m[:,:,0]))

        # convert final data from numpy arrays to tensors
        data_out_m = torch.Tensor(data_out_m)
        labels_out_m = torch.Tensor(labels_out_m).long()
        #     print("data type = ", data_out_m.type())
        #     print("label type = ", labels_out_m.type())
        data_out_m = data_out_m.permute(0, 2, 1)
        #     print("permuted = ", data_out_m[:,0:2,:].size())

        # Output: x, y tuple; x = Data arrays of chunks (i.e. 600 users *~10 chunks/user) x 100(chunk len)x3 (angle, dt, duration) arrays; y = labels = Separate array/list of 600 (users) x 1 (User Type)
        return data_out_m[:, 0:2, :], labels_out_m  # x, y ; x = 5000 x 100 x 3 ; y = 5000 x 1
