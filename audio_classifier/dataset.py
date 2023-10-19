'''
    PyTorch dataset class for COCO-CT-formatted datasets.

    2022 Benjamin Kellenberger / Adapted by Elijah Cole
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
import torchaudio
import torch
import torchvision

def get_split_idx(num_items, frac, seed=1234):
    '''
        Generates indices for splitting a dataset. 
    '''
    # compute size of each split:
    num_split_1 = int(np.round((1.0 - frac) * num_items)) # number of items in first split
    num_split_2 = num_items - num_split_1 # number of items in second split
    # get indices for each split:
    rng = np.random.default_rng(seed) # Want the split to be the same every time.
    idx_rand = rng.permutation(num_items) # Permutation of 0, 1, ..., num_items-1
    idx_split_1 = idx_rand[:num_split_1]
    idx_split_2 = idx_rand[-num_split_2:]
    assert len(idx_split_1) + len(idx_split_2) == num_items # check that the split sizes add up
    return idx_split_1, idx_split_2

def min_max_normalize(x):
    x = x - torch.min(x)
    x = x / torch.max(x)
    return x

class AudioDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        
        # define transforms:
        resamp = torchaudio.transforms.Resample(
            orig_freq=22050,
            new_freq=16000
            )
        to_spec = torchaudio.transforms.MelSpectrogram(
            n_fft=512,
            hop_length=128
            )
        time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=60, # mask up to 60 consecutive time windows
        )
        freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=8, # mask up to 8 consecutive frequency bins
        )
        if split == 'train':
            self.transform = Compose([                              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                resamp,                                             # resample to 16 kHz
                to_spec,                                            # convert to a spectrogram
                torchaudio.transforms.AmplitudeToDB(),
                torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
                time_mask,                                          # randomly mask out a chunk of time
                freq_mask,                                          # randomly mask out a chunk of frequencies
                Resize(cfg['image_size']),
            ])
        else:
            self.transform = Compose([                              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                resamp,                                             # resample to 16 kHz
                to_spec,                                            # convert to a spectrogram
                torchaudio.transforms.AmplitudeToDB(),
                torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
                Resize(cfg['image_size']),
            ])

        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            'audio_ml.csv'
        )
        with open(annoPath, 'r') as f:
            csv_lines = f.readlines()
        csv_lines = csv_lines[1:] # get rid of the header row
        csv_lines = [l.rstrip() for l in csv_lines] # delete newline character (\n) from the end of each line
        
        # get the filenames and labels for the non-test data we're allowed to use for model development:
        dev_filenames = []
        dev_labels = []
        for l in csv_lines:
            # split out the fields in the current row of the csv:
            asset_id, label, split_assignment, samplerate, channels, samples, duration_seconds = l.split(',')
            # Note: From the dataset documentation, we know that all of the audio files have 1 channel, are 10s long, and have a sample rate of 22.05 kHz. But let's check those assumptions.
            assert int(channels) == 1
            assert float(duration_seconds) == 10.0
            assert float(samplerate) == 22.05 * 1000
            if split_assignment == 'train':
                dev_filenames.append(asset_id + '.wav')
                dev_labels.append(int(label))
            else:
                continue
                '''
                The official splits of SSW60 are train and test. We don't want 
                to touch any of the test data. We will make our own val split 
                from the training set below. 
                '''
        dev_filenames = np.array(dev_filenames)
        dev_labels = np.array(dev_labels)
        
        # SSW60 does not have an official val set, so we create one by taking some of the training data:
        val_frac = 0.15
        print('Creating {:.0f}/{:.0f} split (train/val). Choosing {}.'.format(100*(1-val_frac), 100*val_frac, split))
        idx_train, idx_val = get_split_idx(len(dev_filenames), val_frac)
        
        # pick filenames and labels based on the split we're currently working with:
        if split == 'train':
            filenames = dev_filenames[idx_train]
            labels = dev_labels[idx_train]
        elif split == 'val':
            filenames = dev_filenames[idx_val]
            labels = dev_labels[idx_val]
        
        # index data into list
        self.data = []
        for filename, label in zip(filenames, labels):
            self.data.append([filename, label])
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]

        # load image
        audio_path = os.path.join(self.data_root, 'audio_ml', image_name)
        waveform, sample_rate = torchaudio.load(audio_path)
        image = self.transform(waveform)
        image = image.expand(3, -1, -1) # replicate to 3 channels

        return image, label
