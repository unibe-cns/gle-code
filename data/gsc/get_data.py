import os
import math
import random
import numpy as np
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data.sampler import WeightedRandomSampler
from data.gsc.tf1_gsc_splitter import prepare_data_index


EPS = 1e-9
SAMPLE_RATE = 16000

# default labels from GSC dataset
ALL_LABELS = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',

    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',

    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',

    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',

    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',

    'up',
    'visual',
    'wow',
    'yes',
    'zero',
    'silence',
]

# ALL_LABELS = [
#     'eight',
#     'five',
#     'four',

#     'nine',

#     'one',
#     'seven',

#     'six',
#     'three',
#     'two',

#     'zero',
# ]

KW12_LABELS = [
    'unknown',
    'silence',
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
]

def _get_speechcommands_metadata(filepath: str, path: str):
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    return relpath, SAMPLE_RATE, label

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str, path="./data/gsc", classes=ALL_LABELS, n_mels=40, interpolate=1, silence_percentage=0.1, fft_type='mel', fft_window=1024, fft_stride=512, tf1_splits=False, random_seed=0):
        super().__init__(path, download=True)
        if fft_type == 'mel':
            self.to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=fft_window, n_mels=n_mels, hop_length=int(fft_stride))
            self.to_log_fft = lambda x: (self.to_mel(x) + EPS).log2()
        else:
            self.to_log_fft = transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=n_mels, log_mels=True,  melkwargs={"n_fft": fft_window, "hop_length": int(fft_stride), "n_mels": n_mels})

        self.subset = subset
        self.classes = classes
        self.interpolate = interpolate

        self._label_to_idx = {label: i for i, label in enumerate(self.classes)}
        if 'unknown' in self.classes:
            for c in ALL_LABELS: # for all labels in the dataset
                if c not in self._label_to_idx:
                    self._label_to_idx[c] = self._label_to_idx['unknown'] # set all labels not in the selected subset as unknown

        # reverse mapping
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fh:
                return [
                    os.path.join(self._path, line.strip()) for line in fh
                ]

        self._noise = []

        # choose training/validation/testing subset
        if not tf1_splits:
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

                # prepare noise waveforms
                noise_paths = [w for w in os.listdir(os.path.join(self._path, "_background_noise_")) if w.endswith(".wav")]
                for item in noise_paths:
                    noise_path =  os.path.join(self._path, "_background_noise_", item)
                    noise_waveform, noise_sr = torchaudio.sox_effects.apply_effects_file(noise_path, effects=[])
                    noise_waveform = transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
                    self._noise.append(noise_waveform)
            else:
                raise ValueError(f"Unknown subset {subset}. Use validation/testing/training")
        else:
            if subset == "validation":
                # self._walker already contains all files
                self._walker = prepare_data_index(self._walker, unknown_percentage=10, wanted_words=[label for label in KW12_LABELS if label not in ['unknown', 'silence']],validation_percentage=10, testing_percentage=10, random_seed=random_seed)['validation']
            elif subset == "testing":
                # self._walker already contains all files
                self._walker = prepare_data_index(self._walker, unknown_percentage=10, wanted_words=[label for label in KW12_LABELS if label not in ['unknown', 'silence']],validation_percentage=10, testing_percentage=10, random_seed=random_seed)['testing']
            elif subset == "training":
                # self._walker already contains all files
                self._walker = prepare_data_index(self._walker, unknown_percentage=10, wanted_words=[label for label in KW12_LABELS if label not in ['unknown', 'silence']], validation_percentage=10, testing_percentage=10, random_seed=random_seed)['training']

                # prepare noise waveforms
                noise_paths = [w for w in os.listdir(os.path.join(self._path, "_background_noise_")) if w.endswith(".wav")]
                for item in noise_paths:
                    noise_path =  os.path.join(self._path, "_background_noise_", item)
                    noise_waveform, noise_sr = torchaudio.sox_effects.apply_effects_file(noise_path, effects=[])
                    noise_waveform = transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
                    self._noise.append(noise_waveform)
            else:
                raise ValueError(f"Unknown subset {subset}. Use validation/testing/training")
        if 'unknown' not in classes:
            # drop not selected classes if not mapped to unknown
            self._walker = [w for i,w in enumerate(self._walker) if _get_speechcommands_metadata(w, self._archive)[2] in self.classes]

        if 'silence' in classes:
            # add silence
            self._walker += ['silence'] * int(len(self._walker) * silence_percentage)

    def _noise_augment(self, waveform):
        noise_waveform = random.choice(self._noise)

        noise_sample_start = 0
        if noise_waveform.shape[1] - waveform.shape[1] > 0:
            noise_sample_start = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
        noise_waveform = noise_waveform[:, noise_sample_start:noise_sample_start+waveform.shape[1]]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3]
        snr = random.choice(snr_dbs)

        snr = math.exp(snr / 10)
        scale = snr * noise_power / signal_power
        noisy_signal = (scale * waveform + noise_waveform) / 2
        return noisy_signal

    def _shift_augment(self, waveform):
        shift = random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _augment(self, waveform):
        if random.random() < 0.8:
            waveform = self._noise_augment(waveform)
        
        waveform = self._shift_augment(waveform)

        return waveform

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for path in self._walker:
            path_splits = path.split('/')
            class_name = path_splits[-2] if len(path_splits)> 2 else path_splits[-1]
            count[self._label_to_idx[class_name]] += 1

        print("class counts: ", count)

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, path in enumerate(self._walker):
            path_splits = path.split('/')
            class_name = path_splits[-2] if len(path_splits)> 2 else path_splits[-1]
            weight[idx] = weight_per_class[self._label_to_idx[class_name]]

        print("class weights: ", weight_per_class)
        return weight

    def __getitem__(self, n):
        if self._walker[n] == 'silence':
            waveform = torch.ones(1, SAMPLE_RATE) * EPS
            sample_rate = SAMPLE_RATE
            label = 'silence'
        else:
            waveform, sample_rate, label, _, _ = super().__getitem__(n)

        if sample_rate != SAMPLE_RATE: 
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.subset == "training":
            waveform = self._augment(waveform)
        log_mel = self.to_log_fft(waveform)
        # normalize each sample to distribution 0, 1
        log_mel -= log_mel.mean()
        log_mel /= log_mel.std()

        if self.interpolate != 1:
            log_mel = torch.nn.functional.interpolate(log_mel, scale_factor=self.interpolate, mode='linear')

        return log_mel, label

def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def collate_fn(batch, label_to_idx):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_idx[label])

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets

def get_gsc_dataloaders2(n_mels, params, use_gpu, interpolate=1, fft_type='mel', fft_window=1024, fft_stride=512, randomsample_on_val_test=True, tf1_splits=False, random_seed=0):

    if use_gpu:
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if params['dataset_type'] == "full":
        classes = ALL_LABELS
    else:
        classes = KW12_LABELS

    train_dataset = SubsetSC(subset="training", classes=classes, n_mels=n_mels, interpolate=interpolate, fft_type=fft_type, fft_window=fft_window, fft_stride=fft_stride, tf1_splits=tf1_splits, random_seed=random_seed)
    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    kw_args = {"sampler": sampler}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **kw_args,
        batch_size=params['batch_size'],
        collate_fn=lambda x: collate_fn(x, train_dataset._label_to_idx),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    validation_dataset = SubsetSC(subset="validation", classes=classes, n_mels=n_mels, interpolate=interpolate, fft_type=fft_type, fft_window=fft_window, fft_stride=fft_stride, tf1_splits=tf1_splits, random_seed=random_seed)
    weights = validation_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    kw_args = {"sampler": sampler} if randomsample_on_val_test else {"shuffle": False}
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **kw_args,
        batch_size=params['batch_size'],
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, validation_dataset._label_to_idx),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataset = SubsetSC(subset="testing", classes=classes, n_mels=n_mels, interpolate=interpolate, fft_type=fft_type, fft_window=fft_window, fft_stride=fft_stride, tf1_splits=tf1_splits, random_seed=random_seed)
    weights = test_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    kw_args = {"sampler": sampler} if randomsample_on_val_test else {"shuffle": False}
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **kw_args,
        batch_size=params['batch_size'],
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, test_dataset._label_to_idx),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, validation_loader, test_loader
