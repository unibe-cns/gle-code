#!/usr/bin/env python3


class Slicer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.input_idx = 0

    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, idx):
        if self.input_idx >= len(self):
            raise StopIteration

        slice = self.data[..., self.input_idx: self.input_idx + 1]

        self.input_idx += 1
        return slice
