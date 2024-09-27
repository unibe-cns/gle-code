#!/usr/bin/env python3


class SlidingWindow:

    def __init__(self, data, window_size, dilation=1, n_updates=1):
        self.data = data
        self.dilation = dilation
        self.window_size = window_size
        self.window_idx = 0

        # upsample data
        if n_updates > 1:
            # repeat data (piecewise constant interpolation)
            self.data = self.data.repeat_interleave(n_updates, dim=-1)

    def __len__(self):
        return self.data.shape[-1] - self.dilation * (self.window_size - 1)

    def __getitem__(self, idx):
        if self.window_idx >= len(self):
            raise StopIteration

        window = self.data[..., self.window_idx:self.window_idx + self.dilation * self.window_size:self.dilation]

        self.window_idx += 1
        return window


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
