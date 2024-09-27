# Code repository for the paper "Backpropagation through space, time and the brain"
This repository contains the code to reproduce the results and figures of the preprint "Backpropagation through space, time and the brain" which can be found on [arXiv](https://arxiv.org/abs/2403.16933).

The code is organized as follows:
- GLE layers, dynamics and abstract_net are implemented in the corresponding files in `lib/gle/` directory
- The `data` directory contains code to generate small datasets (e.g., XOR, Yin-Yang) and helper functions to modify the MNIST-1D dataset
- The `experiments` directory contains the code for different experiments, which are further divided into subdirectories:
  - `mimic` contains code for the teacher-student experiments on different regression tasks 
  - `mnist1d` contains code for the [MNIST-1D](https://github.com/greydanus/mnist1d) experiments using different architectures
  - `gsc` contains code for the [Speech Commands](https://arxiv.org/abs/1804.03209) experiments using different architectures
- The `results` folder contains the tracked metrics of the experiments as `*.pkl` files
- The `utils` directory contains helper functions for training and evaluation
See below for instructions on how to reproduce the experiments and figures.

## Setting up the environment
The code was tested with Python 3.11.9 and 3.12.5 but should work with other versions as well.
Packages are pinned to specific versions (see `requirements.txt`) to ensure reproducibility and can be installed using the following command (ideally in a virtual environment):
```bash
pip install -r requirements.txt
```
The code works with both CPU and GPU and should automatically detect the available hardware.
## Reproducing the experiments
All experiments should be run from the root directory of the repository (see below), will produce command-line output and save results in the `results` directory

### Teacher-student experiments
For the LagLine teacher-student experiments in Figure 5, run the following command:
```bash
python -m experiments.mimic.lagline  --model='gle'
```
Available models are
- instantaneous BP: `bp`
- BPTT for different truncation windows:`bptt_tw4`, `bptt_tw2` and `bptt_tw1`
- and GLE: `gle`

A single run of the experiment will take a few minutes and results will be saved in the `results/lagline` directory.

For the LagNet experiment in Figure 6, run the following command:
```bash
python -m experiments.mimic.lagnet
```
A run of the experiment will take a few minutes and results will be saved in the `results/lagnet` directory.

### MNIST-1D and GSC experiments
For the MNIST-1D experiments in Figure 8, there is are separate scripts for the different architectures:
The command for the end-to-end trained GLE network is:
```bash
python -m experiments.mnist1d.plastic_e2e --seed=0
```
Results will be saved in the `results/mnist1d` directory.
Use the `--seed` argument to set the random seed for reproducibility (although results might still vary depending on the hardware used for training).
To obtain the results in the paper we used seeds 12, 34, 56, 78, 98, 76, 54, 32, 42 and 69.
Note that running the script with the same seed will overwrite the results in the `results/mnist1d` directory.
A single run of the experiment will take around 2 hours.

Similarly, the command for static LagNet is with an (G)LE-MLP on top is:
```bash
python -m experiments.mnist1d.static_lagnet --seed=0
```
Results will be saved in the `results/mnist1d` directory.
To obtain the results in the paper we used seeds 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9.
A single run of the experiment will take around 1.5 hours.

For the GSC experiments in Figure 8, run the following command:
```bash
python -m experiments.gsc.gsc_mel_gle --max-epochs=420 --seed=0
```
Results will be saved in the `results/gsc` directory.
To obtain the results in the paper we used seeds 1, 2, 3, 5, 7, 8, 12, 113, 114, 115.
Note that a single run of this experiment will take around 240h on a single GPU.

## Reproducing the figures
To recreate the figures in the paper, go into the ```plotting``` folder and run the following commands:
```bash
gridspeccer --mplrc matplotlibrc --loglevel WARNING fig*.py
```
This will create the figures in the `fig` directory.
This command might need to be adapted for shell environments other than bash and zsh.
