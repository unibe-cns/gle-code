#!/usr/bin/env python3
# encoding: utf-8

import matplotlib.ticker as tck
from matplotlib import gridspec as gs
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import glob

from gridspeccer import core
from gridspeccer.core import log
from gridspeccer import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """

    gs_main = gs.GridSpec(2, 1,
                          left=0.1, right=0.97, top=0.95, bottom=0.05,
                          height_ratios=[0.53, 2],
                          hspace=0.1)

    gs_datasets = gs.GridSpecFromSubplotSpec(1, 3, gs_main[0, 0],
                                             width_ratios=[0.8, 1, 1.3],
                                             wspace=0.2)

    gs_gsc_samples = gs.GridSpecFromSubplotSpec(3, 1, gs_datasets[0, 1],
                                                    height_ratios=[1, 1, 1],
                                                    hspace=0.2)

    gs_curves = gs.GridSpecFromSubplotSpec(2, 1, gs_main[1, 0],
                                             height_ratios=[1, 1],
                                             hspace=0.25)

    gs_spatiotemporal = gs.GridSpecFromSubplotSpec(1, 2, gs_curves[0, 0],
                                                   width_ratios=[1, 1],
                                                   wspace=0.25)

    gs_spatial = gs.GridSpecFromSubplotSpec(1, 2, gs_curves[1, 0],
                                            width_ratios=[1, 1],
                                            wspace=0.25)


    return {
        # ### schematics
        "mnist1d_samples": gs_datasets[0, 0],
        "gsc_samples": gs_datasets[0, 1],
        "cifar10_samples": gs_datasets[0, 2],
        "mnist1d_acc": gs_spatiotemporal[0, 0],
        "speechcommands_acc": gs_spatiotemporal[0, 1],
        "mnist1d_le_acc": gs_spatial[0, 0],
        "cifar10_acc": gs_spatial[0, 1],
        "gsc_waveform": gs_gsc_samples[0, 0],
        "gsc_mel": gs_gsc_samples[1, 0],
        "gsc_mfcc": gs_gsc_samples[2, 0],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
            "mnist1d_samples",
            "gsc_samples",
            "gsc_waveform",
            "gsc_mel",
            "gsc_mfcc",
    ]:
        axes[k].set_frame_on(False)



def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "mnist1d_samples",
                         "gsc_samples",
                         "cifar10_samples",
                         "mnist1d_acc",
                         "speechcommands_acc",
                         "mnist1d_le_acc",
                         "cifar10_acc",
                     ],
                     label_ypos={
                         "mnist1d_samples": 0.97,
                         "gsc_samples": 0.97,
                         "cifar10_samples": 0.97,
                         "mnist1d_acc": 0.97,
                         "speechcommands_acc": 0.97,
                         "mnist1d_le_acc": 0.97,
                         "cifar10_acc": 0.97,
                     },
                     label_xpos={
                         "mnist1d_samples": -0.1,
                         "gsc_samples": -0.1,
                         "cifar10_samples": -0.1,
                     },
                     )


def get_fig_kwargs():
    width = 6.4
    alpha = 1.33
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

import pickle

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def plot_mnist1d_samples(ax):
    # plot samples
    from mnist1d.data import get_dataset, get_dataset_args
    args = get_dataset_args(as_dict=False)
    args.final_seq_length = 360
    dataset = get_dataset(args, path='../data/mnist1d_data_360.pkl', regenerate=True, download=False)
    for j in range(10):
        for i in range(1000):
            if j  == dataset['y'][i]:
                ax.plot(9*(9 - j) + dataset['x'][i], label='{}'.format(dataset['y'][i]))
                break
            else:
                continue
    ax.set_xlim(-50, 360)
    leg = ax.legend(fontsize=7.4, frameon=False, loc='upper left')
    for line in leg.get_lines():
        line.set_linestyle('None')
    # move legend to left
    leg.set_bbox_to_anchor((-0.3, 1.0))

# plot average over multiple runs
def plot_mnist1d_runs(ax, paths, label, metric='test_acc', plot_error=False):
    accs = []
    for path in paths:
        df = pd.read_pickle(f"{path}")
        try:
            accs.append(df[metric])
        except KeyError:
            print(f"KeyError: metric {metric} not found in file {path}")
    mean = np.mean(accs, axis=0)
    if plot_error:
        mean = 100 - mean
    std = np.std(accs, axis=0)
    ax.plot(mean, label=label, linewidth=1.0, alpha=0.8)
    ax.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2)

    if metric == 'test_acc':
        print(f"{label}: {mean[-1]:.1f} +- {std[-1]:.1f}")

def plot_mnist1d_42k(ax, metric='test_acc', plot_error=False):
    # 10 seeds with 42k params
    ids = glob.glob("../results/mnist1d/42k_plastic_e2e_*.pkl")
    plot_mnist1d_runs(ax, ids, "GLE (42k)", metric=metric, plot_error=plot_error)
    # change linstyle of 42k runs to dashed
    for line in ax.lines:
        line.set_linestyle('--')

def plot_mnist1d_15k(ax, metric='test_acc', plot_error=False):
    # ten seeds with 15k params
    ids = glob.glob("../results/mnist1d/plastic_e2e_*.pkl")
    plot_mnist1d_runs(ax, ids, "GLE (15k)", metric=metric, plot_error=plot_error)
    # same color for GLE runs
    for line in ax.lines:
        line.set_color('C0')
    for coll in ax.collections:
        coll.set_color('C0')

def plot_mnist1d_baselines(ax, metric='test_acc', plot_error=False):
    # plot original baselines
    # they perform 6000 steps with batch size 100
    # we train 100 epochs with batch size 100
    # and one epoch uses 4000 samples, so 1 epoch corresponds to 40 steps
    # and 6000 steps correspond to 150 epochs
    archs = ["lin", "mlp", "cnn", "gru"]
    labels = ["Linear${}_0$", "MLP${}_0$", "TCN${}_0$", "GRU${}_0$"]
    colors = ['None', 'C2', 'C3', 'C4']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for arch, label, color in zip(archs, labels, colors):
        # valid_acc = pd.read_pickle(f"../results/mnist1d_{arch}.pkl")
        # ax.plot(valid_acc, label=label, linewidth=1.0, alpha=0.8)
        accs = []
        if arch == "lin":
            continue
        for seed in seeds:
            with open(f"../results/mnist1d/baselines/mnist1d_{arch}_{seed}.pkl", "rb") as f:
                try:
                    accs.append(pd.read_pickle(f)[metric])
                except KeyError:
                    print(f"KeyError: metric {metric} not found in file ../results/mnist1d_{arch}_{seed}.pkl")
            # accs.append(pd.read_pickle(f"../results/mnist1d_{arch}_{seed}.pkl")['test_acc'])
        if len(accs) == 0:
            print(f"No data found for {arch}")
            continue
        mean = np.mean(accs, axis=0)
        if plot_error:
            mean = 100 - mean
        std = np.std(accs, axis=0)
        # create x axis from 0 to 150
        x = np.linspace(0, 150, len(mean))
        # ax.plot(x, mean, label=label, linewidth=1.0, alpha=0.8, color=color)
        # ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        # remove error bars of original baselines (indexed 0 in LE panel below)
        # ax.errorbar(152.5, mean[-1], yerr=std[-1], fmt='x', color=color, ms=0, capsize=3, label=label)

        # print last acc with std
        if metric == 'test_acc':
            print(f"{label}: {mean[-1]:.1f} +- {std[-1]:.1f}")

    # plot optimized CNN & GRU
    for arch, label, color in zip(["mlp", "cnn", "gru"], ["MLP", "TCN", "GRU"], ['C2', 'C3', 'C4']):
        accs = []
        for seed in seeds:
            with open(f"../results/mnist1d/baselines/mnist1d_{arch}_super_{seed}.pkl", "rb") as f:
                try:
                    accs.append(pd.read_pickle(f)[metric])
                except KeyError:
                    print(f"KeyError: metric {metric} not found in file ../results/mnist1d_{arch}_best_{seed}.pkl")
            # accs.append(pd.read_pickle(f"../results/mnist1d_{arch}_{seed}.pkl")['test_acc'])
        if len(accs) == 0:
            print(f"No data found for {arch}")
            continue

        mean = np.mean(accs, axis=0)
        if plot_error:
            mean = 100 - mean
        std = np.std(accs, axis=0)
        # create x axis from 0 to 150
        x = np.linspace(0, 150, len(mean))
        ax.plot(x, mean, label=label, linewidth=1.0, alpha=0.8, linestyle='-', color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        # plot cross with error bar for last value
        # ax.errorbar(150, mean[-1], yerr=std[-1], fmt='x', color=color, ms=0, capsize=3, label=label)

        # print last acc with std
        if metric == 'test_acc':
            print(f"{label}: {mean[-1]:.1f} +- {std[-1]:.1f}")

    ax.set_xlabel("epoch")
    if metric == 'test_acc':
        if plot_error:
            ax.set_ylabel("validation error")
        else:
            ax.set_ylabel(r"validation accuracy $[\%]$")
    elif metric == 'test_loss':
        ax.set_ylabel("validation loss")
    else:
        ax.set_ylabel(metric)
    ax.set_xlim(0, 155)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=1, title="MNIST-1D", title_fontsize=10, frameon=False)

def plot_mnist1d_acc(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)
    plot_error = False
    metric = 'test_acc'
    # plot 42k GLE
    plot_mnist1d_42k(ax, metric=metric, plot_error=plot_error)
    # plot 15k GLE
    plot_mnist1d_15k(ax, metric=metric, plot_error=plot_error)
    # plot baselines
    plot_mnist1d_baselines(ax, metric=metric, plot_error=plot_error)
    if plot_error:
        ax.set_ylim(1, 100)
        ax.set_yscale('log')
    ax.set_ylabel(r"validation accuracy $[\%]$")

def extract_data_from_logs(log_text):
    import re

    # pattern1 = r"Step\s+#(\d+):\s+rate\s+([\d.]+),\s+accuracy\s+([\d.]+)%,\s+cross\s+entropy\s+([\d.]+)" # train acc
    pattern2 = r"Step\s+(\d+):\s+Validation\s+accuracy\s+=\s+([\d.]+)%\s+\(N=(\d+)\)" # validation acc
    pattern3 = r"Final\s+test\s+accuracy\s+=\s+([\d.]+)%\s+\(N=(\d+)\)" # test acc

    # matches1 = re.findall(pattern1, log_text)
    matches2 = re.findall(pattern2, log_text)
    matches3 = re.findall(pattern3, log_text)

    # for match in matches1:
    #     step, rate, accuracy, cross_entropy = match
    #     print(f"Step: {step}, Rate: {rate}, Accuracy: {accuracy}%, Cross Entropy: {cross_entropy}")

    validation_accs = []
    for match in matches2:
        step, validation_accuracy, n_value = match
        validation_accs.append(float(validation_accuracy))
        #print(f"Step: {step}, Validation Accuracy: {validation_accuracy}%, N: {n_value}")

    test_accuracy = "0"
    for match in matches3:
        test_accuracy, n_value = match
        #print(f"Final Test Accuracy: {test_accuracy}%, N: {n_value}")

    return validation_accs, [float(test_accuracy)]

import pandas as pd
def plot_speechcommands_acc(ax, TDNN=False):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    # GLE
    # open 10 seeds
    ids = glob.glob("../results/gsc/GLE_*_val_acc.csv")
    max_epochs = 420
    accs = []
    for path in ids:
        df = pd.read_csv(f"{path}")
        accs.append(df.iloc[:max_epochs, 2])

    mean = np.mean(accs, axis=0)
    std = np.std(accs, axis=0)
    # create x axis from 0 to 150
    gle_epochs = len(mean)
    x = np.linspace(0, gle_epochs, len(mean))
    ax.plot(x, mean, label='GLE', linewidth=1.0, alpha=0.8, linestyle='--', color='C0')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='C0')

    print(f"GSC: GLE (V): {mean[-1]:.2f} +- {std[-1]:.2f}")

    # MLP L
    # load log txt to string
    log1 = open('../results/gsc_baselines/mlp_s1_debug_err_long.txt', 'r').read()
    log2 = open('../results/gsc_baselines/mlp_s2_debug_err_long.txt', 'r').read()
    log3 = open('../results/gsc_baselines/mlp_s3_debug_err_long.txt', 'r').read()
    log4 = open('../results/gsc_baselines/mlp_s4_debug_err_long.txt', 'r').read()
    log5 = open('../results/gsc_baselines/mlp_s5_debug_err_long.txt', 'r').read()
    log6 = open('../results/gsc_baselines/mlp_s6_debug_err_long.txt', 'r').read()
    log7 = open('../results/gsc_baselines/mlp_s7_debug_err_long.txt', 'r').read()
    log8 = open('../results/gsc_baselines/mlp_s8_debug_err_long.txt', 'r').read()
    log9 = open('../results/gsc_baselines/mlp_s9_debug_err_long.txt', 'r').read()
    log10 = open('../results/gsc_baselines/mlp_s10_debug_err_long.txt', 'r').read()

    # extract val and test acc
    val_accs = []
    test_accs = []
    for log in [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]:
        val_acc, test_acc = extract_data_from_logs(log)
        
        val_accs.append([8.3] + val_acc)
        test_accs.append(test_acc)

    mean = np.mean(val_accs, axis=0)
    std = np.std(val_accs, axis=0)
    # create x axis from 0 to 150
    # validate every x epoch = number of training samples / 400 batches * 100 samples
    validate_every_x_epoch = 400 * 100 / 36923
    x = np.linspace(0, validate_every_x_epoch * len(mean), len(mean))
    ax.plot(x, mean, label='MLP L', linewidth=1.0, alpha=0.8, linestyle='--', color='C2')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='C2')
    print(f"GSC: MLP L (V): {mean[-1]:.2f} +- {std[-1]:.2f}")
    print(f"GSC: MLP L (T): {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}")
    # ax.scatter(max_epochs + 2, 87.0, marker='x', label='MLP L', color='C2', s=20, alpha=0.8)
    
    # TCN L
    # load log txt to string
    log1 = open('../results/gsc_baselines/cnn_s1_debug_err_long.txt', 'r').read()
    log2 = open('../results/gsc_baselines/cnn_s2_debug_err_long.txt', 'r').read()
    log3 = open('../results/gsc_baselines/cnn_s3_debug_err_long.txt', 'r').read()
    log4 = open('../results/gsc_baselines/cnn_s4_debug_err_long.txt', 'r').read()
    log5 = open('../results/gsc_baselines/cnn_s5_debug_err_long.txt', 'r').read()
    log6 = open('../results/gsc_baselines/cnn_s6_debug_err_long.txt', 'r').read()
    log7 = open('../results/gsc_baselines/cnn_s7_debug_err_long.txt', 'r').read()
    log8 = open('../results/gsc_baselines/cnn_s8_debug_err_long.txt', 'r').read()
    log9 = open('../results/gsc_baselines/cnn_s9_debug_err_long.txt', 'r').read()
    log10 = open('../results/gsc_baselines/cnn_s10_debug_err_long.txt', 'r').read()

    # extract val and test acc
    val_accs = []
    test_accs = []
    for log in [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]:
        val_acc, test_acc = extract_data_from_logs(log)
        
        val_accs.append([8.3] + val_acc)
        test_accs.append(test_acc)

    mean = np.mean(val_accs, axis=0)
    std = np.std(val_accs, axis=0)
    # create x axis from 0 to 150
    # validate every x epoch = number of training samples / 400 batches * 100 samples
    validate_every_x_epoch = 400 * 100 / 36923
    x = np.linspace(0, validate_every_x_epoch * len(mean), len(mean))
    ax.plot(x, mean, label='TCN L', linewidth=1.0, alpha=0.8, linestyle='--', color='C3')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='C3')
    print(f"GSC: TCN L (V): {mean[-1]:.2f} +- {std[-1]:.2f}")
    print(f"GSC: TCN L (T): {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}")
    # ax.scatter(max_epochs + 2, 92.0, marker='x', label='TCN L', color='C3', s=20, alpha=0.8)

    # GRU L
    # load log txt to string
    log1 = open('../results/gsc_baselines/gru_s1_debug_err_long.txt', 'r').read()
    log2 = open('../results/gsc_baselines/gru_s2_debug_err_long.txt', 'r').read()
    log3 = open('../results/gsc_baselines/gru_s3_debug_err_long.txt', 'r').read()
    log4 = open('../results/gsc_baselines/gru_s4_debug_err_long.txt', 'r').read()
    log5 = open('../results/gsc_baselines/gru_s5_debug_err_long.txt', 'r').read()
    log6 = open('../results/gsc_baselines/gru_s6_debug_err_long.txt', 'r').read()
    log7 = open('../results/gsc_baselines/gru_s7_debug_err_long.txt', 'r').read()
    log8 = open('../results/gsc_baselines/gru_s8_debug_err_long.txt', 'r').read()
    log9 = open('../results/gsc_baselines/gru_s9_debug_err_long.txt', 'r').read()
    log10 = open('../results/gsc_baselines/gru_s10_debug_err_long.txt', 'r').read()

    # extract val and test acc
    val_accs = []
    test_accs = []
    for log in [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]:
        val_acc, test_acc = extract_data_from_logs(log)
        
        val_accs.append([8.3] + val_acc)
        test_accs.append(test_acc)

    mean = np.mean(val_accs, axis=0)
    std = np.std(val_accs, axis=0)
    # create x axis from 0 to 150
    # validate every x epoch = number of training samples / 400 batches * 100 samples
    validate_every_x_epoch = 400 * 100 / 36923
    x = np.linspace(0, validate_every_x_epoch * len(mean), len(mean))
    ax.plot(x, mean, label='GRU L', linewidth=1.0, alpha=0.8, linestyle='--', color='C4')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='C4')
    print(f"GSC GRU L (V): {mean[-1]:.2f} +- {std[-1]:.2f}")
    print(f"GSC: GRU L (T): {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}")
    # ax.scatter(max_epochs + 2, 93.5, marker='x', label='GRU L', color='C4', s=20, alpha=0.8)

    # LSTM L
    # load log txt to string
    log1 = open('../results/gsc_baselines/lstm_s1_debug_err_long.txt', 'r').read()
    log2 = open('../results/gsc_baselines/lstm_s2_debug_err_long.txt', 'r').read()
    log3 = open('../results/gsc_baselines/lstm_s3_debug_err_long.txt', 'r').read()
    log4 = open('../results/gsc_baselines/lstm_s4_debug_err_long.txt', 'r').read()
    log5 = open('../results/gsc_baselines/lstm_s5_debug_err_long.txt', 'r').read()
    log6 = open('../results/gsc_baselines/lstm_s6_debug_err_long.txt', 'r').read()
    log7 = open('../results/gsc_baselines/lstm_s7_debug_err_long.txt', 'r').read()
    log8 = open('../results/gsc_baselines/lstm_s8_debug_err_long.txt', 'r').read()
    log9 = open('../results/gsc_baselines/lstm_s9_debug_err_long.txt', 'r').read()
    log10 = open('../results/gsc_baselines/lstm_s10_debug_err_long.txt', 'r').read()

    # extract val and test acc
    val_accs = []
    test_accs = []
    for log in [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]:
        val_acc, test_acc = extract_data_from_logs(log)
        
        val_accs.append([8.3] + val_acc)
        test_accs.append(test_acc)

    mean = np.mean(val_accs, axis=0)
    std = np.std(val_accs, axis=0)
    # create x axis from 0 to 150
    # validate every x epoch = number of training samples / 400 batches * 100 samples
    validate_every_x_epoch = 400 * 100 / 36923
    x = np.linspace(0, validate_every_x_epoch * len(mean), len(mean))
    ax.plot(x, mean, label='LSTM L', linewidth=1.0, alpha=0.8, linestyle='--', color='C8')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='C8')
    print(f"GSC: LSTM L (V): {mean[-1]:.2f} +- {std[-1]:.2f}")
    print(f"GSC: LSTM L (T): {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}")
    # ax.scatter(max_epochs + 2, 92.6, marker='x', label='LSTM L', color='C8', s=20, alpha=0.8)

    ax.set_xlim(0, max_epochs - 100)
    ax.legend(fontsize=8, title="SpeechCommands", title_fontsize=10, frameon=False, loc='lower right')
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 100)
    # ax.set_xscale("log")
    ax.set_ylabel(r"validation accuracy $[\%]$")

def plot_cifar10_acc(ax):
    pass

def plot_le_mnist1d(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)
    ids = glob.glob("../results/mnist1d/static_lagnet_*_metrics.pkl")
    plot_mnist1d_runs(ax, ids, '(G)LE', metric='test_acc', plot_error=False)

    # plot baselines
    plot_mnist1d_baselines(ax, metric='test_acc', plot_error=False)

    ax.set_xlabel('epoch')
    ax.set_ylabel(r'valid accuracy [\%]')
    ax.legend(fontsize=8, loc='lower right', title='MNIST-1D', title_fontsize=10, frameon=False)

def plot_mnist1d_le_acc(ax):
    plot_le_mnist1d(ax)
    ax.set_xlim(0, 155)
    ax.set_ylim(20, 100)
    # ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation accuracy $[\%]$")

def plot_waveform(ax, waveform, sr, title="Waveform"):
    waveform = waveform

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sr

    ax.plot(time_axis, waveform[0], linewidth=0.1)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(ax, specgram, title=None, xlabel="time", ylabel="freq_bin"):
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")

def plot_gsc_waveform(ax):
    speech_waveform = np.load('../results/happy_waveform.npy')
    plot_waveform(ax, speech_waveform, 16000, title=f"")

def plot_gsc_mel(ax):
    melspec = np.load('../results/happy_mel.npy')
    plot_spectrogram(ax, melspec, title="", xlabel="", ylabel="Mel bins")

def plot_gsc_mfcc(ax):
    mfcc = np.load('../results/happy_mfcc.npy')
    plot_spectrogram(ax, mfcc, title="", xlabel="Time/Windows", ylabel="MFCC Coefficients")

