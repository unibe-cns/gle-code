#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd
from matplotlib import gridspec as gs

from gridspeccer import core
from gridspeccer.core import log
from gridspeccer import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """

    gs_main = gs.GridSpec(2, 1,
                          left=0.08, right=0.97, top=0.95, bottom=0.08,
                          height_ratios=[0.9, 1],
                          hspace=0.15)

    gs_outputs = gs.GridSpecFromSubplotSpec(7, 4, gs_main[0, 0],
                                            wspace=0.15, hspace=0.1,
                                            width_ratios=[1, 1, 1, 1],
                                            height_ratios=[0.2, 1, 1, 1, 0.2, 1, 1])

    gs_params_loss = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.3)

    gs_params = gs.GridSpecFromSubplotSpec(2, 1, gs_params_loss[0, 0], hspace=0.3)

    return {
        ### schematics
        "r_2_before": gs_outputs[1, 2],
        "r_1_before": gs_outputs[2, 2],
        "r_0_before": gs_outputs[3, 2],
        "e_1_before": gs_outputs[5, 2],
        "e_0_before": gs_outputs[6, 2],
        "r_2_correct_tau": gs_outputs[1, 1],
        "r_1_correct_tau": gs_outputs[2, 1],
        "r_0_correct_tau": gs_outputs[3, 1],
        "e_1_correct_tau": gs_outputs[5, 1],
        "e_0_correct_tau": gs_outputs[6, 1],
        "r_2_after": gs_outputs[1, 3],
        "r_1_after": gs_outputs[2, 3],
        "r_0_after": gs_outputs[3, 3],
        "e_1_after": gs_outputs[5, 3],
        "e_0_after": gs_outputs[6, 3],
        "chain_model": gs_outputs[:, 0],
        "weights": gs_params[0, 0],
        "tau_m": gs_params[1, 0],
        "loss": gs_params_loss[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
            "chain_model",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "chain_model",
                         "r_2_correct_tau",
                         "weights",
                     ],
                     label_ypos={
                         "chain_model": 1.0,
                         "r_2_correct_tau": 1.25,
                         "weights": 1.05,
                     },
                     label_xpos={
                         "chain_model": -0.2,
                         "r_2_correct_tau": -0.3,
                         "weights": -0.1,
                     },
                     )


def get_fig_kwargs():
    width = 6.4
    alpha = 1.2
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

# with correct BPTT implementation
ids = ["bp", "gle", "bptt_tw4", "bptt_tw2", "bptt_tw1"]  # PE, 0, 400, 200, 100 with Adam and log_interval = 10
data = [pd.read_pickle(f"../results/lagline/{i}_metrics.pkl") for i in ids]
labels = ["BP(TT)", "GLE", "BPTT (TW=4)", "BPTT (TW=2)", "BPTT (TW=1)"]
linestyle = ["--", "-", ",", ",", ","]
param_ls = ('--', '-')

# windows to plot
correct_tau = (120, 180)
before = (320, 380)
after = (29720, 29780)

lw = 1.0

def plot_IE_vs_PE(ax, data, name, start, end, target=False, yLabel=None):
    # rate colors
    # error colors
    if 'e' in name:
        colors = ["deepskyblue", "cornflowerblue", None]
    else:
        colors = ["deeppink", "red", None]
    for d, l, ls, c in zip(data, labels, linestyle, colors):
        if "BPTT" in l:
            continue
        ax.plot(d["time"].values[start:end], d[name].values[start:end], label=f"({l})", color=c, alpha=0.8, lw=lw, ls=ls)
    if target:
        ax.plot(d["time"].values[start:end], d["target"].values[start:end], label=f"target ({l})", linestyle='--', color='black', linewidth=0.5)
    if yLabel:
        ax.set_ylabel(yLabel, rotation=0, labelpad=10, fontsize=14, y=0.4)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_r_2_correct_tau(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_1", *correct_tau, target=True, yLabel=r"$r_2$")
    ax.set_title("correct $\\tau^\\mathrm{m}_i$, wrong $w_i$", pad=20)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["BP", "GLE", "target"], frameon=False, ncol=3, fontsize=9,
              loc=(0.0, 0.95))

def plot_r_1_correct_tau(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_0", *correct_tau, target=False, yLabel=r"$r_1$")

def plot_r_0_correct_tau(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "input", *correct_tau, target=False, yLabel=r"$r_0$")
    ax.annotate("", (12.9, 0), (17, 0), arrowprops={'arrowstyle':'<->', 'shrinkA': 0, 'shrinkB': 0})
    ax.annotate(r"$4\,\mathrm{a.u.}$", (15.1, 0.1), (15.1, 0.1))

def plot_e_1_correct_tau(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_1", *correct_tau, target=False, yLabel=r"$e_2$")

    # create custom legend with GLE and BP
    ax.legend(frameon=False, ncol=3, fontsize=8, loc='best')
    handles, labels = ax.get_legend_handles_labels()
    # set legend location
    ax.legend(handles=handles[:2], labels=["BP", "GLE"], frameon=False, ncol=3, fontsize=9,
              loc=(0.0, 0.9))

def plot_e_0_correct_tau(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_0", *correct_tau, target=False, yLabel=r"$e_1$")
    n_ticks = int((correct_tau[1]-correct_tau[0])/10+1)
    ax.set_xticks(np.linspace(correct_tau[0]/10, correct_tau[1]/10, n_ticks))
    ax.set_xticklabels(n_ticks * [""])
    ax.set_xlabel("time [a.u.]")

def plot_r_2_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_1", *before, target=True)
    ax.set_title("before learning", pad=20)

def plot_r_1_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_0", *before, target=False)

def plot_r_0_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "input", *before, target=False)

def plot_e_1_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_1", *before, target=False)

def plot_e_0_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_0", *before, target=False)
    n_ticks = int((before[1]-before[0])/10+1)
    ax.set_xticks(np.linspace(before[0]/10, before[1]/10, n_ticks))
    ax.set_xticklabels(n_ticks * [""])
    ax.set_xlabel("time [a.u.]")

def plot_r_2_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_1", *after, target=True)
    ax.set_title("after learning", pad=20)
    ax.set_ylim(1.4, 1.9)

def plot_r_1_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "r_0", *after, target=False)

def plot_r_0_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "input", *after, target=False)

def plot_e_1_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_1", *after, target=False)

def plot_e_0_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_IE_vs_PE(ax, data, "prosp_e_0", *after, target=False)
    n_ticks = int((after[1]-after[0])/10+1)
    ax.set_xticks(np.linspace(after[0]/10, after[1]/10, n_ticks))
    ax.set_xticklabels(n_ticks * [""])
    ax.set_xlabel("time [a.u.]")

def plot_weights(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    start = before[0]
    end = after[1]

    # [IE, GLE, BPTT]
    param_colors = ["C4", "C2", "C1", "C5", "C6"]
    windows = [1, 1, 40, 20, 10]

    step_width = int(2 / 0.01)
    for idx, (d, l, c, w) in enumerate(zip(data, labels, param_colors, windows)):
        ax.plot(d["time"].values[start:end:w], d["w_0"].values[start:end:w], linestyle[idx], color=c)
        ax.plot(d["time"].values[start:end:w], d["w_1"].values[start:end:w], linestyle[idx], color=c)

    # plot horizontal line at 0
    ax.axhline(1, color='black', linewidth=0.5, linestyle=param_ls[0], label=r"target $w_0$")
    ax.axhline(2, color='black', linewidth=0.5, linestyle=param_ls[1], label=r"target $w_1$")
    ax.legend(loc='upper left', frameon=False, ncol=2, fontsize=10)
    ax.set_ylim(-0.3, 2.7)
    ax.set_xscale('log')
    # ax.set_xlim(40, 2e3)
    ax.set_ylabel("weights [a.u.]")
    ax.set_xlabel("time [a.u.]")

def plot_tau_m(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    start = before[0]
    end = after[1]

    param_colors = ["C4", "C2", "C1", "C5", "C6"]
    windows = [1, 1, 40, 20, 10]

    step_width = int(2 / 0.01)
    for idx, (d, l, c, w) in enumerate(zip(data, labels, param_colors, windows)):
        ax.plot(d["time"].values[start:end:w], d["tau_m_0"].values[start:end:w], linestyle[idx], color=c)
        ax.plot(d["time"].values[start:end:w], d["tau_m_1"].values[start:end:w], linestyle[idx], color=c)

    ax.axhline(1, color='black', linewidth=0.5, linestyle=param_ls[0], label=r"target $\tau_0$")
    ax.axhline(2, color='black', linewidth=0.5, linestyle=param_ls[1], label=r"target $\tau_1$")
    ax.legend(loc='upper left', frameon=False, ncol=2, fontsize=10)
    ax.set_ylim(0.0, 2.6)
    ax.set_xscale('log')
    # ax.set_xlim(40, 2e3)
    ax.set_ylabel(r"membrane $\tau$ [a.u.]")
    ax.set_xlabel("time [a.u.]")

def plot_loss(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    start = before[0]
    end = after[1]

    param_colors = ["C4", "C2", "C1", "C5", "C6"]
    # windows = [1, 1, 40, 20, 10]  # actual window length
    windows = [40, 40, 40, 40, 40]  # average all over 40 steps
    linestyles = ['--', '-', ',', ',', ',']

    for d, l, c, w , ls in zip(data, labels, param_colors, windows, linestyles):
        # plot average over window steps
        avg_loss = d["loss"].values[start:end]
        avg_loss = [np.mean(avg_loss[i:i+w]) for i in range(0, len(avg_loss), w)]
        ax.plot(d["time"].values[start:end:w], avg_loss, ls, label=l, color=c)

    ax.set_xscale('log')
    ax.set_yscale("log")
    ax.set_xlabel("time [a.u.]")
    ax.set_ylabel(r"mean loss (over $4\,\mathrm{a.u.}$)")
    # ax.set_ylim(1e-10, 1e0)
    ax.legend(frameon=False, loc='lower left', ncol=1, fontsize=10, numpoints=50)
