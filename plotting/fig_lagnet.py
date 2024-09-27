#!/usr/bin/env python3

import matplotlib.ticker as tck
from matplotlib import gridspec as gs
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import os.path as osp
import pickle

from gridspeccer import core
from gridspeccer.core import log
from gridspeccer import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """

    gs_main = gs.GridSpec(3, 1,
                          left=0.1, right=0.97, top=0.95, bottom=0.08,
                          height_ratios=[4, 3, 1.5],
                          hspace=0.4,
                          wspace=0.3,
                          )

    gs_outputs = gs.GridSpecFromSubplotSpec(4, 3, gs_main[0, 0], wspace=0.2,
                                            hspace=0.2,
                                            width_ratios=[1.7, 1., 1.],
                                            height_ratios=[1, 1, 1, 1]
                                            )

    gs_freq = gs.GridSpecFromSubplotSpec(2, 2, gs_main[1, 0],
                                         wspace=1.0,
                                         hspace=0.9,
                                         width_ratios=[1, 1]
                                         )

    gs_weigths_loss = gs.GridSpecFromSubplotSpec(1, 2, gs_main[2, 0],
                                                 wspace=0.3,
                                                 hspace=0.2,
                                                 #width_ratios=[1, 1]
                                                 )
    return {
        ### schematics
        "small_net_model": gs_outputs[:, 0],
        "output_before": gs_outputs[0, 1],
        "output_after": gs_outputs[0, 2],
        "input_before": gs_outputs[1, 1],
        "input_after": gs_outputs[1, 2],
        "e1_before": gs_outputs[2, 1],
        "e1_after": gs_outputs[2, 2],
        "e1_1_before": gs_outputs[3, 1],
        "e1_1_after": gs_outputs[3, 2],

        "freq_0": gs_freq[0, 1],
        "phase_0": gs_freq[0, 0],
        "freq_1": gs_freq[1, 1],
        "phase_1": gs_freq[1, 0],

        "weights": gs_weigths_loss[0, 0], #gs_weigths_loss[0, 0],
        "loss": gs_weigths_loss[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
            "small_net_model",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "small_net_model",
                         "output_before",
                         "phase_0",
                         "freq_0",
                         "weights",
                         "loss"

                     ],
                     label_ypos={
                         "small_net_model": 0.97,
                         "output_before": 0.97,
                         "phase_0": 0.97,
                         "weights": 0.96,
                         "loss": 0.96,
                     },
                     label_xpos={
                         "small_net_model": -0.1,
                         "output_before": -0.2,
                         "phase_0": -0.1,
                     },
                     )


def get_fig_kwargs():
    width = 5.9
    alpha = 1.2
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into


# load data
data_path = '../results/lagnet/lagnet_049107198_small_w.pickle'
data_path_early_l = '../results/lagnet/lagnet_0512_early_214.pickle'
data_path_mid_l = '../results/lagnet/lagnet_049107198_mid_120_small_w.pickle'
data_path_late_l = '../results/lagnet/lagnet_0512_late_468.pickle'
with open(data_path, 'rb') as handle:
    data = pickle.load(handle)
with open(data_path_early_l, 'rb') as handle:
    data_early = pickle.load(handle)
with open(data_path_mid_l, 'rb') as handle:
    data_mid = pickle.load(handle)
with open(data_path_late_l, 'rb') as handle:
    data_late = pickle.load(handle)

#data = [pe, bptt]
labels = ["GLE", "AM"]
layers = ["0", "1"]
linestyle = ["-", "--"]

mod_names_mapping = {"ie": "AM",
                     "pe": "GLE",
                     "pe_prosp": "GLE",
                     "teacher": "target"}

linestyle_dict = {"ie": "--",
                  "pe": '-',
                  "pe_prosp": '-'
                  }

linestyle_output_dict = {"ie": "-",
                         "pe": '-',
                         "pe_prosp": '-',
                         "teacher": "dashed"
                         }
linestyle_output_alg_enc_dict = {"ie": "--",
                                 "pe": '-',
                                 "pe_prosp": '-',
                                 "teacher": "dashed"
                                 }
linewidth_output_dict = {"ie": 1.5,
                         "pe": 1.5,
                         "pe_prosp": 1.5,
                         "teacher": 0.5
                         }
alpha_output_dict = {"ie": .7,
                     "pe": .7,
                     "pe_prosp": .7,
                     "teacher": 1
                     }

color_weights_dict = {"ie": "C1",
                      "pe": "C2",
                      "pe_prosp": "C2",
                      "teacher": "black"
                      }

color_dict = {"ie": "darkblue",
              "pe": "cornflowerblue",
              "pe_prosp": "cornflowerblue",
              "teacher": "black"
              }

linestyle_weights_list = ['-', '--']
alpha_loss = 0.8
alpha_freq = 0.8
alpha_shift = 0.8
alpha_weights = 0.8
fs_legend = 10
label_fs = 11

learning_start_t = 7999
before = [3000, 5000]
after = [98000, 100000]

before_x_ticks = [before[0], (before[0] + before[1])/2, before[1]]
after_x_ticks = [after[0], (after[0] + after[1])/2, after[1]]
before_x_ticks_str = []
after_x_ticks_str = []
dt = 0.01

early_t = 214
mid_t = 120
late_t = 468
freq_list = [0.49 * 1 / (2 * np.pi), 1.07 / (2 * np.pi), 1.98 / (2 * np.pi)]
max_idx = int(100*0.3)

for x_ticks_l in [before_x_ticks, after_x_ticks]:
    for n, elem in enumerate(x_ticks_l):
        x_ticks_l[n] = elem * dt
        if x_ticks_l == before_x_ticks:
            before_x_ticks_str.append(str(int(x_ticks_l[n])))
        else:
            after_x_ticks_str.append(str(int(x_ticks_l[n])))

time_array = data["time"]
input_array = data["input"]
output_dict = data["outputs_per_model"]
error_dict = data["errors"]
loss_dict = data["loss_per_model"]
neurons_per_layer = data["neurons_per_layer"]
weights_teacher = data["weights_teacher"]
weights_per_model = data["weights_per_model"]
prosp_e_and_lambdas = data["prosp_e_and_lambdas"] # props_e in 'prosp_e', lambdas in 'ie'
prosp_e_and_lambdas_early = data_early["prosp_e_and_lambdas"]
prosp_e_and_lambdas_mid = data_mid["prosp_e_and_lambdas"]
prosp_e_and_lambdas_late = data_late["prosp_e_and_lambdas"]

#from early learning
prosp_e_and_lambdas_early = data_early["prosp_e_and_lambdas"]
# from mid
prosp_e_and_lambdas_mid = data_mid["prosp_e_and_lambdas"]

def plot_input(ax, time, input_array, start, end):
    ax.plot(time[start:end],
            input_array[start:end],
            color='red',
            ls='-')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['left'].set_visible(False)

def plot_input_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_input(ax,
               time=time_array,
               input_array=input_array,
               start=before[0],
               end=before[1],
               )
    ax.set_ylim(-3.3, 3.3)
    ax.set_ylabel(r"$r_0$", rotation=0, labelpad=10, fontsize=label_fs)


def plot_input_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_input(ax,
               time=time_array,
               input_array=input_array,
               start=after[0],
               end=after[1])

def plot_output(ax, time, output_dict, start, end):
    time = time[1:]
    #for key in reversed(output_dict.keys()):
    for key in ["ie", "pe", "teacher"]:
        if key == "teacher":
            ax.plot(time[start:end],
                    output_dict[key][start:end],
                    color=color_dict[key],
                    #color='red',
                    ls=linestyle_output_dict[key],
                    lw=linewidth_output_dict[key],
                    alpha=alpha_output_dict[key],
                    label='target'
                    )
        else:
            ax.plot(time[start:end],
                    output_dict[key][start:end],
                    #label=mod_names_mapping[key],
                    #color=color_dict[key],
                    color='red',
                    ls=linestyle_output_alg_enc_dict[key],
                    lw=linewidth_output_dict[key],
                    alpha=alpha_output_dict[key]
                    )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-1.2, 1.2)

def plot_output_before(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_output(ax,
                time=time_array,
                output_dict=output_dict,
                start=before[0],
                end=before[1])
    ax.set_ylabel(r"$r_3$", rotation=0, labelpad=10, fontsize=label_fs)
    ax.legend(loc='lower center', frameon=False, ncol=1,
              fontsize=fs_legend, bbox_to_anchor=(0.59, -0.23))
    ax.set_title("before learning")


def plot_output_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_output(ax,
                time=time_array,
                output_dict=output_dict,
                start=after[0],
                end=after[1])
    ax.set_title("after learning")

def plot_p_error_and_lambdas(ax, time, e_l_dict, start, end, layer="top", neuron=0, mult_factor=1.0):
    layer_n = int(layer.split("_")[1])
    #for n in [0, 1]:
    n = neuron
    for key in ["ie", "pe_prosp"]:
        if key == "ie":
            label_prefix = r'\lambda'
            label_postfix = '(AM)'
        else:
            label_prefix = 'e'
            label_postfix = '(GLE)'
        if n == 0:
            upper_idx = '\\mathrm{{i}}'
        else:
            upper_idx = '\\mathrm{{r}}'
        ax.plot(time[start:end],
                e_l_dict[key][layer][start:end, n]*mult_factor,
                color=color_dict[key],
                lw=linewidth_output_dict[key],
                label=f'${label_prefix}_{layer_n}^{upper_idx}$'# {label_postfix}'
                )
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

def plot_e1_before(ax):
    # error and lambda for fast neuron
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_p_error_and_lambdas(ax,
                             time=time_array,
                             e_l_dict=prosp_e_and_lambdas,
                             start=before[0],
                             end=before[1],
                             layer="hid_1",
                             neuron=0
                             )
    ax.set_ylim(-1.2, 1.2)  # use same y-axis as in output signal
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(r"$e_1^\mathrm{i}, \lambda_1^\mathrm{i}$", rotation=90, labelpad=10, fontsize=label_fs)

def centered_moving_average(x, w):
    # Pad the input array to ensure the output has the same length
    padding = w // 2
    x_padded = np.pad(x, (padding, padding), mode='edge')
    # Compute the moving average
    moving_avg = np.convolve(x_padded, np.ones(w) / w, mode='valid')
    return moving_avg

def plot_e1_1_before(ax):
    # error and lambda for slow neuron
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_p_error_and_lambdas(ax,
                             time=time_array,
                             e_l_dict=prosp_e_and_lambdas,
                             start=before[0],
                             end=before[1],
                             layer="hid_1",
                             neuron=1
                             )
    w = 500
    e_averaged = centered_moving_average(prosp_e_and_lambdas["pe_prosp"]["hid_1"][:, 1],
                                         w=w)
    ax.plot(time_array[before[0]:before[1]],
            e_averaged[before[0]:before[1]],
            #color=color_dict["pe_prosp"],
            color='darkorange',
            ls=(0, (3, 1, 1, 1, 1, 1)),
            alpha=0.9,
            label="moving average"
            )

    ax.set_ylim(-3.5, 3.5)  # use same y-axis as in output signal
    ax.set_xticks(before_x_ticks)
    ax.set_xticklabels(before_x_ticks_str)
    ax.set_xlabel("time [a.u.]", fontsize=label_fs)
    ax.set_ylabel(r"$e_1^\mathrm{r}, \lambda_1^\mathrm{r}$", rotation=90, labelpad=10, fontsize=label_fs)


ylim_after = 3
legend_xpos = -0.1
legend_ypos = 0.6
bbox_pos = (legend_xpos, legend_ypos)
def plot_e1_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_p_error_and_lambdas(ax,
                             time=time_array,
                             e_l_dict=prosp_e_and_lambdas,
                             start=after[0],
                             end=after[1],
                             layer="hid_1",
                             neuron=0
                             )
    ax.set_ylim(-ylim_after, ylim_after)  # use same y-axis as in output signal
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=bbox_pos, fontsize=fs_legend)


def plot_e1_1_after(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_p_error_and_lambdas(ax,
                             time=time_array,
                             e_l_dict=prosp_e_and_lambdas,
                             start=after[0],
                             end=after[1],
                             layer="hid_1",
                             neuron=1
                             )
    ax.set_ylim(-ylim_after, ylim_after)  # use same y-axis as in output signal
    ax.set_xticks(after_x_ticks)
    ax.set_xticklabels(after_x_ticks_str)
    ax.set_xlabel("time [a.u.]", fontsize=label_fs)
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=bbox_pos, fontsize=fs_legend)

import numpy as p
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_loss(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    start = 0

    for key in loss_dict.keys():
        loss = loss_dict[key][before[1]:after[1]]
        # apply moving average to loss
        window = 100
        loss = moving_average(loss, window)
        ax.plot(time_array[before[1]:after[1]-window+1],
                loss,
                color=color_weights_dict[key],
                label=mod_names_mapping[key],
                alpha=alpha_loss
                )

    ax.set_yscale("log")
    ax.set_ylabel("loss", fontsize=label_fs)
    ax.set_ylim(1e-10, 1e1)
    ax.set_xlim(time_array[before[1]], time_array[after[1] + 1])
    ax.legend(frameon=False, loc='lower left', ncol=2, fontsize=fs_legend, bbox_to_anchor=(0.0, -0.1))
    ax.set_xlabel("time [a.u.]", fontsize=label_fs)
    ax.set_xscale("log")


def plot_weights(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)
    enable_markers_bptt = 0

    layer = 'hid_1'
    n_pre_neurons = neurons_per_layer[0]
    n_post_neurons = neurons_per_layer[1]
    count = 0
    for pre_id in range(n_pre_neurons):
        for post_id in range(n_post_neurons):
            count += 1
            # plot teacher
            ax.axhline(weights_teacher[layer][post_id, pre_id],
                       linewidth=0.5,
                       color="black",
                       linestyle="dashed"
                       )
            for mod_name in reversed(weights_per_model.keys()):
                ls = linestyle_weights_list[post_id]
                if mod_name == 'ie':
                    ls = linestyle[1]
                else:
                    ls = linestyle[0]
                if layer == 'top':
                    weights_to_plot = weights_per_model[mod_name][layer][:, pre_id]
                else:
                    weights_to_plot = weights_per_model[mod_name][layer][:, post_id, pre_id]
                if post_id == 0:
                    post = 'i'
                else:
                    post = 'r'
                ax.plot(time_array[before[1]: after[1]],
                        weights_to_plot[before[1]: after[1]],
                        ls=linestyle[post_id],
                        color=color_weights_dict[mod_name],
                        label=r"$w^\mathrm{{{}}}_0$ ({})".format(post, mod_names_mapping[mod_name]),
                        alpha=alpha_weights
                        )
                if mod_name == 'ie' and enable_markers_bptt:
                    w = 300
                    marker = 'o'
                    marker_size = 1
                    init_offset = 101
                    ax.plot(time_array[learning_start_t+init_offset: after[1]: w],
                            weights_to_plot[learning_start_t+init_offset: after[1]: w],
                            linestyle='None',
                            marker=marker,
                            markersize=marker_size,
                            color=color_weights_dict[mod_name],
                            alpha=alpha_weights
                            )


    ax.vlines(mid_t, -2.5, 2.1, color='black', linestyle='dashed', linewidth=0.5)
    ax.text(mid_t, 1.55, "early", fontsize=fs_legend,
            ha='center',rotation=0)
    ax.set_ylabel('weights', fontsize=label_fs)
    ax.legend(loc='upper right', frameon=False, ncol=2, bbox_to_anchor=(1.09,0.6), fontsize=fs_legend)
    ax.set_ylim(-2.3, 1.4)
    ax.set_xlim(int(before[1]*dt) + 1, 10**3)
    ax.set_xscale("log")
    ax.set_xlabel("time [a.u.]", fontsize=label_fs)


# def plot_small_net_model(ax):
#     img = plt.imread("../images/small_net_model.png")
#     ax.imshow(img, aspect=0.9)
#     ax.axis('off')

# Freq. analysis
from scipy.fft import fft, fftfreq

# for an array of frequencies, find the index of the frequency closest an specific one
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def plot_freq(ax, freq, e_f_abs, lambda_f_abs, n_id):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    if n_id == 0:
        upper_idx = 'i'
    else:
        upper_idx = 'r'
    for f, label in zip([lambda_f_abs, e_f_abs], [rf"$\lambda^\mathrm{{{upper_idx}}}_1$", rf"$e^\mathrm{{{upper_idx}}}_1$"]):
        if label == rf"$e^\mathrm{{{upper_idx}}}_1$":
            color = color_dict['pe_prosp']
        else:
            color = color_dict['ie']


        label += " before"
        ax.plot(freq[0:max_idx] * 2 * np.pi, f[0:max_idx],
                label=label,
                color=color,
                alpha=alpha_freq)
    for idx in idx_max_f:
        ax.plot(freq[idx] * 2 * np.pi, 10**3.5, 'r+')
    ax.set_ylabel("amplitude", fontsize=label_fs, y=0.3)
    ax.set_yscale("log")

e_1 = prosp_e_and_lambdas["pe_prosp"]["hid_1"][:learning_start_t, 1]
lambda_1 = prosp_e_and_lambdas["ie"]["hid_1"][:learning_start_t, 1]
e_1_f = fft(e_1)
lambda_1_f = fft(lambda_1)
e_1_f_abs = np.abs(e_1_f)
lambda_1_f_abs = np.abs(lambda_1_f)
freq = fftfreq(len(e_1), d=dt)


e_0_mid = prosp_e_and_lambdas_mid["pe_prosp"]["hid_1"][:learning_start_t, 0]
e_1_mid = prosp_e_and_lambdas_mid["pe_prosp"]["hid_1"][:learning_start_t, 1]
e_0_f_mid = fft(e_0_mid)
e_1_f_mid = fft(e_1_mid)
e_0_f_abs_mid = np.abs(e_0_f_mid)
e_1_f_abs_mid = np.abs(e_1_f_mid)


idx_max_f = []
for f in freq_list:
    idx, f_v = find_nearest(freq, f)
    idx_max_f.append(idx)

freq_legend_xpos = -0.28
freq_legend_ypos = 0.9

def plot_freq_1(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_freq(ax,
              freq=freq,
              e_f_abs=e_1_f_abs,
              lambda_f_abs=lambda_1_f_abs,
              n_id=1)
    ax.plot(freq[0:max_idx] * 2 * np.pi,
            e_1_f_abs_mid[0:max_idx],
            color='deepskyblue',
            ls="-",
            label=r"$e^\mathrm{{r}}_1$ early"
            )
    ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=fs_legend, bbox_to_anchor=(freq_legend_xpos,freq_legend_ypos - 0.2))
    ax.set_xlabel(r"$\omega$ [2$\pi$/time]", fontsize=label_fs)
    ax.set_ylim(3*10**-0, 10**4)
    ax.set_yticks([10**1, 10**3])

e_1_ps_list = [] # phase shifts for main frequencies
lambda_1_ps_list = [] # phase shifts for main frequencies
freq_list_measured = []
for idx in idx_max_f:
    e_1_ps_list.append(np.angle(e_1_f[idx]))
    lambda_1_ps_list.append(np.angle(lambda_1_f[idx]))
    freq_list_measured.append(freq[idx])


e_0 = prosp_e_and_lambdas["pe_prosp"]["hid_1"][:learning_start_t, 0]
lambda_0 = prosp_e_and_lambdas["ie"]["hid_1"][:learning_start_t, 0]
e_0_f = fft(e_0)
lambda_0_f = fft(lambda_0)
e_0_f_abs = np.abs(e_0_f)
lambda_0_f_abs = np.abs(lambda_0_f)
freq = fftfreq(len(e_0), d=dt)


def plot_freq_0(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    plot_freq(ax,
              freq=freq,
              e_f_abs=e_0_f_abs,
              lambda_f_abs=lambda_0_f_abs,
              n_id=0)
    ax.plot(freq[0:max_idx] * 2 * np.pi,
            e_0_f_abs_mid[0:max_idx],
            color='deepskyblue',
            ls="-",
            label=r"$e^\mathrm{{i}}_1$ early"
            )
    ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=fs_legend, bbox_to_anchor=(freq_legend_xpos,freq_legend_ypos))
    ax.set_xlabel(r"$\omega$ [2$\pi$/time]", fontsize=label_fs)
    ax.set_ylim(1*10**0, 10**4)
    ax.set_yticks([10**1, 10**3])
    # ax.set_xticks([])

e_0_ps_list = [] # phase shifts for main frequencies
lambda_0_ps_list = [] # phase shifts for main frequencies
freq_list_measured = []
for idx in idx_max_f:
    e_0_ps_list.append(np.angle(e_0_f[idx]))
    lambda_0_ps_list.append(np.angle(lambda_0_f[idx]))
    freq_list_measured.append(freq[idx])


def calculate_angle(fourier):
    return np.angle(fourier) * 180 / np.pi


def min_ps_diff(ps_e, ps_lambda):
    """
    ps_ angle in degrees
    """
    delta_ps = np.zeros(len(ps_lambda))
    for s in range(len(ps_lambda)):
        potential_min_ps = [ps_e[s] - ps_lambda[s],
                            ps_e[s] - (ps_lambda[s] + 360),
                            ps_e[s] - (ps_lambda[s] - 360)]
        min_ps = find_nearest(potential_min_ps, 0)[1]
        delta_ps[s] = min_ps
    return delta_ps

e_top = prosp_e_and_lambdas["pe_prosp"]["top"][:learning_start_t]
e_top_f = fft(e_top)
e_top_ps_list = []
top_e_angle = calculate_angle(e_top_f)
activate_ps_top_e = 1

def plot_phase_0(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    e_angle = calculate_angle(e_0_f)
    lambda_angle = calculate_angle(lambda_0_f)
    delta_ps = min_ps_diff(e_angle, lambda_angle)
    omega = freq * 2 * np.pi

    ax.plot(omega[0:max_idx],
            ((e_angle - delta_ps)[0:max_idx] - top_e_angle[0:max_idx] * activate_ps_top_e) * np.pi / 180,
            color=color_dict['ie'],
            label=r"$\lambda^{{i}}_1$",
            alpha=alpha_shift)
    ax.plot(omega[0:max_idx],
            (e_angle[0:max_idx] - top_e_angle[0:max_idx] * activate_ps_top_e) * np.pi / 180,
            color=color_dict['pe_prosp'],
            label=r"$e^{{i}}_1$",
            alpha=alpha_shift)
    if not activate_ps_top_e:
        ax.plot(omega[0:max_idx],
                top_e_angle[0:max_idx] * np.pi / 180,
                color='red',
                )
    for idx in idx_max_f:
        ax.plot(omega[idx], 200 * np.pi / 180, 'r+')
    ax.set_xlabel(r"$\omega$ [2$\pi$/time]", fontsize=label_fs)
    ax.set_ylabel("phase shift", fontsize=label_fs, y=0.3)
    ax.set_yticks([-np.pi/2, 0, np.pi])
    ax.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\pi$"])
    ax.set_ylim(-np.pi/1.8, 1.3*np.pi)
    #ax.legend(loc='upper right', frameon=False, ncol=1, fontsize=8)

def plot_phase_1(ax):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)
    pos_angle = 1 # if 1, then the angles are positive

    e_angle = calculate_angle(e_1_f)
    lambda_angle = calculate_angle(lambda_1_f)
    delta_ps = min_ps_diff(e_angle, lambda_angle)
    lambda_from_e = e_angle - delta_ps
    omega = freq * 2 * np.pi
    e_1_ps = (e_angle[0:max_idx] - top_e_angle[0:max_idx] * activate_ps_top_e) * np.pi / 180
    e_1_ps_pos = np.where(e_1_ps < 0, e_1_ps.copy() + 2 * np.pi * pos_angle, e_1_ps.copy())
    lambda_1_ps = (lambda_from_e[0:max_idx] - top_e_angle[0:max_idx] * activate_ps_top_e) * np.pi / 180
    lambda_1_ps_pos = np.where(lambda_1_ps < 0, lambda_1_ps.copy() + 2 * np.pi * pos_angle, lambda_1_ps.copy())

    ax.plot(omega[0:max_idx],
            lambda_1_ps_pos,
            color=color_dict['ie'],
            label=r"$\lambda^{{r}}_1$",
            alpha=alpha_shift)
    ax.plot(omega[0:max_idx],
            e_1_ps_pos,
            color=color_dict['pe_prosp'],
            label=r"$e^{{r}}_1$",
            alpha=alpha_shift)
    if not activate_ps_top_e:
        ax.plot(omega[0:max_idx],
                top_e_angle[0:max_idx] * np.pi / 180,
                color='red',
                )
    for idx in idx_max_f:
        ax.plot(omega[idx], 200 * np.pi / 180, 'r+')
    ax.set_xlabel(r"$\omega$ [2$\pi$/time]", fontsize=label_fs)
    ax.set_ylabel("phase shift", fontsize=label_fs, y=0.3)
    ax.set_yticks([-np.pi/2, 0, np.pi])
    ax.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\pi$"])
    ax.set_ylim(-np.pi/1.8, 1.3*np.pi)
