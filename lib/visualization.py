import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""

    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

# does not use pyplot to avoid memory leaks
def plot_sliding_outputs(x, y, figsize=(12, 8)):
    """Create a new matplotlib figure containing one axis"""
    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    ax = fig.subplots(3, 1)
    ax[0].plot(x, lw=1.0)
    # calculate softmax for each output
    p = softmax(y)
    for idx in range(y.shape[-1]):
        ax[1].plot(y[:, idx], label=str(idx), color='C{}'.format(idx), alpha=0.5)
        ax[2].plot(p[:, idx], label=str(idx), color='C{}'.format(idx), alpha=0.5)
    ax[0].set_ylabel('input')
    ax[1].set_ylabel('logit output')
    ax[2].set_ylabel('softmax activation')
    ax[2].set_xlabel('sample step')
    ax[2].legend(fontsize='x-small', loc='upper right')
    return fig, ax
