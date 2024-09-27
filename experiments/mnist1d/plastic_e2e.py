#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import pickle

from lib.gle.abstract_net import GLEAbstractNet
from lib.gle.layers import GLELinear
from lib.gle.dynamics import GLEDynamics
from data.datasets import get_mnist1d_splits

from .mnist1d_training import mnist1d_run
from .networks import E2ELagMLPNet
from lib.utils import get_loss_and_derivative, get_phi_and_derivative


if __name__ == '__main__':
    # parse parameters from command line which often change
    parser = argparse.ArgumentParser(description='Train an GLE network E2E on the MNIST1D dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    params = {
        # NN parameters
        "seed": args.seed,
        "epochs": 150,
        "lr": 5e-3,  # 5e-3 for 17 fast and 2.5e3 for 30 fast
        "batch_size": 100,
        "batch_size_test": 1000,
        "log_interval": 10,
        "checkpoint_interval": 100,
        "input_size": 1,
        "hidden_layers": 6,  # hidden lag layers
        "hidden_fast_size": 17,  # LE units
        "hidden_slow_size": 36,  # LI units
        "phi": 'tanh',
        "output_phi": 'linear',
        "loss_fn": 'ce',
        # LE parameters
        "dt": 0.2,
        "tau": 1.2,
        "beta": 1.0,
        "gamma": 0.0,
        "n_updates": 1,
        "prospective_errors": True,
        "use_cuda": True,
    }


    torch.manual_seed(params["seed"])
    print("Using seed: {}".format(params['seed']))

    # supersampling
    sample_length = 72  # original length of MNIST-1D samples in arbitrary units
    params['steps_per_sample'] = int(sample_length / params['dt']) # supersampling with factor 1/dt
    print('Using {} steps per sample.'.format(params['steps_per_sample']))

    # rescale learning rate to sample length
    params['lr'] *= sample_length / params['steps_per_sample']
    print('Using learning rate {}.'.format(params['lr']))

    loss_fn, loss_fn_deriv = get_loss_and_derivative(params['loss_fn'], params['output_phi'])
    print("Using {} loss with {} output nonlinearity.".format(params['loss_fn'], params['output_phi']))

    E2ELagMLPNet.compute_target_error = loss_fn_deriv

    model = E2ELagMLPNet(dt=params['dt'], tau=params['tau'],
                         prospective_errors=params['prospective_errors'],
                         n_hidden_layers=params['hidden_layers'],
                         hidden_fast_size=params['hidden_fast_size'],
                         hidden_slow_size=params['hidden_slow_size'],
                         phi=params['phi'], output_phi=params['output_phi'])

    print(f"Using {model.hidden_layers} hidden layers with {params['hidden_fast_size']} LE and {params['hidden_slow_size']} LI units each.")

    # check for CUDA
    if torch.cuda.is_available() and params['use_cuda']:
        DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.cuda()
        print("Using CUDA.")
        print("Device: {}".format(DEVICE))
    else:
        params['use_cuda'] = False
        print("Not using CUDA.")

    from lib.memory import Slicer
    memory = Slicer
    memory.kwargs = {}
    print("Using memory:", memory.__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=2,
                                                           factor=0.5,
                                                           verbose=True)

    torch.manual_seed(params['seed'])  # reset seed for reproducibility

    # returns metrics dictionary
    metrics = mnist1d_run(params, memory, model, loss_fn, None,
                          *get_mnist1d_splits(final_seq_length=params['steps_per_sample']),
                          optimizer=optimizer, lr_scheduler=scheduler,
                          use_le=True)

    print(f"Finished training with final test accuracy: {metrics['test_acc'][-1]:.2f}%")

    # convert metrics dict to pandas DF and dump to pickle
    df = pd.DataFrame.from_dict(metrics)
    fname = f"./results/mnist1d/plastic_e2e_{params['seed']}_metrics.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dumped metrics to: {fname}")
