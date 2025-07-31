#!/usr/bin/env python3

import torch
import argparse
import pandas as pd
import pickle

from data.datasets import get_mnist1d_splits
from mnist1d_training import mnist1d_run
from networks import GLELagWindow, GLETDNN
from lib.utils import get_loss_and_derivative
from lib.abstract_net import GLEAbstractNet
from lib.layers import GLELinear
from lib.dynamics import GLEDynamics


if __name__ == '__main__':

    # parse seed from command line
    parser = argparse.ArgumentParser(description='Train a TDNN on MNIST1D dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    params = {
        # NN parameters
        'seed': args.seed,
        'epochs': 150,
        'lr' : 5e-3,
        'batch_size': 100,
        'batch_size_test': 1000,
        'log_interval': 10,
        'checkpoint_interval': 100,
        'input_size': 10,
        'hidden_size': 60,
        'phi': 'tanh',
        'output_phi': 'linear',
        'loss_fn': 'ce',
        # LE parameters
        'dt': 0.2,
        'beta': 1.0,
        'gamma': 0.0,
        'tau': 2.0,
        'max_memory_tau': 1.8,
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

    # train using LE-TDNN
    from networks import GLETDNN
    GLETDNN.compute_target_error = loss_fn_deriv
    model = GLETDNN(input_size=params['input_size'], dt=params['dt'],
                    hidden_size=params['hidden_size'], output_size=10,
                    phi=params['phi'], output_phi=params['output_phi'],
                    tau=params['tau'], gamma=params['gamma'])


    # check for CUDA
    if torch.cuda.is_available() and params['use_cuda']:
        DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.cuda()
        print("Using CUDA.")
        print("Device: {}".format(DEVICE))
    else:
        params['use_cuda'] = False
        print("Not using CUDA.")

    memory = GLELagWindow
    memory.kwargs = {'dt': params['dt'], 'tau': params['max_memory_tau']}
    print(f"Using memory: {memory.__name__} with τ_m_max={params['max_memory_tau']} and dt={params['dt']}")

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
    fname = f"./results/mnist1d/static_lagnet_{params['seed']}_metrics.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dumped metrics to: {fname}")
