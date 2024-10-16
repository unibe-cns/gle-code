#!/usr/bin/env python

import argparse
import torch
import numpy as np

from .gsc_training import gsc_run
from lib.gle.abstract_net import GLEAbstractNet
from lib.gle.layers import GLELinear
from lib.gle.dynamics import GLEDynamics
from lib.utils import get_loss_and_derivative, get_phi_and_derivative
from data.gsc import get_gsc_dataloaders2, ALL_LABELS, KW12_LABELS

import pandas as pd
import pickle


class E2ELagMLPNet(GLEAbstractNet, torch.nn.Module):

    def __init__(self, *, dt, prospective_errors=False, input_size=32,
                 n_hidden_layers=4, tau_ms=[1.1, 1.1], tau_rs=[1.1, 0.1],
                 hidden_sizes=[100, 300], output_size=12, phi='tanh',
                 lag_phi='tanh', output_phi='linear'):
        super().__init__(full_forward=False, full_backward=False)

        self.output_size = output_size

        self.dt = dt

        self.lag_phi, self.lag_phi_prime = get_phi_and_derivative(lag_phi)
        self.phi, self.phi_prime = get_phi_and_derivative(phi)
        self.output_phi, self.output_phi_prime = get_phi_and_derivative(output_phi)

        layers = []
        dyns = []

        hidden_size = sum(hidden_sizes)

        assert (len(tau_ms) == 1 and len(tau_rs) >=1) or (len(tau_ms) >= 1 and len(tau_rs) == 1) or (len(tau_ms) == len(tau_rs)), "tau_ms and tau_rs must be the same length or one of them must be of length 1."

        tau_m = self.dt * torch.ones(hidden_size)
        tau_r = self.dt * torch.ones(hidden_size)

        current_lower = 0

        for i, hidden_size_i in enumerate(hidden_sizes):
            tau_m_i = tau_ms[i] if len(tau_ms) > i else tau_ms[-1]
            tau_r_i = tau_rs[i] if len(tau_rs) > i else tau_rs[-1]

            tau_m[current_lower:current_lower + hidden_size_i] = tau_m_i
            tau_r[current_lower:current_lower + hidden_size_i] = tau_r_i
            current_lower += hidden_size_i

        print("Using tau_r:", tau_r)

        # clamp taus to dt and 100
        self.tau_r = tau_r.clamp(self.dt, 100)
        self.tau_m = tau_m.clamp(self.dt, 100)

        # input layer
        layer = GLELinear(input_size, hidden_size)
        layers.append(layer)
        dyns.append(GLEDynamics(layer, dt=self.dt, tau_m=self.tau_m,
                                tau_r=self.tau_r,
                                #phi=self.lag_phi, phi_prime=self.lag_phi_prime,
                                prospective_errors=prospective_errors))

        # half lagged, half instantaneous
        for i in range(n_hidden_layers - 1):
            layer = GLELinear(hidden_size, hidden_size)
            layers.append(layer)
            dyns.append(GLEDynamics(layer, dt=self.dt, tau_m=self.tau_m,
                                   tau_r=self.tau_r, phi=self.lag_phi,
                                   phi_prime=self.lag_phi_prime,
                                   prospective_errors=prospective_errors))

        # instantaneous output layer
        layer = GLELinear(hidden_size, self.output_size)
        layers.append(layer)
        dyns.append(GLEDynamics(layer, dt=self.dt,
                               tau_m=torch.ones(self.output_size) * tau_m[0],
                               tau_r=torch.ones(self.output_size) * tau_m[0],
                               phi=self.output_phi,
                               phi_prime=self.output_phi_prime,
                               prospective_errors=prospective_errors))

        # turn all variables in layers into attributes of the model
        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)

        # turn all variables in dyns into attributes of the model
        for i, dyn in enumerate(dyns):
            setattr(self, f'dyn_{i}', dyn)

        self.hidden_layers = n_hidden_layers

        print("Initialized {} model with {} parameters".format(self.__class__.__name__, self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def compute_target_error(self, r, r_prime, target, beta):
        pass

if __name__ == "__main__":

    params = {
        # NN parameters
        "input_size": 1,
        "hidden_layers": 3,
        "hidden_sizes": [150, 150, 150, 150, 150],
        "optimizer": 'adam',  # 'sgd',
        "lag_phi": 'tanh',
        'phi': 'tanh',
        'output_phi': 'linear',
        'loss_fn': 'ce',
        # gsc parameters
        'dataset_type': 'kw12',
        'randomsample_on_val_test': False,
        'tf1_splits': True, # this will split the GSC data into train, val, test like tensorflow 1 did
        # memory parameters
        'memory_type': 'delay_line',
        # other parameters
        "use_cuda": True,
        "log_interval": 1e1000,
        # LE parameters
        'dt': 0.05,
        'beta': 1,
        'gamma': 0.0,
        'tau': 1.0,
        'settling_steps': 0,
        'scale_fft_stride_by_dt': False, # if False, interpolate by 1/dt
        'fft_type': 'mel',
        'fft_window': 1024,
        'fft_stride': 400,
        'use_le': True,
        'running_metrics': True,
    }

    CLASSES = ALL_LABELS if params['dataset_type'] == 'full' else KW12_LABELS

    # parse parameters from command line which often change
    parser = argparse.ArgumentParser(description='Train a GLE network on the MNIST1D dataset.')
    parser.add_argument('--seed', type=int, default=115, help='Random seed.')
    parser.add_argument("--batch-size", type=int, default=256, help='batch size')
    # parser.add_argument("--weight-decay", type=float, default=0, help='weight decay')
    parser.add_argument("--optim", choices=['sgd', 'adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--learning-rate", type=float, default=5e-4, help='learning rate for optimization')
    parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=3, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.9, help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--max-epochs", type=int, default=420, help='max number of epochs')
    parser.add_argument("--n-mels", choices=[32, 40], default=32, help='input of NN')
    parser.add_argument('--prospective_errors', action='store_true', default=True, help='Use prospective errors.')
    parser.add_argument('--tau_m', type=float, action='append', default=[2.4, 0.6, 1.2, 1.8, 2.4], help='Set base tau from which all other taus are derived.')
    parser.add_argument('--tau_r', type=float, action='append', default=[2.4, 0.1, 0.1, 0.1, 0.1], help='Set base tau from which all other taus are derived.')
    args = parser.parse_known_args()[0]

    params['seed'] = args.seed
    print("Using seed: {}".format(params['seed']))

    # Try attaching to GPU
    params['use_cuda'] = True if torch.cuda.is_available() else False
    DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Using:', DEVICE)

    params['epochs'] = args.max_epochs
    print("Using max epochs: {}".format(params['epochs']))

    params['batch_size'] = args.batch_size
    print("Using batch size: {}".format(params['batch_size']))

    params['lr'] = args.learning_rate
    # print("Using learning rate: {}".format(params['lr']))

    params['optimizer'] = args.optim
    print("Using optimizer: {}".format(params['optimizer']))

    params['lr_scheduler'] = args.lr_scheduler
    print("Using lr scheduler: {}".format(params['lr_scheduler']))

    params['lr_scheduler_patience'] = args.lr_scheduler_patience
    print("Using lr scheduler patience: {}".format(params['lr_scheduler_patience']))

    params['lr_scheduler_step_size'] = args.lr_scheduler_step_size
    print("Using lr scheduler step size: {}".format(params['lr_scheduler_step_size']))

    params['lr_scheduler_gamma'] = args.lr_scheduler_gamma
    print("Using lr scheduler gamma: {}".format(params['lr_scheduler_gamma']))

    print("Using LE: {}".format(params['use_le']))

    params['n_mels'] = args.n_mels
    print("Using n_mels: {}".format(params['n_mels']))

    params['classes'] = len(CLASSES)
    print("Using {} classes.".format(params['classes']))

    params['prospective_errors'] = args.prospective_errors
    print("Using prospective errors: {}".format(params['prospective_errors']))

    assert np.all([np.abs(tau) >= params['dt'] for tau in args.tau_m]), "τ_m must be larger than or equal to dt."
    params['tau_m'] = args.tau_m
    print("Using tau_m: {}".format(params['tau_m']))

    assert np.all([np.abs(tau) >= params['dt'] for tau in args.tau_r]), "τ_r must be larger than or equal to dt."
    params['tau_r'] = args.tau_r
    print("Using tau_r: {}".format(params['tau_r']))

    # dt = 1 means we use the original sample length
    original_sample_length = 16000 / params['fft_stride'] # 16kHz sample rate of ~1s is 16000 samples, and we stride by params['fft_stride']
    print('Original sample length:', original_sample_length)
    # # supersample input if dt is not 1
    params['steps_per_sample'] = int(original_sample_length / params['dt'])
    assert params['steps_per_sample'] >= params['input_size'], 'steps_per_sample length must be >= input size'
    print('Using {} steps per sample.'.format(params['steps_per_sample']))

    # rescale learning rate if steps per sample is not 40
    params['lr'] *= params['dt']  * (16000 / 512 / 0.05) / params['steps_per_sample'] # varies with dt and if steps_per_sample is not 640 (= 16000 / 512 / 0.05)
    print('Using learning rate {}.'.format(params['lr']))

    # adapt seq length by supersampling or higher mels
    params['interpolate'] = 1 / params['dt']

    loss_fn, loss_fn_deriv = get_loss_and_derivative(params['loss_fn'], params['output_phi'])
    print("Using {} loss with {} output nonlinearity.".format(params['loss_fn'], params['output_phi']))


    torch.manual_seed(params["seed"])

    if params["use_le"]:
        E2ELagMLPNet.compute_target_error = loss_fn_deriv

        model = E2ELagMLPNet(dt=params['dt'], 
                             prospective_errors=params['prospective_errors'],
                             input_size=params['n_mels'],
                             n_hidden_layers=params['hidden_layers'],
                             hidden_sizes=params['hidden_sizes'],
                             tau_ms=params['tau_m'], tau_rs=params['tau_r'],
                             output_size=len(CLASSES),
                             phi=params['phi'], lag_phi=params['lag_phi'],
                             output_phi=params['output_phi'])

        params['tau_m_tensor'] = str(model.tau_m)
        params['tau_r_tensor'] = str(model.tau_r)

        print(f"Using {model.hidden_layers} hidden layers with {params['hidden_sizes']} LE/LI neurons of taus {params['tau_m']}/{params['tau_r']} in each layer.")

    else:
        raise NotImplementedError('Only implemented with a GLE-TDNN model.')

    if params['use_cuda']:
        model.cuda()

    from lib.memory import Slicer
    memory = Slicer
    memory.kwargs = {}
    print("Using memory:", memory.__name__)
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    if params['lr_scheduler'] == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['lr_scheduler_patience'], factor=params['lr_scheduler_gamma'], eps=1e-20)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['lr_scheduler_step_size'], gamma=params['lr_scheduler_gamma'], last_epoch=-1)

    torch.manual_seed(params["seed"]) # reset seed for reproducibility
    metrics = gsc_run(params, memory, model, loss_fn, None,
                      *get_gsc_dataloaders2(params['n_mels'], params, params['use_cuda'],
                                            interpolate=params['interpolate'], fft_type=params['fft_type'],
                                            fft_window=params['fft_window'], fft_stride=params['fft_stride'],
                                            randomsample_on_val_test=params['randomsample_on_val_test'],
                                            tf1_splits=params['tf1_splits'], random_seed=params['seed']),
                      optimizer, lr_scheduler, use_le=True)

    print(f"Finished training with final test accuracy: {metrics['test_acc'][-1]:.2f}%")

    # convert train & valid metrics to pandas DF and dump to pickle
    metrics = {k: v for k, v in metrics.items() if k in ['train_acc', 'val_loss', 'val_acc']}
    df = pd.DataFrame.from_dict(metrics)
    fname = f"./results/gsc/GLE_{params['seed']}_val_acc.csv"
    df.to_csv(fname, header=False, index=False)
    print(f"Dumped metrics to: {fname}")
