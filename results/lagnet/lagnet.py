#!/usr/bin/env python3

import copy
from data.datasets import get_mnist1d_splits
import datetime
import logging
import math
import numpy as np
import os
import pickle
import shutil
import torch
import torch.nn as nn

from lib.abstract_net import GLEAbstractNet
from lib.layers import GLELinear
from lib.dynamics import GLEDynamics
from lib.utils import (check_if_path_to_results_exist_and_create_it_if_it_doesnt,
                   save_dict,
                   write_to_file,
                   check_granularity_is_sufficient,
                   init_weights_log_dict,
                   init_errors_log_dict,
                   )

class GLEnet(GLEAbstractNet, torch.nn.Module):

    def __init__(self, *, n_inputs, prospective_errors, phi, output_phi, tau,
                 tau_r, dt, gamma, bias=True, use_autodiff=False, kargs={}):
        super().__init__(full_forward=False)

        self.dt = dt
        self.tau = tau
        self.tau_r = tau_r
        self.gamma = gamma

        # new attributes for 2 hid. layer network
        self.n_inputs = kargs['n_inputs']
        self.n_hid_1 = kargs['n_hid_1']
        self.n_hid_2 = kargs['n_hid_2']
        self.n_top = kargs['n_top']
        self.tau_m_hid_1 = kargs['tau_m_hid_1']
        self.tau_r_hid_1 = kargs['tau_r_hid_1']
        self.tau_m_hid_2 = kargs['tau_m_hid_2']
        self.tau_r_hid_2 = kargs['tau_r_hid_2']

        if phi == "softplus":
            self.phi = torch.nn.Softplus()
            self.phi_prime = torch.nn.Sigmoid()
        elif phi == "tanh":
            self.phi = torch.tanh
            self.phi_prime = lambda x: 1 - torch.tanh(x)**2
        elif phi == "sigmoid":
            self.phi = torch.sigmoid
            self.phi_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif phi == "linear":
            self.phi = lambda x: x
            self.phi_prime = lambda x: torch.ones_like(x)
        else:
            raise ValueError("Unknown output activation: {}".format(phi))

        # set output activation
        if output_phi == "linear":
            self.output_phi = lambda x: x
            self.output_phi_prime = lambda x: torch.ones_like(x)
        elif output_phi == "softplus":
            self.output_phi = torch.nn.Softplus()
            self.output_phi_prime = torch.nn.Sigmoid()
        elif output_phi == "tanh":
            self.output_phi = torch.tanh
            self.output_phi_prime = lambda x: 1 - torch.tanh(x)**2
        elif output_phi == "sigmoid":
            self.output_phi = torch.sigmoid
            self.output_phi_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        else:
            raise ValueError("Unknown output activation: {}".format(output_phi))

        self.hid_1 = GLELinear(self.n_inputs, self.n_hid_1, bias=bias)
        self.hid_2 = GLELinear(self.n_hid_1, self.n_hid_2, bias=bias)
        self.top = GLELinear(self.n_hid_2, self.n_top, bias=bias)

        # no phi for hid
        self.hid_1_dynamics = GLEDynamics(conn=self.hid_1, tau_m=self.tau_m_hid_1, tau_r=self.tau_r_hid_1,
                                          dt=self.dt, prospective_errors=prospective_errors,
                                          phi=self.phi, phi_prime=self.phi_prime,
                                          gamma=self.gamma, use_autodiff=use_autodiff)
        self.hid_2_dynamics = GLEDynamics(conn=self.hid_2, tau_m=self.tau_m_hid_2, tau_r=self.tau_r_hid_2,
                                          dt=self.dt, prospective_errors=prospective_errors,
                                          phi=self.phi, phi_prime=self.phi_prime,
                                          gamma=self.gamma, use_autodiff=use_autodiff)
        self.top_dynamics = GLEDynamics(conn=self.top, tau_m=self.tau, tau_r=self.tau, dt=self.dt,
                                        prospective_errors=prospective_errors,
                                        phi=self.output_phi, phi_prime=self.output_phi_prime,
                                        use_autodiff=use_autodiff)

        self.layers_list = [ self.hid_1, self.hid_2, self.top]

    def compute_target_error(self, r, target, beta):
        e = (target - r)
        return beta * e

def rescale(*param_name_list, param_dict: dict):
    """
    rescale parameters considerinf dt, T_pres and beta

    Parameters
    ----------
    param_name_list : strings
        parameters names to rescale
    param_dict : dict
    """
    for param_name in param_name_list:
        #if param_dict[param_name].includes("lr"):
        param_dict[param_name] *= param_dict['dt'] * 1. / param_dict['beta']
        #elif param_name == "momentum":
        #    param_dict[param_name] *= params['dt']
    return param_dict

def generate_input(batch_size=1, dataset='Ornstein-Uhlenbeck', n_inputs=1,
                   extra_input_params = {}):
    """
    Input generator
    Parameters
    -----------
    kargs_options:
        sine_freq: list
            list of float frequency in Hz.
        cosine_freq: list
            list of float frequency in Hz.

    Returns
    -------
    t : array
        time array
    x : array
        input to the network
    """
    # general params for all datasets
    n_steps = int(params['T'] / params['dt'])
    t = torch.linspace(0, params['T'], n_steps)
    x = torch.zeros(batch_size, n_steps, 2)

    if dataset == 'Ornstein-Uhlenbeck':
        # generate ornstein-uhlenbeck process
        # adapted from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
        σ = 0.5
        μ = 0.0
        τ = 1.0
        sigma_bis = σ * math.sqrt(2. / τ)
        sqrtdt = math.sqrt(params['dt'])

        for i in range(n_steps - 1):
            x[:, i + 1, :] = x[:, i, :] + params['dt'] * (-(x[:, i, :] - μ) / τ) \
                + sigma_bis * sqrtdt * torch.randn(batch_size, 2)

    elif dataset == 'step function':
        # generate sequence of 0 & 1
        x[:, :, 0] = torch.zeros_like(t) - 1
        width = int(400 / params['dt'])
        for i in torch.linspace(0, n_steps, int(n_steps / width / 2) + 1):
            x[:, int(i):int(i)+width, 0] = torch.ones_like(x[:, int(i):int(i)+width, 0])

        # x[:, :, 1] = torch.roll(x[:, :, 0], int(width / 2), dims=1)
        x[:, :, 1] = torch.repeat_interleave(x[:, :, 0], 2, dim=1)[0, :n_steps]

    elif dataset == 'sine & cosine':
        # generate sine wave to check Fourier analysis
        for freq_sine in extra_input_params['sine_freq']:
            x[:, :, 0] += torch.sin(2 * math.pi * freq_sine * t)
        for freq_cosine in extra_input_params['cosine_freq']:
            x[:, :, 1] += torch.cos(2 * math.pi * freq_cosine * t)
    else:
        assert dataset == 'Ornstein-Uhlenbeck', 'Unknown dataset'

    if n_inputs == 1:
        # only use first input
        x = x[:, :, 0].unsqueeze(2)

    if extra_input_params['scale']:
        max_amplitud = torch.max(torch.abs(x))
        x /= max_amplitud
    if extra_input_params['bias'] != 0.:
        x += extra_input_params['bias']

    return t, x

def initialize_hid_identity(layers, n_inputs=1, bias=True):
    if n_inputs == 1:
        for layer in layers:
            layer.weight.data = torch.tensor([[1], [1]], dtype=torch.float32)
            if bias:
                layer.bias.data = torch.tensor([0, 0], dtype=torch.float32)
    elif n_inputs == 2:
        for layer in layers:
            layer.weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
            if bias:
                layer.bias.data = torch.tensor([0, 0], dtype=torch.float32)
    else:
        raise ValueError('Unknown number of inputs: {}'.format(n_inputs))

def initialize_top_identity(layers, bias=True):
    for layer in layers:
        layer.weight.data = torch.tensor([[1, 1]], dtype=torch.float32)
        if bias:
            layer.bias.data = torch.tensor([0], dtype=torch.float32)

def _initialize_layer_identity(layer,
                              n_pre_neurons,
                              n_post_neurons,
                              bias):
    if n_pre_neurons != n_post_neurons:
        identity_matrix = np.ones((n_post_neurons, n_pre_neurons))
    else:
        identity_matrix = np.identity(n_pre_neurons)
    layer.weight.data = torch.tensor(identity_matrix, dtype=torch.float32)
    if bias:
        layer.bias.data = torch.tensor([0] * n_post_neurons, dtype=torch.float32)

def _initialize_layer_strong_slow_path(layer,
                                       n_pre_neurons,
                                       n_post_neurons,
                                       bias,
                                       negative_fast_top=True
                                       ):
    if n_pre_neurons != n_post_neurons:
        identity_matrix = np.ones((n_post_neurons, n_pre_neurons))
        identity_matrix[0, 0] /= 2
    else:
        identity_matrix = np.identity(n_pre_neurons)
        #identity_matrix[1, 1] *= 5
        identity_matrix[0, 0] /= 2
    layer.weight.data = torch.tensor(identity_matrix, dtype=torch.float32)
    if bias:
        layer.bias.data = torch.tensor([0] * n_post_neurons, dtype=torch.float32)

def initialize_layer(mode: str,
                    layer,
                    n_pre_neurons: int,
                    n_post_neurons: int,
                    bias = True):
    if mode == 'identity':
        _initialize_layer_identity(layer,
                                 n_pre_neurons,
                                 n_post_neurons,
                                 bias=bias)
    elif mode == 'strong_slow_path':
        _initialize_layer_strong_slow_path(layer,
                                          n_pre_neurons,
                                          n_post_neurons,
                                          bias=bias)
    else:
        raise ValueError('not implemented yet')


def dL_dw(from_t: int,
          to_t: int,
          r_pre,
          dt: float,
          tau_m,
          tau_r,
          r_prime,
          backward_matrix,
          lambda_layer_above,
          n_pre: int,
          n_layer: int,
          layer_name: str,
          lambda_T=0,
          ):
    # int_{from_t}^{to_t} r_pre(t) * lambda_in_layer(t) dt
    # sum_{t=from_t}^{to_t} r_pre[t] * lambda_in_layer_[t] * dt
    debug = 0
    gradient_dim = (n_layer, n_pre)
    acum_gradient_dim = (len(r_pre), n_layer, n_pre)
    gradient_aux_acum = torch.zeros(acum_gradient_dim, requires_grad=False)
    gradient = torch.zeros(gradient_dim, requires_grad=False)
    lambda_layer = lambda_calc(tau_m=tau_m,
                               tau_r=tau_r,
                               r_prime=r_prime,
                               backward_matrix=backward_matrix,
                               lambda_layer_above=lambda_layer_above,
                               dt=dt,
                               n_neurons_layer=n_layer,
                               layer_name=layer_name,
                               lambda_T=lambda_T,
                               )
    for step in range(len(r_pre)):
        current_term = torch.mm(lambda_layer[step].t(), r_pre[step])
        gradient_aux_acum[step] = current_term
        if debug:
            logging.debug('lambda in layer')
            logging.debug(lambda_layer[step])
            logging.debug(lambda_layer[step].shape)
            logging.debug('r_pre')
            logging.debug(r_pre[step])
            logging.debug(r_pre[step].shape)
            logging.debug('CURRENT TERM')
            logging.debug(current_term)
            logging.debug(f"dLdW. step: {step}")
            logging.debug('pre rate')
            logging.debug(r_pre)
            logging.debug('pre rate (current sample)')
            logging.debug(r_pre[step])
            logging.debug('current_term')
            logging.debug(current_term)
            logging.debug('acum gradient')
            logging.debug(gradient_aux_acum)
            logging.debug("===================")
    gradient = torch.sum(gradient_aux_acum, 0) * dt  # dL_dw integrate over time
    if debug:
        logging.debug('gradient_sum')
        logging.debug(gradient  / dt)
        logging.debug('gradient.shape')
        logging.debug(gradient.shape)
        logging.debug("===================")
    return gradient, lambda_layer

def lambda_calc(tau_m: float,
                tau_r: float,
                r_prime,
                backward_matrix,
                lambda_layer_above,
                dt,
                n_neurons_layer,
                layer_name,
                lambda_T=0,
                ):
    debug = 0
    t_fin = len(r_prime)
    t_ini = 0
    lambda_dim = (t_fin, 1, n_neurons_layer) # time x 1 x dims
    lambda_in_time = torch.zeros(lambda_dim, requires_grad=False)
    for idx, step in enumerate(reversed(range(t_ini, t_fin))):
        if idx == 0:
            lambda_in_time[step] = lambda_T # + int_term
        else:
            lookback = look_back(tau=tau_r,
                                 r_prime=r_prime[step + 1],
                                 r_prime_prev=r_prime[step],
                                 backward_matrix=backward_matrix,
                                 lambda_layer_above=lambda_layer_above[step + 1],
                                 lambda_layer_above_prev=lambda_layer_above[step],
                                 dt=dt)
            int_term = torch.tensor(dt / tau_m) * lookback
            lambda_in_time[step] = int_term +  torch.exp(-torch.tensor(dt / tau_m)) * lambda_in_time[step + 1]
            if debug:
                logging.debug(f'INSIDE LAMBDA CALCULATION. step: {step}')
                logging.debug('r_prime[step]')
                logging.debug(r_prime[step])
                logging.debug('tensor tau_m')
                logging.debug(torch.tensor(tau_m))
    return lambda_in_time

def look_back(tau,
              r_prime,
              r_prime_prev,
              backward_matrix,
              lambda_layer_above,
              lambda_layer_above_prev,
              dt):
    # D^{-}_{\tau}[alpha(t)] = (1 - \tau d{}/dt) \alpha(t)
    # Euler approximation
    # D^{-}_{\tau}[alpha(t)] \approx alpha[t] - \tau/dt (alpha[t] - alpha[t-dt])
    debug = 0
    alpha = torch.mm(lambda_layer_above, backward_matrix)
    alpha *= r_prime
    alpha_prev = torch.mm(lambda_layer_above_prev, backward_matrix)
    alpha_prev *= r_prime_prev
    lookback = alpha - tau / dt * (alpha - alpha_prev)
    if debug:
        logging.debug("INSIDE LOOK_BACK")
        logging.debug('r_prime')
        logging.debug(r_prime)
        logging.debug('backward matrix')
        logging.debug(backward_matrix)
        logging.debug('lambda layer above')
        logging.debug(lambda_layer_above)
        logging.debug('lookback')
        logging.debug(lookback)
        logging.debug('lookback shape')
        logging.debug(lookback.shape)
        logging.debug('alpha')
        logging.debug(alpha)
        logging.debug('r_prime')
        logging.debug(r_prime)
        logging.debug('tau')
        logging.debug(tau)
    return lookback


if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)

    params = {
        "seed": 5, # init negative weights
        "T": 1001*1, #1000,
        # "truncation_window": 4,
        "tau": 1.0,
        "tau_r": 0.1,
        "dt": 0.01,
        "lr": 2 * 1e-2, #5 * 1e-1,
        "lr_bias": 1e-1,
        "momentum": 1 * 0.9,
        "beta": 1,
        "gamma": 10**-3,
        "batch_size": 1,
        "LE": True,
        "optimizer": 'sgd',
        'phi': 'tanh',  # nonlinearity for both hid layers
        'output_phi': 'linear',  # nonlinearity
        'log_interval': 100,
        #'dataset': 'Ornstein-Uhlenbeck',
        #'dataset': 'step function',
        'dataset': 'sine & cosine',
        #'dataset': 'mnist1d',
        # network structure
        'n_inputs': 1,
        'n_hid_1': 2,
        'n_hid_2': 2,
        'n_top': 1,
        # layer tau_m dist.
        'tau_m_hid_1': 1,
        'tau_m_hid_2': 1,
        # layer tau_r dist.
        'tau_r_hid_1': [1, 0.1],
        'tau_r_hid_2': [1, 0.1],
        'identity_teacher': True,
        'strong_slow_path': True,
        'negative_top': False, # negative weights[0][0] in top layer (for fast neuron)
        'negative_hid_1': True, # negative weights[0][0] in hid_1 layer (for fast neuron)
        'bias': False,
        'freeze_hid_1': False,
        'freeze_hid_2': True,
        'freeze_top': True,
        'force_init_w_values': False, # force init values for weights
        'students_same_params_as_teacher': False,
        'record_u_and_prosp_u': False,  # it also records errors!
        'rescale_params_dt': True,  # rescales params considering dt and beta
        'ie_enable_bptt': True, # enable bptt for ie student model
        'pe_enable_ad': False, # enable autodiff for pe student model
        'ie_prospective_errors': True, # enable prospective errores in ie_network
        'bptt_update_every_m_dts': 300*1, # update model params every m steps (dts)
        'bptt_lr': 7 * 1e-2,
        'ad_lr': 1e-0,
        'start_learning_step': 8000,  # time / dt
        'bptt_update_every_m_dts_before_learning': 8000*1, # Shold be == start_learning_step!
    }

    input_params = {
        'sine_freq': [0.49 * 1 / (2*np.pi*1) , 1.07 / (2*np.pi*1), 1.98 * 1 / (2*np.pi*1)],  # list Hz with Hz values
        'cosine_freq': [1, 2, 5],  # list Hz with Hz values
        'samples_length': 40, # sample length for mnist1d
        'mnist1d_sample': [0, 6], # first and last index
        'scale': False,  # normalize input so its within the [-1,1] range
        'bias': 0., # include bias in input so we can explore differents parts of the non-linear AF. Its applied after scaling
    }
    save_results = True
    # check if dt is sufficient to have a good numerical representation of the input signals
    if params["dataset"] == "sine & cosine":
        max_freq = check_granularity_is_sufficient(params=params, extra_input_params=input_params)
    else:
        max_freq = 100/1.2

    params['bptt_update_every_m_dts'] = int(params['bptt_update_every_m_dts'])
    params['bptt_update_every_m_dts_before_learning'] = int(params['bptt_update_every_m_dts_before_learning'])

    # rescale parameters considering dt
    if params["rescale_params_dt"]:
        #params = rescale("lr", "lr_bias", "momentum", param_dict=params)
        params = rescale("lr", "lr_bias", param_dict=params)

    if save_results:
        results_path = f"./results/lagnet/"
        check_if_path_to_results_exist_and_create_it_if_it_doesnt(results_path)
        # save a copy of the file
        print(__file__)
        file_name = os.path.basename(__file__)
        shutil.copyfile(src="./experiments/mimic/" + file_name,
                        dst=results_path + file_name)

    # set seed
    torch.manual_seed(params["seed"])

    if params['LE']:
        ie_model = GLEnet(n_inputs=params["n_inputs"], tau=params["tau"],
                          tau_r=params['tau_r'], dt=params["dt"],
                          phi=params["phi"], output_phi=params["output_phi"],
                          prospective_errors=params["ie_prospective_errors"], bias=params["bias"],
                          gamma=params["gamma"]*0, # no gamma for AM
                          kargs=params
                          )

        pe_model = GLEnet(n_inputs=params["n_inputs"], tau=params["tau"],
                          tau_r=params['tau_r'], dt=params["dt"],
                          prospective_errors=True, phi=params["phi"],
                          output_phi=params["output_phi"], bias=params["bias"],
                          gamma=params["gamma"],
                          use_autodiff=params['pe_enable_ad'],
                          kargs=params
                          )

        pe_model.load_state_dict(copy.deepcopy(ie_model.state_dict()))

        # load teacher model
        teacher = GLEnet(n_inputs=params["n_inputs"], tau=params["tau"],
                         tau_r=params['tau_r'], dt=params["dt"],
                         prospective_errors=False, phi=params["phi"],
                         output_phi=params["output_phi"], bias=params["bias"],
                         gamma=params["gamma"],
                         kargs=params
                         )

        assert not (params['freeze_hid_1'] and params['freeze_hid_2']
                    and params['freeze_top']), 'Cannot freeze both hid and top'


        layers_names_list = ['hid_1', 'hid_2', 'top']
        is_freeze_layer = [params['freeze_hid_1'], params['freeze_hid_2'], params['freeze_top']]
        neurons_per_layer_list = [params['n_inputs'], params['n_hid_1'], params['n_hid_2'], params['n_top']]
        init_mode = None
        if params['identity_teacher']:
            init_mode = 'identity'
        if params['strong_slow_path']:
            init_mode = 'strong_slow_path'

        # freeze layers
        ie_params = []
        pe_params = []
        models_params_dict = {'ie': ie_params,
                              'pe': pe_params}
        for mod_name, mod in zip(['ie', 'pe'], [ie_model, pe_model]):
            for layer_name, freeze_layer, n_pre_neurons, n_post_neurons in zip(layers_names_list,
                                                                               is_freeze_layer,
                                                                               neurons_per_layer_list[:-1],
                                                                               neurons_per_layer_list[1:]):
                layer = getattr(mod, layer_name)
                if freeze_layer:
                    if init_mode != None:
                        initialize_layer(mode=init_mode,
                                        layer=layer,
                                        n_pre_neurons=n_pre_neurons,
                                        n_post_neurons=n_post_neurons,
                                        bias=params['bias'])

                    # do not require gradients for frozen layers
                    for param in layer.parameters():
                        param.requires_grad = False

                else:
                   not_frozen_layer = layer_name
                   layer_params_list = list(layer.parameters())
                   models_params_dict[mod_name] += layer_params_list
                   if params["force_init_w_values"]:
                       # mid 120 049107198 small w
                       layer.weight.data = torch.tensor([[-0.07], [1]], dtype=torch.float32)

        if (params['lr'] != params['lr_bias'] and params['bias']):
            ie_weight = [ie_model.hid.weight, ie_model.top.weight]
            pe_weight = [pe_model.hid.weight, pe_model.top.weight]
            ie_bias = [ie_model.hid.bias, ie_model.top.bias]
            pe_bias = [pe_model.hid.bias, pe_model.top.bias]
        if init_mode != None:
            for layer, n_pre_neurons, n_post_neurons in zip(layers_names_list,
                                                            neurons_per_layer_list[:-1],
                                                            neurons_per_layer_list[1:]):
                layer = getattr(teacher, layer)
                initialize_layer(mode=init_mode,
                                layer=layer,
                                n_pre_neurons=n_pre_neurons,
                                n_post_neurons=n_post_neurons,
                                bias=params['bias'])

        if params['negative_top']:
            teacher.top.weight.data[0][0] = -2
            pe_model.top.weight.data[0][0] = -2
            ie_model.top.weight.data[0][0] = -2
        if params['negative_hid_1']:
            teacher.hid_1.weight.data[0][0] = -0.5

    print("Parameters of IE model:")
    for name, param in ie_model.named_parameters():
        print(name, param.shape, param.requires_grad)
    print("Parameters of PE model:")
    for name, param in pe_model.named_parameters():
        print(name, param.shape, param.requires_grad)

    # initialize IE and PE with same params as teacher
    if params["students_same_params_as_teacher"]:
        ie_model.load_state_dict(copy.deepcopy(teacher.state_dict()))
        pe_model.load_state_dict(copy.deepcopy(teacher.state_dict()))

    # update lr if bptt is enables
    if params['ie_enable_bptt']:
        params['ie_lr'] = params['bptt_lr']
    else:
        params['ie_lr'] = params['lr']

    if params['pe_enable_ad']:
        params['pe_lr'] = params['ad_lr']
    else:
        params['pe_lr'] = params['lr']

    # print all parameters before training
    write_to_file("Simulations params",
                  path_to_file=results_path)
    for param_name, param_value in params.items():
        write_to_file(param_name, param_value,
                      path_to_file=results_path,
                      verbose=False)
    for param_name, param_value in input_params.items():
        write_to_file(param_name, param_value,
                      path_to_file=results_path,
                      verbose=False)
    write_to_file("Parameters before training:",
                  path_to_file=results_path)
    write_to_file("Parameters ie model:",
                  path_to_file=results_path)
    for n, p in ie_model.named_parameters():
        write_to_file(p, n, path_to_file=results_path)
    write_to_file("Parameters pe model:",
                  path_to_file=results_path)
    for n, p in pe_model.named_parameters():
        write_to_file(p, n, path_to_file=results_path)

    if params['optimizer'] == 'adam':
        if params['lr'] != params['lr_bias'] and params['bias']:
            ie_optimizer = torch.optim.Adam([
                {'params': ie_weight},
                {'params': ie_bias, 'lr': params["lr_bias"]}
            ], lr=params["ie_lr"])
            pe_optimizer = torch.optim.Adam([
                {'params': pe_weight},
                {'params': pe_bias, 'lr': params["lr_bias"]}
            ], lr=params["pe_lr"])
        else:
            ie_optimizer = torch.optim.Adam(ie_params, lr=params["ie_lr"])
            pe_optimizer = torch.optim.Adam(pe_params, lr=params["pe_lr"])
    elif params['optimizer'] == 'sgd':
        if params['lr'] != params['lr_bias'] and params['bias']:
            ie_optimizer = torch.optim.SGD([
                {'params': ie_weight},
                {'params': ie_bias, 'lr': params["lr_bias"]}
            ], lr=params["ie_lr"], momentum=params["momentum"])
            pe_optimizer = torch.optim.SGD([
                {'params': pe_weight},
                {'params': pe_bias, 'lr': params["lr_bias"]}
            ], lr=params["pe_lr"], momentum=params["momentum"])
        else:
            ie_optimizer = torch.optim.SGD(ie_params, lr=params["ie_lr"],
                                        momentum=params["momentum"])
            pe_optimizer = torch.optim.SGD(pe_params, lr=params["pe_lr"],
                                           momentum=params["momentum"])
    else:
        raise ValueError('Unknown optimizer')

    loss_module = nn.MSELoss()

    ie_model.train()
    pe_model.train()
    loss = 0.

    if params['dataset'] != 'mnist1d':
        t, x = generate_input(batch_size=params['batch_size'],
                                dataset=params['dataset'],
                                n_inputs=params["n_inputs"],
                                extra_input_params=input_params)
    # mnist1D
    else:
        assert params['n_inputs'] == 1,  'n_inputs should be == 1 for mnist1d'
        n_samples = input_params['mnist1d_sample'][1] - input_params['mnist1d_sample'][0]
        n_steps_per_sample = int(input_params['samples_length'] / params['dt'])
        n_steps = int(n_steps_per_sample * n_samples)
        x = torch.zeros(params['batch_size'], n_steps, 2)
        t = torch.linspace(0, input_params['samples_length'] * n_samples, n_steps)
        train_data, _ = get_mnist1d_splits(final_seq_length=n_steps_per_sample)
        for idx, n_sample in enumerate(range(input_params['mnist1d_sample'][0],
                                             input_params['mnist1d_sample'][1])):
            first_idx = n_steps_per_sample * idx
            last_idx = n_steps_per_sample * (idx + 1)
            x[0, first_idx:last_idx, 0] = train_data[n_sample][0]

        x = x[:, :, 0].unsqueeze(2)

    log_target = torch.zeros_like(t)
    log_ie_output = torch.zeros_like(t)
    log_pe_output = torch.zeros_like(t)
    log_ie_loss = torch.zeros_like(t)
    log_pe_loss = torch.zeros_like(t)

    log_ie_errors_dict = init_errors_log_dict(time_array=t,
                                            n_hid_1=params['n_hid_1'],
                                            n_hid_2=params['n_hid_2'],
                                            )

    log_ie_lambda_dict = init_errors_log_dict(time_array=t,
                                              n_hid_1=params['n_hid_1'],
                                              n_hid_2=params['n_hid_2'],
                                              )

    log_pe_errors_dict = init_errors_log_dict(time_array=t,
                                            n_hid_1=params['n_hid_1'],
                                            n_hid_2=params['n_hid_2'],
                                            )

    log_pe_prosp_errors_dict = init_errors_log_dict(time_array=t,
                                            n_hid_1=params['n_hid_1'],
                                            n_hid_2=params['n_hid_2'],
                                            )

    log_teacher_errors_dict = init_errors_log_dict(time_array=t,
                                            n_hid_1=params['n_hid_1'],
                                            n_hid_2=params['n_hid_2'],
                                            is_teacher=True
                                            )

    log_errors_per_model_dict = {'ie': log_ie_errors_dict,
                                'pe': log_pe_errors_dict,
                                'teacher': log_teacher_errors_dict}

    log_ie_weights_dict = init_weights_log_dict(t_shape=t.shape[0],
                                                n_inputs=params['n_inputs'],
                                                n_hid_1=params['n_hid_1'],
                                                n_hid_2=params['n_hid_2'],
                                                n_top=params['n_top'],
                                                )
    log_pe_weights_dict = init_weights_log_dict(t_shape=t.shape[0],
                                                n_inputs=params['n_inputs'],
                                                n_hid_1=params['n_hid_1'],
                                                n_hid_2=params['n_hid_2'],
                                                n_top=params['n_top'],
                                                )
    log_teacher_weights_dict = init_weights_log_dict(t_shape=t.shape[0],
                                                n_inputs=params['n_inputs'],
                                                n_hid_1=params['n_hid_1'],
                                                n_hid_2=params['n_hid_2'],
                                                n_top=params['n_top'],
                                                is_teacher=True
                                                )
    log_weights_per_model_dict = {'ie': log_ie_weights_dict,
                                  'pe': log_pe_weights_dict}

    log_autodiff_grads = []
    log_adjoint_grads = []

    if params["bias"]:
        log_ie_top_bias = torch.zeros((t.shape[0], 1))
        log_pe_top_bias = torch.zeros((t.shape[0], 1))
        log_ie_hid_bias = torch.zeros((t.shape[0], 2))
        log_pe_hid_bias = torch.zeros((t.shape[0], 2))
    if params['record_u_and_prosp_u']:
        log_teacher_top_u = torch.zeros_like(t)
        log_ie_top_u = torch.zeros_like(t)
        log_pe_top_u = torch.zeros_like(t)
        log_teacher_hid_u = torch.zeros((t.shape[0], 2))
        log_ie_hid_u = torch.zeros((t.shape[0], 2))
        log_pe_hid_u = torch.zeros((t.shape[0], 2))
        log_teacher_top_prosp_u = torch.zeros_like(t)
        log_ie_top_prosp_u = torch.zeros_like(t)
        log_pe_top_prosp_u = torch.zeros_like(t)
        log_teacher_hid_prosp_u = torch.zeros((t.shape[0], 2))
        log_ie_hid_prosp_u = torch.zeros((t.shape[0], 2))
        log_pe_hid_prosp_u = torch.zeros((t.shape[0], 2))
        log_ie_top_e_inst = torch.zeros_like(t)
        log_pe_top_e_inst = torch.zeros_like(t)
        log_ie_hid_e_inst = torch.zeros((t.shape[0], 2))
        log_pe_hid_e_inst = torch.zeros((t.shape[0], 2))


    print('Training...')
    rate_mismatch_top_layer = []
    lambda_top = torch.empty((t.shape[0], params['n_hid_2'], params['n_top']))
    lambda_hid_2 = torch.empty((t.shape[0], params['n_hid_1'], params['n_hid_2']))
    lambda_hid_1 = torch.empty((t.shape[0], params['n_inputs'], params['n_hid_1']))
    log_lambda_per_layer_dict = {'top': lambda_top,
                                 'hid_2': lambda_hid_2,
                                 'hid_1': lambda_hid_1}
    log_bottom_rate_per_layer_dict = {'top': torch.empty((t.shape[0], 1, params['n_hid_2'])),
                                      'hid_2': torch.empty((t.shape[0], 1, params['n_hid_1'])),
                                      'hid_1': torch.empty((t.shape[0], 1, params['n_inputs']))}
    log_rate_prime_per_layer_dict = {'top': torch.empty((t.shape[0], 1, params['n_top'])),
                                     'hid_2': torch.empty((t.shape[0], 1, params['n_hid_2'])),
                                     'hid_1': torch.empty((t.shape[0], 1, params['n_hid_1']))}
    bptt_lambda_per_layer = {'top': [],
                             'hid_2': [],
                             'hid_1': []}
    bptt_grad_per_layer = {'top': torch.empty((params['n_hid_2'], params['n_top'])),
                           'hid_2': torch.empty((params['n_hid_1'], params['n_hid_2'])),
                           'hid_1': torch.empty((params['n_inputs'], params['n_hid_1']))}

    bptt_layers_list = ['top', 'hid_2', 'hid_1']
    bptt_pre_layers_list = ['hid_2',  'hid_1', 'inputs']
    bptt_above_layers_list = ['', 'top', 'hid_2']
    debug = 0

    pe_loss_sum = 0.

    ie_optimizer.zero_grad()
    pe_optimizer.zero_grad()

    prev_grad = {}
    for step in range(len(t)):
        input = x[:, step]
        target = teacher(input)
        if step <= params["start_learning_step"]:
            update_every = params['bptt_update_every_m_dts_before_learning']
        else:
            update_every = params['bptt_update_every_m_dts']

        # IE or adjoint dynamics
        if params['ie_enable_bptt']:
            ie_output = ie_model(input, target, beta=params['beta'])
            ie_loss = loss_module(ie_output, target)

            rate_mismatch_top_layer.append([[target[0] - ie_output[0]]])

            for layer_name, pre_layer_name in zip(bptt_layers_list, bptt_pre_layers_list):
                if pre_layer_name == 'inputs':
                    # pre layer dynamics is the input
                    log_bottom_rate_per_layer_dict[layer_name][step] = input
                else:
                    pre_layer_dynamics = getattr(ie_model, pre_layer_name + '_dynamics')
                    log_bottom_rate_per_layer_dict[layer_name][step] = pre_layer_dynamics.r
                layer_dynamics = getattr(ie_model, layer_name + '_dynamics')
                log_rate_prime_per_layer_dict[layer_name][step] = layer_dynamics.r_prime

            if step % update_every == 0 and step > 0:
                init_t = int(step - update_every)
                final_t = int(step)
                for layer_name, layer_above_name, pre_layer_name in zip(bptt_layers_list, bptt_above_layers_list, bptt_pre_layers_list):
                    logging.debug('INSIDE BPTT')
                    logging.debug('layer')
                    logging.debug(layer_name)
                    logging.debug('pre layer')
                    logging.debug(pre_layer_name)
                    if layer_name == 'top':
                        lambda_layer_above = torch.tensor(rate_mismatch_top_layer, requires_grad=False)
                        tau_m = torch.tensor([[params['tau']]], requires_grad=False)
                        tau_r = torch.tensor([[params['tau']]], requires_grad=False)
                        backward_matrix = torch.ones((1, 1))
                        lambda_T = tau_r /tau_m * lambda_layer_above[step-1][0][0] * log_rate_prime_per_layer_dict[layer_name][step-1]
                    else:
                        lambda_layer_above = bptt_lambda_per_layer[layer_above_name]
                        tau_r = torch.tensor([params[f'tau_r_{layer_name}']], requires_grad=False)
                        tau_m = torch.tensor([[params[f'tau_m_{layer_name}']] * tau_r.shape[1]], requires_grad=False)
                        layer_above = getattr(ie_model, layer_above_name)
                        forward_matrix = getattr(layer_above, 'weight')
                        backward_matrix = forward_matrix #.t()
                        lambda_T = tau_r / tau_m * lambda_layer_above[step-1][0][0] * log_rate_prime_per_layer_dict[layer_name][step-1]
                    n_pre = params[f'n_{pre_layer_name}']
                    n_layer = params[f'n_{layer_name}']
                    layer.weight.grad = torch.zeros(layer.weight.shape)

                    bptt_grad_per_layer[layer_name], \
                    bptt_lambda_per_layer[layer_name][init_t:final_t] = dL_dw(from_t=init_t,
                                                                              to_t=final_t,
                                                                              r_pre=log_bottom_rate_per_layer_dict[layer_name][init_t:final_t],
                                                                              dt=params['dt'],
                                                                              tau_m=tau_m,
                                                                              tau_r=tau_r,
                                                                              r_prime=log_rate_prime_per_layer_dict[layer_name][init_t:final_t],
                                                                              backward_matrix=backward_matrix,
                                                                              lambda_layer_above=lambda_layer_above[init_t:final_t],
                                                                              n_pre=n_pre,
                                                                              n_layer=n_layer,
                                                                              layer_name=layer_name,
                                                                              lambda_T=lambda_T,
                                                                              )
                    layer = getattr(ie_model, layer_name)
                    if step <= params["start_learning_step"]:
                        layer.weight.grad = - bptt_grad_per_layer[layer_name]*0
                    else:
                        layer.weight.grad = - bptt_grad_per_layer[layer_name]
                    logging.debug('BPTT UPDATE')
                    logging.debug('layer')
                    logging.debug(layer_name)
                    logging.debug('grad shape')
                    logging.debug(layer.weight.grad.shape)
                    logging.debug('grad updated')
                    logging.debug(layer.weight.grad)

                # print and log Adjoint gradients
                if params['ie_enable_bptt']:
                    for name, param in ie_model.named_parameters():
                        if param.requires_grad:
                            print("IE/Adjoint", name, "gradient", param.grad.t().numpy())
                    ie_not_frozen_layer_obj = getattr(ie_model, not_frozen_layer)
                    log_adjoint_grads.append(ie_not_frozen_layer_obj.weight.grad.clone().detach())

                ie_optimizer.step()
                ie_optimizer.zero_grad()
            else:
                pass

        else:
            # instantaneous errors
            ie_optimizer.zero_grad()
            with torch.no_grad():
                ie_output = ie_model(input, target, beta=params['beta'])
                ie_loss = loss_module(ie_output, target)
                ie_optimizer.step()

        # PE or autodiff
        pe_output = pe_model(input, target, beta=params['beta'])  # with non-zero apicals that are not used to update weights
        # pe_output = pe_model(input)
        pe_loss = loss_module(pe_output, target)
        if params["pe_enable_ad"]:
            # accumulate loss for autodiff a la truncated BPTT
            pe_loss_sum += pe_loss * params['dt']

            # update AD model using accumulated loss
            if step % params['bptt_update_every_m_dts'] == 0 and step > 0:

                # truncated BPTT: autodiff through accumulated loss
                pe_loss_sum.backward()

                # print and log AD gradients
                if params['pe_enable_ad']:
                    for name, param in pe_model.named_parameters():
                        if param.requires_grad:
                            print("PE/AD", name, "gradient", param.grad.t().numpy())
                    pe_not_frozen_layer_obj = getattr(pe_model, not_frozen_layer)
                    log_autodiff_grads.append(pe_not_frozen_layer_obj.weight.grad.clone().detach())
                    #log_autodiff_grads.append(pe_model.hid_1.weight.grad.clone().detach())

                pe_optimizer.step()
                pe_model._detach_dynamic_variables()
                pe_loss_sum = 0.

        else:
            if step <= params["start_learning_step"]:
                pass
            else:
                pe_optimizer.step()

        pe_optimizer.zero_grad()

        log_target[step] = target[0]
        log_ie_output[step] = ie_output[0]
        log_pe_output[step] = pe_output[0].clone().detach()
        log_ie_loss[step] = ie_loss
        log_pe_loss[step] = pe_loss.clone().detach()

        for layer_name in layers_names_list:
            for mod, mod_name in zip([ie_model, pe_model], ['ie', 'pe']):
                # log weights
                layer = getattr(mod, layer_name)
                log_weights_per_model_dict[mod_name][layer_name][step] = getattr(layer, 'weight').detach().clone()
                # log errors
                layer_dynamics = getattr(mod, layer_name + '_dynamics')
                errors = getattr(layer_dynamics, 'inst_e')
                if mod == pe_model:
                    pe_prosp_errors = getattr(layer_dynamics, 'prosp_v')
                    log_pe_prosp_errors_dict[layer_name][step] = pe_prosp_errors[0]
                log_errors_per_model_dict[mod_name][layer_name][step] = errors[0]

            # log teacher weights in first iteration
            if step == 0:
                layer = getattr(teacher, layer_name)
                log_teacher_weights_dict[layer_name] = getattr(layer, 'weight')

        if params["bias"]:
            log_ie_top_bias[step] = ie_model.top.bias
            log_ie_hid_bias[step] = ie_model.hid.bias
            log_pe_top_bias[step] = pe_model.top.bias
            log_pe_hid_bias[step] = pe_model.hid.bias
        if params['record_u_and_prosp_u']:
            log_teacher_top_u[step] = teacher.top_dynamics.u[0]
            log_ie_top_u[step] = ie_model.top_dynamics.u[0]
            log_pe_top_u[step] = pe_model.top_dynamics.u[0]
            log_teacher_hid_u[step] = teacher.hid_dynamics.u[0]
            log_ie_hid_u[step] = ie_model.hid_dynamics.u[0]
            log_pe_hid_u[step] = pe_model.hid_dynamics.u[0]
            log_teacher_top_prosp_u[step] = teacher.top_dynamics.prosp_u[0]
            log_ie_top_prosp_u[step] = ie_model.top_dynamics.prosp_u[0]
            log_pe_top_prosp_u[step] = pe_model.top_dynamics.prosp_u[0]
            log_teacher_hid_prosp_u[step] = teacher.hid_dynamics.prosp_u[0]
            log_ie_hid_prosp_u[step] = ie_model.hid_dynamics.prosp_u[0]
            log_pe_hid_prosp_u[step] = pe_model.hid_dynamics.prosp_u[0]

        if step % params['log_interval'] == 0:
            print(f'\nStep: {step} | IE Loss: {"{:2e}".format(ie_loss)} | PE Loss: {"{:2e}".format(pe_loss)}')
            write_to_file(f'Step: {step} | IE Loss: {"{:2e}".format(ie_loss)} | PE Loss: {"{:2e}".format(pe_loss)}',
                                path_to_file=results_path)

    common_fname = results_path + (f'{params["dataset"].replace(" ", "_")}_with_'
                                               f'{params["n_inputs"]}_inputs_frozenHid1={params["freeze_hid_1"]}_'
                                               f'frozenHid2={params["freeze_hid_2"]}_'
                                               f'frozenTop={params["freeze_top"]}')

    title = params['dataset'] + ' input with ' + params['freeze_top'] * " frozen top " + params['freeze_hid_2'] * " frozen hid 2 " + \
    params['freeze_hid_1'] * "frozen hid 1 " + \
    " $\\tau_m$ = " + str(params['tau']) + " & $\\tau_r$ = " + str(params['tau_r']) + ", $\\phi_{hid}$ = " + str(params['phi']) + \
    " and $\\phi_{top}$ = " + str(params["output_phi"])

    # calculate Fourier spectrum
    from scipy.fft import fft, fftfreq
    idx = int(len(t)/10)
    x1_fft = np.abs(fft(x[0, -idx:, 0].numpy()))
    if params["n_inputs"] > 1:
        x2_fft = np.abs(fft(x[0, -idx:, 1].numpy()))
    ie_fft = np.abs(fft(log_ie_output[-idx:].numpy()))
    pe_fft = np.abs(fft(log_pe_output[-idx:].numpy()))
    target_fft = np.abs(fft(log_target[-idx:].numpy()))
    x_fft = fftfreq(len(t[-idx:]), d=params['dt'])

    log_fft_dict = {'teacher': target_fft,
                     'ie': ie_fft,
                     'pe': pe_fft,
                     'x_axis': x_fft}

    loss_first_idx = 2
    log_loss_dict = {'ie': log_ie_loss[loss_first_idx:],
                     'pe': log_pe_loss[loss_first_idx:]}

    output_first_idx = 1
    log_outputs_dict = {'teacher': log_target[output_first_idx:],
                     'ie': log_ie_output[output_first_idx:],
                     'pe': log_pe_output[output_first_idx:]}

    log_ins_and_prosp_errors_per_model_dict = {'pe': log_errors_per_model_dict['pe'],
                                               'pe_prosp': log_pe_prosp_errors_dict
                                               }

    log_ie_lambda_dict['top'] = np.array(log_ie_lambda_dict['top'])
    log_ie_lambda_dict['hid_1'] = np.array(log_ie_lambda_dict['hid_1'])
    log_ie_lambda_dict['hid_2'] = np.array(log_ie_lambda_dict['hid_2'])

    t_steps = len(bptt_lambda_per_layer['top'])
    t_len_minus_bptt = len(t) - t_steps
    log_ie_lambda_dict['top'][:-t_len_minus_bptt] = np.array(bptt_lambda_per_layer['top']).reshape(t_steps)
    log_ie_lambda_dict['hid_1'][:-t_len_minus_bptt :] = np.array(bptt_lambda_per_layer['hid_1']).reshape(t_steps, 2)
    log_ie_lambda_dict['hid_2'][:-t_len_minus_bptt :] = np.array(bptt_lambda_per_layer['hid_2']).reshape(t_steps, 2)

    log_prosp_e_vs_lambda_per_model_dict = {'ie': log_ie_lambda_dict,
                                            'pe_prosp': log_pe_prosp_errors_dict
                                            }

    # dictionary to plot in the manuscript
    to_numpy_dicts = [
                      log_teacher_weights_dict,
                      log_teacher_errors_dict,
                      log_outputs_dict,
                      #log_errors_per_model_dict,
                      log_weights_per_model_dict,
                      log_loss_dict
                     ]

    def convert_tensors_to_numpy_dict(input_dict):
        converted_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                if hasattr(value, 'grad'):
                    converted_dict[key] = value.detach().numpy()
                else:
                    converted_dict[key] = value.numpy()
            elif isinstance(value, dict):
                converted_dict[key] = convert_tensors_to_numpy_dict(value)
            elif isinstance(value, (list, tuple)):
                converted_dict[key] = [convert_tensors_to_numpy_dict(item) for item in value]
            else:
                converted_dict[key] = value
        return converted_dict

    data = {
        "input": x[0, :, 0].numpy(),
        "target": log_target.numpy(),
        "weights_teacher": convert_tensors_to_numpy_dict(log_teacher_weights_dict),
        "time": t.numpy(),
        "neurons_per_layer": neurons_per_layer_list,
        "outputs_per_model": convert_tensors_to_numpy_dict(log_outputs_dict),
        "errors": convert_tensors_to_numpy_dict(log_errors_per_model_dict),
        "weights_per_model": convert_tensors_to_numpy_dict(log_weights_per_model_dict),
        "loss_per_model": convert_tensors_to_numpy_dict(log_loss_dict), #teacher included
        "prosp_e_and_lambdas": convert_tensors_to_numpy_dict(log_prosp_e_vs_lambda_per_model_dict)
    }

    with open(f'{results_path}/lagnet_results_seed_{params["seed"]}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save params dicts
    save_dict(dict2save=params, fname="params",
              path_to_file=results_path)
    save_dict(dict2save=input_params, fname="input_params",
              path_to_file=results_path)

    # print all parameters after training
    write_to_file('Parameters after training:',
                  path_to_file=results_path)
    write_to_file('teacher model:',
                  path_to_file=results_path)
    for n, p in teacher.named_parameters():
        write_to_file(p, n, path_to_file=results_path)
    write_to_file("\n", path_to_file=results_path)
    write_to_file('IE model after training:',
                  path_to_file=results_path)
    for n, p in ie_model.named_parameters():
        write_to_file(p, n, path_to_file=results_path)
    write_to_file("\n", path_to_file=results_path)
    write_to_file('PE model after training',
                  path_to_file=results_path)
    for n, p in pe_model.named_parameters():
        write_to_file(p, n, path_to_file=results_path)
