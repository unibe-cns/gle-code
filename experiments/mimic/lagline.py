#!/usr/bin/env python3

import torch
import torch.nn as nn

from lib.gle.abstract_net import GLEAbstractNet
from lib.gle.layers import GLELinear
from lib.gle.dynamics import GLEDynamics
from lib.utils import get_loss_and_derivative


class LagLine(GLEAbstractNet, torch.nn.Module):

    def __init__(self, *, tau_m, tau_r, dt, gamma=1.0, bias=False,
                 prospective_errors=False, use_autodiff=False):
        super().__init__()

        self.tau_m_0 = torch.tensor([tau_m[0]])
        self.tau_m_1 = torch.tensor([tau_m[1]])
        self.tau_r = torch.tensor([tau_r])

        self.dt = dt

        self.phi = torch.nn.Softplus()
        self.phi_prime = lambda x: torch.sigmoid(x)

        self.lin0 = GLELinear(1, 1, bias=bias)
        self.lin1 = GLELinear(1, 1, bias=bias)

        # setup includes two slow layers
        self.lin0_dynamics = GLEDynamics(self.lin0, tau_m=self.tau_m_0, tau_r=self.tau_r,
                                         dt=self.dt, gamma=gamma, learn_tau=True,
                                         phi=self.phi, phi_prime=self.phi_prime,
                                         prospective_errors=prospective_errors,
                                         use_autodiff=use_autodiff)
        self.lin1_dynamics = GLEDynamics(self.lin1, tau_m=self.tau_m_1, tau_r=self.tau_r,
                                         dt=self.dt, gamma=gamma, learn_tau=True,
                                         phi=self.phi, phi_prime=self.phi_prime,
                                         prospective_errors=prospective_errors,
                                         use_autodiff=use_autodiff)


if __name__ == "__main__":

    # default parameters
    params = {
        "seed": 0,
        "dt": 0.01,
        "tau_m": [1.0, 2.0],
        "tau_r": 0.1,
        "beta": 1e-2,
        "lr": 1e-6,
        "gamma": 1.0,
        "T_init": 50,
        "T_train": 4000,
        "LE": True,
        "batch_size": 100,
        "logging_interval": 10,
        "use_neptune": True,
    }

    import argparse
    parser = argparse.ArgumentParser(description='Train a LagLine model.')
    parser.add_argument('--model', type=str, default='gle', help='Experiment name.')
    args = parser.parse_args()

    if args.model == 'bp':
        params['prospective_errors'] = False
        params['use_autodiff'] = False
        params['truncation_steps'] = 1
    elif args.model == 'gle':
        params['prospective_errors'] = True
        params['use_autodiff'] = False
        params['truncation_steps'] = 1
    elif args.model == 'bptt_tw4':
        params['use_autodiff'] = True
        params['prospective_errors'] = False
        params['truncation_steps'] = int(4 / params['dt'])
    elif args.model == 'bptt_tw2':
        params['use_autodiff'] = True
        params['prospective_errors'] = False
        params['truncation_steps'] = int(2 / params['dt'])
    elif args.model == 'bptt_tw1':
        params['use_autodiff'] = True
        params['prospective_errors'] = False
        params['truncation_steps'] = int(1 / params['dt'])
    else:
        raise ValueError('Unknown model type.')

    # calculate number of steps
    params['T'] = params['T_init'] + params['T_train']
    # no rescaling with beta needed because error is not calculated from difference
    params['lr'] /= params['beta']  # rescale learning rate with beta
    if params['truncation_steps'] > 1 and params['use_autodiff']:
        assert not params['prospective_errors'], 'truncation_steps requires no prospective_errors'
        params['lr'] *= params['truncation_steps']  # rescale learning rate for truncated BPTT
        print('Using truncated BPTT with {} steps'.format(params['truncation_steps']))
    else:
        params['truncation_steps'] = 1  # set to default value if not used
        print('Using GLE with' + ('' if params['prospective_errors'] else 'out') + ' prospective errors')

    # define dict for logging
    log = {'time': [], 'input': [], 'target': [], 'output': [], 'loss': [],
           'u_0': [], 'v_0': [], 'prosp_u_0': [], 'r_0': [], 'prosp_e_0': [],
           'u_1': [], 'v_1': [], 'prosp_u_1': [], 'r_1': [], 'prosp_e_1': [],
           'tau_m_0': [], 'tau_m_1': [], 'w_0': [], 'w_1': []}

    torch.manual_seed(params["seed"])

    loss_fn, loss_deriv  = get_loss_and_derivative('mse')
    # assign target error function to model class (before instantiation)
    LagLine.compute_target_error = loss_deriv

    model = LagLine(tau_m=params['tau_m'],
                    tau_r=params['tau_r'],
                    dt=params['dt'],
                    gamma=params['gamma'],
                    prospective_errors=params['prospective_errors'],
                    use_autodiff=params['use_autodiff'])
    teacher = LagLine(tau_m=params['tau_m'],
                      tau_r=params['tau_r'],
                      dt=params['dt'],
                      gamma=params['gamma'],
                      prospective_errors=False,
                      use_autodiff=False)

    # set teacher weights
    teacher.lin0.weight.data = torch.tensor([[1.0]])
    teacher.lin1.weight.data = torch.tensor([[2.0]])

    # change time constants for teacher network
    teacher.lin0_dynamics.tau_m.data = torch.tensor([1.0], requires_grad=False)
    teacher.lin1_dynamics.tau_m.data = torch.tensor([2.0], requires_grad=False)

    # copy teacher weights to model
    model.load_state_dict(teacher.state_dict())

    params_to_learn = []
    # overwrite time constants to random values
    model.lin0_dynamics.tau_m.data = torch.rand_like(model.lin0_dynamics.tau_m.data).requires_grad_()
    model.lin1_dynamics.tau_m.data = torch.rand_like(model.lin1_dynamics.tau_m.data).requires_grad_()
    # add to list of parameters to learn
    params_to_learn.append(model.lin0_dynamics.tau_m)
    params_to_learn.append(model.lin1_dynamics.tau_m)
    # initialize weights to random values
    model.lin0.weight.data = torch.rand_like(model.lin0.weight.data).requires_grad_()
    model.lin1.weight.data = torch.rand_like(model.lin1.weight.data).requires_grad_()
    # add to list of parameters to learn
    params_to_learn.append(model.lin0.weight)
    params_to_learn.append(model.lin1.weight)

    optimizer = torch.optim.Adam(params_to_learn, lr=params['lr'])

    # generate combination of sine waves
    from data.datasets import generate_input
    t, x = generate_input(params['dt'],
                          params['T'],
                          dataset='step function',
                          batch_size=params['batch_size'])

    # smoothen input
    from scipy.ndimage import gaussian_filter1d
    x = torch.tensor(gaussian_filter1d(x.numpy(), sigma=5, axis=1))

    # set τ_m to correct values for plotting rates and errors before training
    # (see first column of Fig. 5b)
    if not params['use_autodiff']:
        τ_0 = model.lin0_dynamics.tau_m.data.clone()
        τ_1 = model.lin1_dynamics.tau_m.data.clone()
        model.lin0_dynamics.tau_m.data = torch.tensor([1.0])
        model.lin1_dynamics.tau_m.data = torch.tensor([2.0])

    # initialize models (without learning)
    for step in range(int(params['T_init'] / params['dt'])):
        model.eval()

        # set τ_m back to randomly initialized values
        if step == int(params['T_init'] / params['dt'] / 2) and not params['use_autodiff']:
            model.lin0_dynamics.tau_m.data = τ_0
            model.lin1_dynamics.tau_m.data = τ_1

        with torch.no_grad():
            data_inputs, data_labels = x[:, step], teacher(x[:, step])

            preds = model(data_inputs, data_labels, beta=params['beta'])
            loss = loss_fn(preds, data_labels.float())

            if log is not None and step % params['logging_interval'] == 0:
                log['time'].append(step * params['dt'])
                log['input'].append(data_inputs[0].item())
                log['target'].append(data_labels[0].item())
                log['output'].append(preds[0].item())
                log['loss'].append(loss.item())
                # log dynamics
                for layer_idx, layer in enumerate(model.children_with_dynamics()):
                    # first sample of the batch
                    log['u_{}'.format(layer_idx)].append(layer.u[0].item())
                    log['v_{}'.format(layer_idx)].append(layer.v[0].item())
                    log['prosp_u_{}'.format(layer_idx)].append(layer.prosp_u[0].item())
                    log['r_{}'.format(layer_idx)].append(layer.r[0].item())
                    log['prosp_e_{}'.format(layer_idx)].append(layer.prosp_v[0].item())
                    # log learned parameters
                    log['w_{}'.format(layer_idx)].append(layer.conn.weight.data.item())
                    log['tau_m_{}'.format(layer_idx)].append(layer.tau_m.data.item())

            if step % 1000 == 0:
                print('init step: {}, loss: {:.4E}'.format(step, loss.item()))

    # detach dynamic variables to start fresh computation graph
    model.detach_dynamic_variables()
    optimizer.zero_grad()
    loss_sum = torch.zeros(1, dtype=torch.float32)

    for step in range(int(params['T_init'] / params['dt']), int((params['T_init'] + params['T_train']) / params['dt'])):
        # Set model to train mode
        model.train()

        with torch.no_grad():
            data_inputs, data_labels = x[:, step], teacher(x[:, step])
            # data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

        preds = model(data_inputs, data_labels, beta=params['beta'])
        loss = loss_fn(preds, data_labels.float())
        loss_sum += loss
        if step % 10000 == 0:
            print('instantaneous loss: {:.4E}'.format(loss.item()))

        # logging during training
        if log is not None and step % params['logging_interval'] == 0:
            log['time'].append(step * params['dt'])
            log['input'].append(data_inputs[0].item())
            log['target'].append(data_labels[0].item())
            log['output'].append(preds[0].item())
            log['loss'].append(loss.item())
            # log dynamics
            for layer_idx, layer in enumerate(model.children_with_dynamics()):
                # first sample of the batch
                log['u_{}'.format(layer_idx)].append(layer.u[0].item())
                log['v_{}'.format(layer_idx)].append(layer.v[0].item())
                log['prosp_u_{}'.format(layer_idx)].append(layer.prosp_u[0].item())
                log['r_{}'.format(layer_idx)].append(layer.r[0].item())
                log['prosp_e_{}'.format(layer_idx)].append(layer.prosp_v[0].item())
                # log learned parameters
                log['w_{}'.format(layer_idx)].append(layer.conn.weight.data.item())
                log['tau_m_{}'.format(layer_idx)].append(layer.tau_m.data.item())

        # update AD model using accumulated loss
        if step % params['truncation_steps'] == 0 and step > params['T_init'] / params['dt'] and params['use_autodiff']:
            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.detach_dynamic_variables()
            loss_sum = 0.

        # update weights only after init_steps
        if not params['use_autodiff'] and step > params['T_init'] / params['dt']:
           optimizer.step()

        with torch.no_grad():
            # clamp dynamic parameters to positive values
            for child in model.children_with_dynamics():
                child.tau_m.data.clamp_(min=10*params['dt'])
                child.tau_r.data.clamp_(min=10*params['dt'])

        # print weight and tau_m values
        if step % 10000 == 0:
            print('train step: {}, loss: {:.4E}'.format(step, loss.item()))
            for layer_idx, layer in enumerate(model.children_with_dynamics()):
                print('layer: {}, weight: {:.4f}, tau_m: {:.4f}'.format(layer_idx, layer.conn.weight.item(), layer.tau_m.item()))

    # dump metrics dictionary to pickle
    if log is not None:
        import pandas as pd
        df = pd.DataFrame(log)
        import pickle
        with open(f'./results/lagline/{args.model}_metrics.pkl', 'wb') as f:
            pickle.dump(df, f)
