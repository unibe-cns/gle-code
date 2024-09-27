#!/usr/bin/env python3

import os
import pickle
import json

import torch
import torch.nn.functional as F
import numpy as np


try:
    from pydenticon import Generator
    import hashlib
    show_matrix_identicons = True
except ImportError:
    print('pydenticon not installed, skipping identicons of weight matrices to visualize equal inits')
    show_matrix_identicons = False

import numpy as np
def print_asciidenticon(np_array):
    data = np.array2string(np_array)

    # Instantiate a generator.
    generator = Generator(4, 4, digest=hashlib.sha256)

    # Call the method and get the results.
    asciidenticon = generator.generate(data, 4, 4, output_format="ascii").replace('+', 'x').replace('-', ' ')
    print(asciidenticon)

def print_model_asciidenticons(model):
    for name, param in model.named_parameters():
        print(f"{name}, shape={param.data.shape}")
        print("------")
        print_asciidenticon(param.data.cpu().numpy())
        print('')

def get_loss_and_derivative(loss, output_phi=None):
    assert type(loss) == str
    # define loss function and derivative
    if loss == 'mse':
        def loss_fn(output, target, *args, **kwargs):
            # one-hot encode target if necessary
            if output.shape[-1] != target.shape[-1]:
                target = F.one_hot(target, output.shape[-1]).float()
            return F.mse_loss(target, output, *args, **kwargs)
        def loss_fn_deriv(self, output, target, beta):
            # one-hot encode target if necessary
            if output.shape[-1] != target.shape[-1]:
                target = F.one_hot(target, output.shape[-1])
            e = target - output
            return beta * e
    elif loss == 'ce':
        assert output_phi == None or 'linear', "CE loss can only be used with linear output layer as it expects logits"
        def loss_fn(output, target, *args, **kwargs):
            # CE loss takes logits and target indices as input
            return F.cross_entropy(output, target, *args, **kwargs)
        def loss_fn_deriv(self, output, target, beta):
            # convert logits to probabilities
            e = F.one_hot(target, output.shape[-1]) - F.softmax(output, dim=1)
            return beta * e
    elif loss == 'nll':
        assert output_phi == 'logsoftmax', "NLL loss can only be used with logsoftmax activation in the output layer"
        def loss_fn(output, target, *args, **kwargs):
            # NLL loss takes log probabilities and target class as input
            return F.nll_loss(output, target, *args, **kwargs)
        def loss_fn_deriv(self, output, target, beta):
            # convert log probabilities to probabilities
            e = F.one_hot(target, output.shape[-1]) - torch.exp(output)
            return beta * e
    else:
        raise NotImplementedError(f"Using {loss} loss  with {output_phi} nonlinearity not implemented")

    return loss_fn, loss_fn_deriv


def check_if_path_to_results_exist_and_create_it_if_it_doesnt(relative_path: str):
    """
    Check whether a relative path exist and creates it in case it's not there yet

    Parameters
    -----------
    relative_path : string
    """
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

def save_dict(dict2save : dict, fname: str, path_to_file : str):
    """
    Save dictionaris to files (useful for not loosing all data from experiments
    Parameters
    -----------
    dict2save : dict
        dict to be saved
    fname : str
        name to the saved file
    """
    f_name_w_extension = path_to_file + fname + ".pkl"
    with open(f_name_w_extension, 'wb') as f:
        pickle.dump(dict2save, f)

def write_to_file(*str_args, path_to_file : str, fname: str = 'initial_and_final_state.txt', verbose : bool = False):
    check_if_path_to_results_exist_and_create_it_if_it_doesnt(path_to_file)
    with open(path_to_file + fname, 'a') as f:
        for element in str_args:
            f.write(str(element))
            f.write('\n')
            if verbose:
                print(element)

def check_granularity_is_sufficient(required_steps_in_period: float = 40,
                                  params: dict = {}, extra_input_params: dict = {}):
    """
    Check wheter granularity is enough for not having numerical problems
    """
    if params["dataset"] != "sine & cosine":
        pass
    elif params["dataset"] == "sine & cosine":
        if params['n_inputs'] == 1:
            max_freq = max(extra_input_params["sine_freq"]) # Hz
        elif params['n_inputs'] == 2:
            max_freq = max(extra_input_params["sine_freq"] + extra_input_params["cosine_freq"]) # Hz
        else:
            raise ValueError("n_inputs should be less or equalt than 2")
        shortest_period = 1 / max_freq
        n_steps_in_period = shortest_period / params["dt"]
        print(f"dt's per period = {n_steps_in_period}")
        if n_steps_in_period < required_steps_in_period:
            raise Exception(f"We need 'at least' {required_steps_in_period} steps per period, "
                            f"but now we are using only {n_steps_in_period}. dt should be {n_steps_in_period/required_steps_in_period} times the current value")
    return max_freq

def init_weights_log_dict(t_shape, n_inputs, n_hid_1, n_hid_2, n_top,
                          is_teacher: bool = False):
    """
    initialize a dictionary for logging weights during training
    Parameters
    ----------
    t_shape :
    n_inputs: int
        number of neurons in input layer
    n_hid_1 : int
        number of neurons in hidden layer 1
    n_hid_2 : int
        number of neurons in hidden layer 2
    n_top : int
        number of neurons in top layer
    is_teacher : bool
        different initialization if we are initializing the teacher network
    """
    if is_teacher == True:
        log_weights_dict = {'hid_1': torch.zeros((n_hid_1, n_inputs)),
                            'hid_2': torch.zeros((n_hid_2, n_hid_1)),
                            'top': torch.zeros((n_hid_2))
                            }
    else:
        log_weights_dict = {'hid_1': torch.zeros((t_shape, n_hid_1, n_inputs)),
                            'hid_2': torch.zeros((t_shape, n_hid_2, n_hid_1)),
                            'top': torch.zeros((t_shape, n_hid_2))
                            }
    return log_weights_dict

def init_errors_log_dict(time_array, n_hid_1, n_hid_2, is_teacher=False):
    """
    initialize a dictionary for logging errors during training
    Parameters
    ----------
    time_array : torch.array
    n_hid_1 : int
        number of neurons in hidden layer 1
    n_hid_2 : int
        number of neurons in hidden layer 2
    """
    t_shape = time_array.shape[0]
    log_errors_dict = {'hid_1': torch.zeros((t_shape, n_hid_1)),
                       'hid_2': torch.zeros((t_shape, n_hid_2)),
                       'top': torch.zeros_like(time_array)
                       }
    if is_teacher:
        for layer_name in log_errors_dict:
            log_errors_dict[layer_name][:] = np.nan


    return log_errors_dict


def get_phi_and_derivative(phi):
    if phi == 'relu':
        phi_fn = torch.nn.ReLU()
        phi_fn_deriv = lambda x: (x > 0).to(x)
    elif phi == 'sigmoid':
        phi_fn = torch.nn.Sigmoid()
        phi_fn_deriv = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
    elif phi == 'hard_sigmoid':
        phi_fn = lambda x: x.clamp(0, 1)
        phi_fn_deriv = lambda x: (x >= 0) * (x <= 1)
    elif phi == 'hard_sigmoid_torch':
        phi_fn = torch.nn.Hardsigmoid()
        phi_fn_deriv = lambda x: (x >= -1) * (x <= 1)
    elif phi == 'tanh':
        phi_fn = torch.nn.Tanh()
        phi_fn_deriv = lambda x: 1 - torch.tanh(x) ** 2
    elif phi == 'hard_tanh':
        phi_fn = torch.nn.Hardtanh()
        phi_fn_deriv = lambda x: (x >= -1) * (x <= 1)
    elif phi == 'softplus':
        phi_fn = torch.nn.Softplus()
        phi_fn_deriv = lambda x: torch.sigmoid(x)
    elif phi == 'linear':
        phi_fn = torch.nn.Identity()
        phi_fn_deriv = lambda x: torch.ones_like(x)
    # the Jacobian matrices for softmax and logsoftmax are not diagonal and
    # therefore cannot be used in the same way as the other activation functions
    elif phi == 'softmax':
        phi_fn = torch.nn.Softmax(dim=1)
        phi_fn_deriv = None
    elif phi == 'logsoftmax':
        phi_fn = torch.nn.LogSoftmax(dim=1)
        phi_fn_deriv = None
    else:
        raise NotImplementedError(f"phi={phi} not implemented")

    return phi_fn, phi_fn_deriv

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    See https://github.com/pytorch/pytorch/issues/61292

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

def connect_to_neptune(mode, project):
    # check if mode is valid
    if mode in ['async', 'sync', 'debug', 'offline', 'read-only']:
        try:
            import neptune
            with open('.neptune_id.json', 'r+') as file:
                    id_data = json.load(file)
                    workspace = id_data["workspace"]
                    api_token = id_data["api_token"]
                    run = neptune.init_run(
                        mode=mode,
                        project=f"{workspace}/{project}",
                        # capture_hardware_metrics=False,
                        # capture_stdout=False,
                        # capture_stderr=False,
                        # https://docs.neptune.ai/logging/system_metrics/
                    )
                    print(f"Starting {mode} {project} experiment in workspace {workspace}...")
        except FileNotFoundError:
            print(f"Neptune API token not found!")
            print(f"Starting experiment without observer...")
            run = None
        except ModuleNotFoundError:
            print(f"Neptune module not found!")
            print(f"Starting experiment without observer...")
            run = None
        except Exception as e:
            print(f"Error: {e}")
            run = None
    # custom mode similar to debug but without loading neptune
    elif mode == 'disabled':
        run = None
    else:
        raise ValueError(f"Neptune mode={mode} not implemented")
    return run
