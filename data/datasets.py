import torch
from torch.utils.data import TensorDataset


def get_mnist1d_splits(classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], padding=[36, 60], final_seq_length=40, scale_coeff=0.4, max_translation=48, iid_noise_scale=2e-2, corr_noise_scale=0.25, shear_scale=0.75):

    # using the pip package mnist1d
    from mnist1d.data import get_dataset, get_dataset_args
    
    args = get_dataset_args(as_dict=False)
    args.classes = classes  # which classes to include
    args.padding = padding  # padding for each pattern
    args.final_seq_length = final_seq_length  # length of single MNIST1D sample
    args.scale_coeff = scale_coeff  # scale of each pattern
    args.iid_noise_scale = iid_noise_scale  # scale of iid noise
    args.max_translation = max_translation  # max translation of each pattern
    args.corr_noise_scale = corr_noise_scale  # scale of correlated noise
    args.shear_scale = shear_scale  # scale of shearing
    dataset = get_dataset(args, path='./data/mnist1d_data_{}.pkl'.format(final_seq_length), download=False, regenerate=False)
    x_train, x_test = torch.Tensor(dataset['x']), torch.Tensor(dataset['x_test'])
    y_train, y_test = torch.LongTensor(dataset['y']), torch.LongTensor(dataset['y_test'])

    train_data = TensorDataset(x_train, y_train)

    test_data = TensorDataset(x_test, y_test)

    return train_data, test_data

def generate_input(dt, T, batch_size=1, dataset='Ornstein-Uhlenbeck', n_inputs=1,
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
    n_steps = int(T / dt)
    t = torch.linspace(0, T, n_steps)
    x = torch.zeros(batch_size, n_steps, 2)

    if dataset == 'Ornstein-Uhlenbeck':
        # generate ornstein-uhlenbeck process
        # adapted from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
        σ = 0.5
        μ = 0.0
        τ = 1.0
        sigma_bis = σ * math.sqrt(2. / τ)
        sqrtdt = math.sqrt(dt)

        for i in range(n_steps - 1):
            x[:, i + 1, :] = x[:, i, :] + dt * (-(x[:, i, :] - μ) / τ) \
                + sigma_bis * sqrtdt * torch.randn(batch_size, 2)

    elif dataset == 'step function':
        # generate sequence of 0 & 1
        x[:, :, 0] = torch.zeros_like(t) - 1
        width = int(2 / dt)
        for i in torch.linspace(0, n_steps, int(n_steps / width / 2) + 1):
            x[:, int(i):int(i)+width, 0] = torch.ones_like(x[:, int(i):int(i)+width, 0])

        # x[:, :, 1] = torch.roll(x[:, :, 0], int(width / 2), dims=1)
        x[:, :, 1] = torch.repeat_interleave(x[:, :, 0], 2, dim=1)[0, :n_steps]

        # shift each sample in x by a random amount
        for i in range(batch_size):
            shift = torch.randint(0, width, (1,)).item()
            x[i, :, 0] = torch.roll(x[i, :, 0], shift, dims=0)
            x[i, :, 1] = torch.roll(x[i, :, 1], shift, dims=0)

    elif dataset == 'sine & cosine':
        # generate sine wave to check Fourier analysis
        for freq_sine in extra_input_params['sine_freq']:
            x[:, :, 0] += torch.sin(2 * math.pi * freq_sine * t.clone().detach().requires_grad_(False))
        for freq_cosine in extra_input_params['cosine_freq']:
            x[:, :, 1] += torch.cos(2 * math.pi * freq_cosine * t.clone().detach().requires_grad_(False))
    else:
        assert dataset == 'Ornstein-Uhlenbeck', 'Unknown dataset'

    if n_inputs == 1:
        # only use first input
        x = x[:, :, 0].unsqueeze(2)

    if extra_input_params.get('scale', False):
        max_amplitud = torch.max(torch.abs(x))
        x /= max_amplitud
    if extra_input_params.get('scale_factor', False):
        x *= extra_input_params['scale_factor']

    return t, x
