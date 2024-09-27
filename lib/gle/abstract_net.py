import torch

from lib.gle.dynamics import GLEDynamics


class GLEAbstractNet:

    def __init__(self, *, full_forward=False, full_backward=False):
        super().__init__()

        self.use_le = True
        self.full_forward = full_forward
        self.full_backward = full_backward
        self.device = torch.device('cpu')

        if self.full_forward or self.full_backward:
            print('Warning: full_forward and full_backward are deprecated and might not work as expected in combination with synaptic filters.')

    def _compute_target_error(self, output, target, beta):
        if target is None:
            return torch.zeros_like(output)
        else:
            return self.compute_target_error(output, target, beta)

    def compute_target_error(self, output, target, beta):
        # this method should be overridden by the user
        # see utils.py for predefined loss functions and corresponding derivatives
        raise NotImplementedError('Please override this method according to the derivative of your loss function w.r.t. the output rate.')

    def _initialize_dynamic_variables(self, shape):
        """Create a dummy input and propagate through all layers to initialize
        dynamic variables to the correct shapes; their initial value
        will be set to zero.

        """
        x = torch.empty(shape, device=self.device).normal_()
        for layer in self.children_with_dynamics():
            x = layer.initialize_dynamic_variables_to_zero(x)

    def detach_dynamic_variables(self):
        for layer in self.children_with_dynamics():
            layer._detach_dynamic_variables()

    def _adjust_batch_dimension(self, batch_size):
        for layer in self.children_with_dynamics():
            layer.adjust_batch_dimension(batch_size)

    def _update_dynamic_variables(self):
        for layer in self.children_with_dynamics():
            layer.update_dynamic_variables()

    def is_not_initialized(self):
        return list(self.children_with_dynamics())[0].is_not_initialized()

    def batch_size_does_not_match(self, batch_size):
        return batch_size != list(self.children_with_dynamics())[0].batch_size()

    def children_with_dynamics(self):
        for value in self.__dict__.values():
            if isinstance(value, GLEDynamics):
                yield value

    def named_children_with_dynamics(self):
        for name, value in self.__dict__.items():
            if isinstance(value, GLEDynamics):
                yield name, value

    def forward(self, x, target=None, *, beta=0.0):

        if self.is_not_initialized():
            self._initialize_dynamic_variables(x.shape)

        if self.batch_size_does_not_match(len(x)):
            self._adjust_batch_dimension(len(x))

        layers = list(self.children_with_dynamics())
        for layer_idx in range(len(layers)):
            if layer_idx == 0:
                if len(layers) > 1:
                    layers[layer_idx](x, layers[layer_idx + 1].e_bottom)
                else:
                    layers[layer_idx](x, self._compute_target_error(layers[layer_idx].r, target, beta))
            elif layer_idx > 0 and layer_idx < len(layers) - 1:
                layers[layer_idx](layers[layer_idx - 1].r, layers[layer_idx + 1].e_bottom)
            elif layer_idx == len(layers) - 1:
                layers[layer_idx](layers[layer_idx - 1].r, self._compute_target_error(layers[layer_idx].r, target, beta))
            else:
                assert False  # should never be reached

            if self.full_forward:
                layers[layer_idx].update_dynamic_variables()

        output = layers[-1].r.clone()

        self._update_dynamic_variables()

        if self.full_backward:
            assert self.full_forward
            layers = list(self.children_with_dynamics())
            for layer_idx in range(len(layers) - 1, -1, -1):
                if layer_idx == 0:
                    layers[layer_idx](x, layers[layer_idx + 1].e_bottom)
                elif layer_idx > 0 and layer_idx < len(layers) - 1:
                    layers[layer_idx](layers[layer_idx - 1].r, layers[layer_idx + 1].e_bottom)
                elif layer_idx == len(layers) - 1:
                    layers[layer_idx](layers[layer_idx - 1].r, self._compute_target_error(layers[layer_idx].r, target, beta))
                else:
                    assert False  # should never be reached

                layers[layer_idx].update_dynamic_variables()

        return output

    def _apply(self, fn):
        super()._apply(fn)
        for layer in self.children_with_dynamics():
            layer._apply(fn)

        self.device = layer.device
        return self

    def named_dynamic_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        for name, module in self.named_children_with_dynamics():
            if remove_duplicate and module not in memo:
                memo.add(module)
                yield prefix + ('.' if prefix else '') + name, module
            else:
                yield prefix + ('.' if prefix else '') + name, module

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        for name, module in super().named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate):
            yield name, module
        for name, module in self.named_dynamic_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate):
            yield name, module

    # extend state_dict to include dynamic variables
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for name, module in self.named_dynamic_modules(prefix=prefix):
            state.update(module.state_dict(destination=destination, prefix=name, keep_vars=keep_vars))
        return state

    def load_dynamic_state_dict(self, state_dict, strict=False):
        for name, module in self.named_dynamic_modules():
            # this is a hack to remove the module name from the state_dict
            module.load_state_dict({k[len(name) + 1:]: v for k, v in state_dict.items() if k.startswith(name)}, strict=strict)

    def load_state_dict(self, state_dict, strict=False):
        # load static dict first but skip dynamic variables
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        # load unexpected keys into dynamic modules
        self.load_dynamic_state_dict({k: v for k, v in state_dict.items() if k in unexpected_keys}, strict=strict)
