import torch
from collections import OrderedDict

class GLEDynamics():

    def __init__(self, conn, *, tau_m, tau_r=None, dt=None, phi=None,
                 phi_prime=None, prospective_errors=False, use_autodiff=False,
                 learn_tau=False, gamma=0.0):
        super().__init__()

        self.dt = dt
        self.conn = conn
        self.learn_tau = learn_tau

        if tau_m is not None:
            if isinstance(tau_m, torch.Tensor):
                self.tau_m = tau_m.clone()
            else:
                self.tau_m = torch.tensor(tau_m)
        else:
            raise ValueError("τ_m must be specified")

        if tau_r is not None:
            if isinstance(tau_r, torch.Tensor):
                self.tau_r = tau_r.clone()
            else:
                self.tau_r = torch.tensor(tau_r)
        elif self.tau_m is not None:
            # default to τ_m if τ_r is not specified (LE)
            self.tau_r = self.tau_m.clone()
        else:
            raise ValueError("τ_r must be specified")

        # promote to learnable parameters if requested
        if self.learn_tau:
            self.tau_m = torch.nn.Parameter(self.tau_m)

        assert torch.all(self.tau_m >= self.dt), "τ_m must be greater than or equal to dt"
        assert torch.all(self.tau_r >= self.dt), "τ_r must be greater than or equal to dt"

        self.phi = phi
        self.phi_prime = phi_prime
        self.prospective_errors = prospective_errors
        self.use_autodiff = use_autodiff
        self.gamma = gamma

        if self.tau_m is not None:
            assert self.dt is not None

        if self.phi is not None:
            assert self.phi_prime is not None
        else:
            self.phi = lambda x: x
            self.phi_prime = lambda x: torch.ones_like(x)

        # collect learnable parameters
        self._parameters = {}
        for var in vars(self):
            if isinstance(getattr(self, var), torch.nn.Parameter):
                self._parameters[var] = getattr(self, var)

        # WARNING: these will be populated on first call to `forward`
        self.u = None
        self.v = None
        self.r = None
        self.r_prime = None
        self.e_bottom = None

        # WARNING: these will be populated on first call to `forward`
        self.next_u = None
        self.next_v = None
        self.next_r = None
        self.next_r_prime = None
        self.next_e_bottom = None

        self.device = torch.device("cpu")

    def adjust_batch_dimension(self, batch_size):
        assert len(self.u) == len(self.v)
        assert len(self.u) == len(self.r)
        assert len(self.u) == len(self.r_prime)
        assert len(self.u) == len(self.e_bottom)

        if self.batch_size() < batch_size:
            u = self.u.clone()
            v = self.v.clone()
            r = self.r.clone()
            r_prime = self.r_prime.clone()
            e_bottom = self.e_bottom.clone()
            while len(u) < batch_size:
                u = torch.cat([u, self.u.clone()])
                v = torch.cat([v, self.v.clone()])
                r = torch.cat([r, self.r.clone()])
                r_prime = torch.cat([r_prime, self.r_prime.clone()])
                e_bottom = torch.cat([e_bottom, self.e_bottom.clone()])
            self.u = u[:batch_size].clone()
            self.v = v[:batch_size].clone()
            self.r = r[:batch_size].clone()
            self.r_prime = r_prime[:batch_size].clone()
            self.e_bottom = e_bottom[:batch_size].clone()
        else:
            self.u = self.u.narrow(0, 0, batch_size)
            self.v = self.v.narrow(0, 0, batch_size)
            self.r = self.r.narrow(0, 0, batch_size)
            self.r_prime = self.r_prime.narrow(0, 0, batch_size)
            self.e_bottom = self.e_bottom.narrow(0, 0, batch_size)

    def batch_size(self):
        return len(self.u)

    # compute grad for time constants
    def compute_tau_grad(self, e, u_dot):
        if self.learn_tau:
            self.tau_m.grad = (e * u_dot).mean(0)

    def is_not_initialized(self):
        return self.u is None

    def initialize_dynamic_variables_to_zero(self, r_bottom):
        assert self.u is None
        assert self.v is None
        assert self.r is None
        assert self.r_prime is None
        assert self.e_bottom is None

        inst_s = self.conn(r_bottom)

        self.u = torch.zeros(inst_s.shape, device=self.device)
        self.v = torch.zeros(inst_s.shape, device=self.device)
        self.r = torch.zeros(inst_s.shape, device=self.device)
        self.r_prime = torch.zeros(inst_s.shape, device=self.device)
        self.e_bottom = torch.zeros(r_bottom.shape, device=self.device)
        return inst_s

    def _detach_dynamic_variables(self):
        self.u = self.u.detach().clone()
        self.v = self.v.detach().clone()
        self.r = self.r.detach().clone()
        self.r_prime = self.r_prime.detach().clone()
        self.e_bottom = self.e_bottom.detach().clone()

    def update_dynamic_variables(self):
        if self.use_autodiff:  # retain graph
            self.u = self.next_u.clone()
            self.v = self.next_v.clone()
            self.r = self.next_r.clone()
            self.r_prime = self.next_r_prime.clone()
            self.e_bottom = self.next_e_bottom.clone()
        else:
            self.u = self.next_u.detach().clone()
            self.v = self.next_v.detach().clone()
            self.r = self.next_r.detach().clone()
            self.r_prime = self.next_r_prime.detach().clone()
            self.e_bottom = self.next_e_bottom.detach().clone()

    def named_dynamic_parameters(self, prefix='', recurse=True):
        for name, param in self._parameters.items():
            yield prefix + '.' + name, param

    def dynamic_parameters(self, recurse=True):
        for name, param in self._parameters.items():
            yield param

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_dynamic_parameters(prefix=prefix, recurse=True):
            destination[name] = param if keep_vars else param.detach()
        return destination

    def load_state_dict(self, state_dict, strict=False):
        for name, param in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data.copy_(param)
            else:
                assert not strict, "unexpected key '{}' in state_dict".format(name)

    def __call__(self, r_bottom, inst_e):
        """
        WARNING: inst_e is the backpropagated error from the layer above
        e_l = W^T e_l+1 which still needs to be scaled by the derivative
        of the activation function of the current layer.

        """

        # compute instantaneous error and scale inst_e
        # by derivative of activation function
        inst_e *= self.r_prime

        # feedforward input signal
        inst_s = self.conn(r_bottom)

        # prospective errors
        if self.prospective_errors and self.tau_r is not None:
            v_dot = (inst_e - self.v) / self.tau_r
            v = self.v + self.dt * v_dot
            prosp_v = self.v + self.tau_m * v_dot
        else:
            # overwrite prospective error with instantaneous one
            v_dot = torch.zeros_like(inst_e)
            v = inst_e.clone()
            prosp_v = inst_e.clone()

        # membrane integration
        if self.tau_m is not None:
            u_dot = (inst_s + self.gamma * prosp_v - self.u) / self.tau_m
            u = self.u + self.dt * u_dot
            prosp_u = self.u + self.tau_r * u_dot
        else:
            u = inst_s + self.gamma * prosp_v
            prosp_u = inst_s + self.gamma * prosp_v

        # populate gradients for this layer
        if not self.use_autodiff:
            # compute grad for this layer (W & b only)
            self.conn.compute_grad(r_bottom, prosp_v)
            # compute grad for dynamic variables
            self.compute_tau_grad(prosp_v, u_dot)

        # backpropagate errors to layer below
        e_bottom = self.conn.compute_error(prosp_v)

        # for logging only
        self.prosp_u = prosp_u.clone().detach()
        self.prosp_v = prosp_v.clone().detach()
        self.inst_e = inst_e.clone().detach()

        # compute firing rate and derivative
        r, r_prime = self.phi(prosp_u), self.phi_prime(prosp_u)

        # update dynamic variables
        self.next_u, self.next_v = u, v
        self.next_r, self.next_r_prime = r, r_prime
        self.next_e_bottom = e_bottom

        return r, e_bottom

    def _apply(self, fn):
        self.tau_m = fn(self.tau_m)
        self.tau_r = fn(self.tau_r)
        self.device = self.tau_r.device

        self.u = fn(self.u) if self.u is not None else None
        self.v = fn(self.v) if self.v is not None else None
        self.r = fn(self.r) if self.r is not None else None
        self.r_prime = fn(self.r_prime) if self.r_prime is not None else None
        self.e_bottom = fn(self.e_bottom) if self.e_bottom is not None else None

        self.next_u = fn(self.next_u) if self.next_u is not None else None
        self.next_v = fn(self.next_v) if self.next_v is not None else None
        self.next_r = fn(self.next_r) if self.next_r is not None else None
        self.next_r_prime = fn(self.next_r_prime) if self.next_r_prime is not None else None
        self.next_e_bottom = fn(self.next_e_bottom) if self.next_e_bottom is not None else None

        return self

    def to(self, *args, **kwargs):
        self.tau_m = self.tau_m.to(*args, **kwargs)
        self.tau_r = self.tau_r.to(*args, **kwargs)
        self.device = self.tau_r.device

        self.u = self.u.to(*args, **kwargs) if self.u is not None else None
        self.v = self.v.to(*args, **kwargs) if self.v is not None else None
        self.r = self.r.to(*args, **kwargs) if self.r is not None else None
        self.r_prime = self.r_prime.to(*args, **kwargs) if self.r_prime is not None else None
        self.e_bottom = self.e_bottom.to(*args, **kwargs) if self.e_bottom is not None else None

        self.next_u = self.next_u.to(*args, **kwargs) if self.next_u is not None else None
        self.next_v = self.next_v.to(*args, **kwargs) if self.next_v is not None else None
        self.next_r = self.next_r.to(*args, **kwargs) if self.next_r is not None else None
        self.next_r_prime = self.next_r_prime.to(*args, **kwargs) if self.next_r_prime is not None else None
        self.next_e_bottom = self.next_e_bottom.to(*args, **kwargs) if self.next_e_bottom is not None else None
        return self
