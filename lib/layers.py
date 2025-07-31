import torch


class GLELinear(torch.nn.Linear):

    def compute_error(self, e):
        return torch.mm(e, self.weight)

    def compute_grad(self, r_bottom, e):
        self.weight.grad = - torch.bmm(e.unsqueeze(2), r_bottom.unsqueeze(1)).mean(0)
        if self.bias is not None:
            self.bias.grad = - e.mean(0)
