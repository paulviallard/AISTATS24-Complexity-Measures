import math
import torch
from torch.optim import Optimizer

###############################################################################


class SGLD(Optimizer):

    def __init__(
        self, params, weight_decay=0, lr=0.1, alpha=1.0
    ):
        defaults = dict(weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

        assert weight_decay >= 0.0
        assert lr >= 0.0
        self.weight_decay = weight_decay
        self.__lr = lr
        self.__alpha = alpha

        for group in self.param_groups:
            if("lr" not in group):
                group["lr"] = self.__lr
            lr = group["lr"]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # We update all the parameters with the SGLD algorithm
        for group in self.param_groups:

            lr = group["lr"]
            for w in group["params"]:

                # We get the gradient
                grad = w.grad

                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "SGLD does not support sparse gradients")

                # We add the weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(w, alpha=group['weight_decay'])

                # We update the weights
                w.data = (
                    w.data - lr*grad
                    + math.sqrt(2.0*lr/self.__alpha)*torch.randn(
                        grad.shape, device=grad.device)
                )

        return loss

###############################################################################
