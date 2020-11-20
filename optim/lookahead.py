# from https://github.com/mgrankin/over9000/blob/master/lookahead.py
# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

import itertools as it
from torch.optim import Optimizer, Adam


# TODO: This code is buggy. It did not support state saving support
#       I did some ad-hoc fixes to get it working but noone should
#       use if for training on condor etc.
#       Sadly this eliminates over9000 + ranger from options to train
#       on servers.
class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")

        self.optimizer = base_optimizer
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults

        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]] for group in self.param_groups
        ]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group["step_counter"] += 1
            if group["step_counter"] % self.k != 0:
                continue
            for p, q in zip(group["params"], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss


def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
    adam = Adam(params, *args, **kwargs)
    return Lookahead(adam, alpha, k)
