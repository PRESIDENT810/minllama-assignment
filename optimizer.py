from typing import Callable, Iterable, Tuple

from sympy import beta
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                lr = group["lr"]

                # Update first and second moments of the gradients
                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                next_m = beta1 * m + (1 - beta1) * grad
                next_v = beta2 * v + (1 - beta2) * grad ** 2

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                cached_beta1 = state.get("cached_beta1", 1)
                cached_beta2 = state.get("cached_beta2", 1)
                cached_beta1 *= beta1
                cached_beta2 *= beta2

                next_m_hat = next_m / (1 - cached_beta1)
                next_v_hat = next_v / (1 - cached_beta2)

                # Update parameters

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

                next_p = p - lr * group["weight_decay"] * p  # Apply weight decay first
                next_p = next_p - lr * next_m_hat / (next_v_hat.sqrt() + group["eps"])

                state["m"] = next_m
                state["v"] = next_v
                state["cached_beta1"] = cached_beta1
                state["cached_beta2"] = cached_beta2
                p.data = next_p

        return loss