import torch
import re
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    
    Reference:
    "Large Batch Training of Convolutional Networks" by Y. You, I. Gitman, and B. Ginsburg.
    https://arxiv.org/abs/1708.03888
    
    Parameters:
    - params: Iterable of parameters to optimize
    - lr (float): Learning rate
    - momentum (float): Momentum factor
    - use_nesterov (bool): Whether to use Nesterov momentum
    - weight_decay (float): Weight decay coefficient
    - exclude_from_weight_decay (list): Parameter names to exclude from weight decay
    - exclude_from_layer_adaptation (list): Parameter names to exclude from layer adaptation
    - classic_momentum (bool): Whether to use classic momentum
    - eeta (float): Trust ratio coefficient
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=1e-6,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=0.001,
    ):
        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        
        # Set exclude_from_layer_adaptation to exclude_from_weight_decay if None
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # Apply weight decay
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # Compute trust ratio for layer adaptation
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.device
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    
                    # Initialize momentum buffer
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for given parameter."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for given parameter."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True