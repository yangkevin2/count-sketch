import torch
from torch.optim import Optimizer
from cms import CountMinSketch

import numpy as np

class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'sum' not in state and 'step' not in state:
                    state['step'] = 0
                    if grad.is_sparse:
                        state['sum'] = [torch.zeros(p.shape[i]).to(p.device) for i in range(len(p.shape))]
                        state['real_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['sum'] = torch.full_like(grad.data, group['initial_accumulator_value'])


                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad_dense = grad.to_dense()
                    grad = grad.coalesce()
                    nonzero_idx = grad._indices()[0]
                    grad = grad._values()

                    sum_sq = state['sum'][0][nonzero_idx]
                    for i in range(1, len(p.shape)):
                        sum_sq = torch.min(sum_sq.unsqueeze(i), state['sum'][i].view([1 for _ in range(i)] + [-1]))
                    sum_sq.addcmul_(grad, grad, value=1)

                    for i in range(len(p.shape)): # TODO optimize later if needed
                        sum_sq_view = sum_sq
                        if i != 0:
                            sum_sq_view = sum_sq_view.transpose(0, i)
                        if len(p.shape) > 1:
                            sum_sq_view = sum_sq_view.flatten(1)
                            sum_sq_view_max = sum_sq_view.max(dim=1)[0]
                        else:
                            sum_sq_view_max = sum_sq_view
                        if i == 0:
                            state['sum'][i][nonzero_idx] = sum_sq_view_max.clone()
                        else:
                            state['sum'][i] = torch.max(state['sum'][i], sum_sq_view_max.clone())

                    state['real_sum'].addcmul_(grad_dense, grad_dense, value=1)
                    
                    full_sum_sq = state['sum'][0]
                    for i in range(1, len(p.shape)):
                        full_sum_sq = torch.min(full_sum_sq.unsqueeze(i), state['sum'][i].view([1 for _ in range(i)] + [-1]))
                    # sum_sq.addcmul_(grad, grad, value=1)
                    if not ((full_sum_sq - state['real_sum']) >= -1e-8).sum() == np.prod(full_sum_sq.shape):
                        import pdb; pdb.set_trace()
                    assert ((full_sum_sq - state['real_sum']) >= -1e-8).sum() == np.prod(full_sum_sq.shape)

                    std = sum_sq.sqrt().add_(1e-10)
                    p.data[nonzero_idx] = p.data[nonzero_idx].addcdiv_(-clr, grad, std)
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)
        return loss
