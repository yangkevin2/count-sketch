import torch
from torch.optim import Optimizer
from cms import CountMinSketch

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
                    # if grad.is_sparse:
                    #     N, D = grad.data.size()
                    #     state['sum'] = CountMinSketch(N, D)
                    # else:
                    #     state['sum'] = torch.full_like(grad.data, group['initial_accumulator_value'])
                    state['sum'] = [torch.zeros(p.shape[i]).to(p.device) for i in range(len(p.shape))]
                    state['real_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                for i in range(len(p.shape)): # TODO optimize later if needed
                    grad_view = grad
                    if i != 0:
                        grad_view = grad.transpose(0, i)
                    if len(p.shape) > 1:
                        grad_view = grad_view.flatten(1)
                        grad_view_max = grad_view.max(dim=1)[0]
                        grad_view_min = grad_view.min(dim=1)[0]
                    else:
                        grad_view_max = grad_view
                        grad_view_min = grad_view
                    grad_view_sq = torch.max(grad_view_max * grad_view_max, grad_view_min * grad_view_min)
                    state['sum'][i].add_(grad_view_sq, alpha=1)
                
                sum_sq = state['sum'][0]
                for i in range(1, len(p.shape)):
                    sum_sq = torch.min(sum_sq.unsqueeze(i), state['sum'][i].view(tuple([1 for _ in range(i)] + [-1])))

                state['real_sum'].addcmul_(1, grad, grad)
                # import pdb; pdb.set_trace()

                # if grad.is_sparse:
                #     grad = grad.coalesce()  # the update is non-linear so indices must be unique
                #     grad_indices = grad._indices()
                #     grad_values = grad._values()
                #     size = grad.size()

                #     def make_sparse(values):
                #         constructor = grad.new
                #         if grad_indices.dim() == 0 or values.dim() == 0:
                #             return constructor().resize_as_(grad)
                #         return constructor(grad_indices, values, size)

                #     std = state['sum'].update(grad_indices, grad_values.pow(2), size)
                #     std_values = std._values().sqrt_().add_(1e-10)
                #     update = grad_values / std_values
                #     p.data.add_(make_sparse(-clr * update))
                # else:
                # state['sum'].addcmul_(1, grad, grad)
                std = sum_sq.sqrt().add_(1e-10)
                p.data.addcdiv_(-clr, grad, std)
        return loss
