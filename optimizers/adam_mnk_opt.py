import math
import torch
from torch.optim import Optimizer

import numpy as np

from exp_cms import CountMinSketch
from exp_sketch import CountSketch 

class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def dense(self, p, grad, group):
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # init exp_avg and exp_avg_sq as sketches
            state['exp_avg'] = [(torch.zeros(p.shape[i]).to(p.device), torch.zeros(p.shape[i]).to(p.device)) for i in range(len(p.shape))]
            state['exp_avg_sq'] = [torch.zeros(p.shape[i]).to(p.device) for i in range(len(p.shape))]
            # Exponential moving average of gradient values
            # state['real_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            # state['real_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        # if amsgrad:
        #     max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(p, alpha=group['weight_decay'])

        exp_avg_ub = state['exp_avg'][0][0]
        exp_avg_lb = state['exp_avg'][0][1]
        exp_avg_sq = state['exp_avg_sq'][0]
        for i in range(1, len(p.shape)):
            exp_avg_ub = torch.min(exp_avg_ub.unsqueeze(i), state['exp_avg'][i][0].view(tuple([1 for _ in range(i)] + [-1])))
            exp_avg_lb = torch.max(exp_avg_lb.unsqueeze(i), state['exp_avg'][i][1].view(tuple([1 for _ in range(i)] + [-1])))
            exp_avg_sq = torch.min(exp_avg_sq.unsqueeze(i), state['exp_avg_sq'][i].view(tuple([1 for _ in range(i)] + [-1])))
        exp_avg_ub.mul_(beta1).add_(grad, alpha=1-beta1)
        exp_avg_lb.mul_(beta1).add_(grad, alpha=1-beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

        for i in range(len(p.shape)):
            exp_avg_ub_view = exp_avg_ub
            exp_avg_lb_view = exp_avg_lb
            if i != 0:
                exp_avg_ub_view = exp_avg_ub_view.transpose(0, i)
                exp_avg_lb_view = exp_avg_lb_view.transpose(0, i)
            if len(p.shape) > 1:
                exp_avg_ub_view = exp_avg_ub_view.flatten(1)
                exp_avg_ub_view_max = exp_avg_ub_view.max(dim=1)[0]
                exp_avg_lb_view = exp_avg_lb_view.flatten(1)
                exp_avg_lb_view_max = exp_avg_lb_view.min(dim=1)[0]
            else:
                exp_avg_ub_view_max = exp_avg_ub_view
                exp_avg_lb_view_max = exp_avg_lb_view
            new_ub = exp_avg_ub_view_max.clone()
            new_lb = exp_avg_lb_view_max.clone()
            state['exp_avg'][i] = (new_ub, new_lb)

        for i in range(len(p.shape)):
            exp_avg_sq_view = exp_avg_sq
            if i != 0:
                exp_avg_sq_view = exp_avg_sq_view.transpose(0, i)
            if len(p.shape) > 1:
                exp_avg_sq_view = exp_avg_sq_view.flatten(1)
                exp_avg_sq_view_max = exp_avg_sq_view.max(dim=1)[0]
            else:
                exp_avg_sq_view_max = exp_avg_sq_view
            state['exp_avg_sq'][i] = exp_avg_sq_view_max.clone()

        # Decay the first and second moment running average coefficient
        # real_exp_avg, real_exp_avg_sq = state['real_exp_avg'], state['real_exp_avg_sq']
        # real_exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # real_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        # if amsgrad:
        #     # Maintains the maximum of all 2nd moment running avg. till now
        #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        #     # Use the max. for normalizing running avg. of gradient
        #     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        # else:
        # assert ((exp_avg_sq - real_exp_avg_sq) >= 0).sum() == np.prod(exp_avg_sq.shape)
        # assert ((exp_avg_ub - real_exp_avg) >= 0).sum() == np.prod(exp_avg_ub.shape)
        # assert ((exp_avg_lb - real_exp_avg) <= 0).sum() == np.prod(exp_avg_lb.shape)
        # exp_avg = real_exp_avg
        # exp_avg_sq = real_exp_avg_sq

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        # cube_avg = (exp_avg_ub**3 + exp_avg_lb**3) / 2
        # p.data.addcdiv_(cube_avg.sign() * cube_avg.abs().pow(1/3), denom, value=-step_size)
        # use_ub_mask = (exp_avg_ub.abs() > exp_avg_lb.abs()).float()
        # p.data.addcdiv_(exp_avg_ub * use_ub_mask, denom, value=-step_size)
        # p.data.addcdiv_(exp_avg_lb * (1-use_ub_mask), denom, value=-step_size)
        p.data.addcdiv_(exp_avg_ub, denom, value=-step_size/2)
        p.data.addcdiv_(exp_avg_lb, denom, value=-step_size/2)

    def sparse(self, p, grad, group):
        state = self.state[p]
        # grad = grad.to_dense()
        # NOTE hack just for word embeddings, basically, to sparsify gradients
        # nonzero_idx = (grad.abs().sum(dim=1) != 0).nonzero().flatten()
        # nonzero_idx = (grad.max(dim=1)[0] != grad.min(dim=1)[0]).nonzero().flatten()
        # grad = grad[nonzero_idx]
        grad = grad.coalesce()
        nonzero_idx = grad._indices()[0]
        grad = grad._values()

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # init exp_avg and exp_avg_sq as sketches
            state['exp_avg'] = [[torch.zeros(p.shape[i]).to(p.device), torch.zeros(p.shape[i]).to(p.device)] for i in range(len(p.shape))]
            state['exp_avg_sq'] = [torch.zeros(p.shape[i]).to(p.device) for i in range(len(p.shape))]

        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        step_size = group['lr'] / bias_correction1

        exp_avg_sq = state['exp_avg_sq'][0].mul_(beta2)[nonzero_idx]
        for i in range(1, len(p.shape)):
            state['exp_avg_sq'][i].mul_(beta2)
            exp_avg_sq = torch.min(exp_avg_sq.unsqueeze(i), state['exp_avg_sq'][i].view([1 for _ in range(i)] + [-1]))
        exp_avg_sq.addcmul_(grad, grad, value=1-beta2)
        for i in range(len(p.shape)):
            exp_avg_sq_view = exp_avg_sq
            if i != 0:
                exp_avg_sq_view = exp_avg_sq_view.transpose(0, i)
            if len(p.shape) > 1:
                exp_avg_sq_view = exp_avg_sq_view.flatten(1)
                exp_avg_sq_view_max = exp_avg_sq_view.max(dim=1)[0]
            else:
                exp_avg_sq_view_max = exp_avg_sq_view
            if i == 0:
                state['exp_avg_sq'][i][nonzero_idx] = exp_avg_sq_view_max.clone()
            else:
                state['exp_avg_sq'][i] = torch.max(state['exp_avg_sq'][i], exp_avg_sq_view_max.clone())
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        exp_avg_ub = state['exp_avg'][0][0].mul_(beta1)[nonzero_idx]
        for i in range(1, len(p.shape)):
            state['exp_avg'][i][0].mul_(beta1)
            exp_avg_ub = torch.min(exp_avg_ub.unsqueeze(i), state['exp_avg'][i][0].view([1 for _ in range(i)] + [-1]))
        exp_avg_ub.add_(grad, alpha=1-beta1)
        for i in range(len(p.shape)):
            exp_avg_ub_view = exp_avg_ub
            if i != 0:
                exp_avg_ub_view = exp_avg_ub_view.transpose(0, i)
            if len(p.shape) > 1:
                exp_avg_ub_view = exp_avg_ub_view.flatten(1)
                exp_avg_ub_view_max = exp_avg_ub_view.max(dim=1)[0]
            else:
                exp_avg_ub_view_max = exp_avg_ub_view
            if i == 0:
                state['exp_avg'][i][0][nonzero_idx] = exp_avg_ub_view_max.clone()
            else:
                state['exp_avg'][i][0] = torch.max(state['exp_avg'][i][0], exp_avg_ub_view_max.clone())
        p.data[nonzero_idx] = p.data[nonzero_idx] - step_size * exp_avg_ub / denom / 2
        # p.data[nonzero_idx].addcdiv_(exp_avg_ub, denom, value=-step_size)

        exp_avg_lb = state['exp_avg'][0][1].mul_(beta1)[nonzero_idx]
        for i in range(1, len(p.shape)):
            state['exp_avg'][i][1].mul_(beta1)
            exp_avg_lb = torch.max(exp_avg_lb.unsqueeze(i), state['exp_avg'][i][1].view([1 for _ in range(i)] + [-1]))
        exp_avg_lb.add_(grad, alpha=1-beta1)
        for i in range(len(p.shape)):
            exp_avg_lb_view = exp_avg_lb
            if i != 0:
                exp_avg_lb_view = exp_avg_lb_view.transpose(0, i)
            if len(p.shape) > 1:
                exp_avg_lb_view = exp_avg_lb_view.flatten(1)
                exp_avg_lb_view_max = exp_avg_lb_view.min(dim=1)[0]
            else:
                exp_avg_lb_view_max = exp_avg_lb_view
            if i == 0:
                state['exp_avg'][i][1][nonzero_idx] = exp_avg_lb_view_max.clone()
            else:
                state['exp_avg'][i][1] = torch.min(state['exp_avg'][i][1], exp_avg_lb_view_max.clone())
        p.data[nonzero_idx] = p.data[nonzero_idx] - step_size * exp_avg_lb / denom / 2
        # p.data[nonzero_idx].addcdiv_(exp_avg_lb, denom, value=-step_size)
        # use_ub_mask = (exp_avg_ub.abs() > exp_avg_lb.abs()).float()
        # p.data[nonzero_idx].addcdiv_(exp_avg_ub * use_ub_mask, denom, value=-step_size*1000)
        # p.data[nonzero_idx].addcdiv_(exp_avg_lb * (1-use_ub_mask), denom, value=-step_size*1000)

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

                if grad.is_sparse:
                    self.sparse(p, grad, group)
                else:
                    self.dense(p, grad, group)
        return loss
