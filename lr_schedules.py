import ast
import torch

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a polynomial
    schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, power=0.9, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.power = power
        self.eta_min = eta_min
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            # Get progress through schedule
            progress = float(self.last_epoch) / float(self.T_max)
            # Clamp to [0,1]
            progress = min(max(progress, 0), 1)
            # Compute annealing factor
            fac = (1.0 - progress) ** self.power
            fac = max(fac, self.eta_min)
            return [base_lr * fac for base_lr in self.base_lrs]



def make_lr_schedulers(optimizer, total_iters, schedule_type, step_epochs, step_gamma, poly_power=0.9):
    lr_epoch_scheduler = None
    lr_iter_scheduler = None
    if schedule_type == 'none':
        pass
    elif schedule_type == 'stepped' and step_epochs is not None and step_epochs.strip() != '':
        if step_epochs is None:
            raise TypeError('lr_step_epochs should not be None')
        if isinstance(step_epochs, str):
            if step_epochs.strip() == '':
                raise ValueError('lr_step_epochs should not be an empty string')
            step_epochs = ast.literal_eval(step_epochs)
        if isinstance(step_epochs, (list, tuple)) and len(step_epochs) > 0:
            lr_epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=step_epochs, gamma=step_gamma)
    elif schedule_type == 'cosine':
        lr_iter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=total_iters, eta_min=0.0
        )
    elif schedule_type == 'poly':
        lr_iter_scheduler = PolynomialLR(
            optimizer=optimizer, T_max=total_iters, power=poly_power, eta_min=0.0)
    else:
        raise ValueError('Unknown schedule_type {}'.format(schedule_type))

    return lr_epoch_scheduler, lr_iter_scheduler
