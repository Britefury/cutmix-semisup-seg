import torch


class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, ema_alpha):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = ema_alpha
        self.target_params = [p for p in target_net.state_dict().values() if p.dtype == torch.float]
        self.source_params = [p for p in source_net.state_dict().values() if p.dtype == torch.float]

        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p[...] = src_p[...]

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')


    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(self.ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)
