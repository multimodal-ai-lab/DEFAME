import torch.nn as nn
from third_party.effort.DeepfakeBench.training.loss.abstract_loss_func import AbstractLossClass
from third_party.effort.DeepfakeBench.training.utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="l1loss")
class L1Loss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, inputs, targets):
        """
        Computes the l1 loss.
        """
        # Compute the l1 loss
        loss = self.loss_fn(inputs, targets)

        return loss