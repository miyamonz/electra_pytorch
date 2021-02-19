import time
from statistics import mean, stdev
import torch
from torch import nn
from fastai.text.all import *

"""
I would like more uniform way to pass the metrics, no matter loss_func or metric,
instantiate it and then pass.
This uniform way also make it possible such as `metrics=[m() for m inTASK_METRICS[task]]`
"""


class GradientClipping(Callback):
    def __init__(self, clip: float = 0.1):
        self.clip = clip
        assert self.clip

    def after_backward(self):
        if hasattr(self, 'scaler'):
            self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)


class RunSteps(Callback):
    toward_end = True

    def __init__(self, n_steps, save_points=None, base_name=None, no_val=True):
        """
        Args:
          `n_steps` (`Int`): Run how many steps, could be larger or smaller than `len(dls.train)`
          `savepoints` 
          - (`List[Float]`): save when reach one of percent specified.
          - (`List[Int]`): save when reache one of steps specified
          `base_name` (`String`): a format string with `{percent}` to be passed to `learn.save`.
        """
        if save_points is None:
            save_points = []
        else:
            assert '{percent}' in base_name
            save_points = [s if isinstance(s, int) else int(
                n_steps*s) for s in save_points]
            for sp in save_points:
                assert sp != 1, "Are you sure you want to save after 1 steps, instead of 1.0 * num_steps ?"
            assert max(save_points) <= n_steps
        store_attr('n_steps,save_points,base_name,no_val', self)

    def before_train(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter/self.n_steps

    def after_batch(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter/self.n_steps
        # when to save
        if self.train_iter in self.save_points:
            percent = (self.train_iter/self.n_steps)*100
            self.learn.save(self.base_name.format(percent=f'{percent}%'))
        # when to interrupt
        if self.train_iter == self.n_steps:
            raise CancelFitException

    def after_train(self):
        if self.no_val:
            if self.train_iter == self.n_steps:
                pass  # CancelFit is raised, don't overlap it with CancelEpoch
            else:
                raise CancelEpochException
