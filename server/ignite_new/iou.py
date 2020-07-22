from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import torch
import numpy as np

class IoU(Metric):

    def __init__(self, threshold, output_transform=lambda x: x, device=None):
      """
      Intersection over Union metric calculation
      """
      self._threshold = threshold
      self._intersection_sum = None
      self._union_sum = None
      super(IoU, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
      """
      Reset of values
      """
      self._intersection_sum = 0
      self._union_sum = 0
      super(IoU, self).reset()

    @reinit__is_reduced
    def update(self, output):
      """
      Update metric for current output
      :param output: batch loading of pair (prediction, original mask)
      """
      y_pred, y = output
      eps = 1e-15
      prediction_bin = (y_pred - self._threshold).clamp_min(0).sign()
      ground_truth_bin = y.sign()

      intersection = (prediction_bin * ground_truth_bin)
      union = torch.where(prediction_bin > 0, prediction_bin, ground_truth_bin)
      
      self._intersection_sum += (intersection.sum() + eps).detach().item()
      self._union_sum += (union.sum() + eps).detach().item()

    @sync_all_reduce("_num_examples")
    def compute(self):
      """
      Final calculation of average IoU metric over all batches
      :return: metric result over all batches
      """
      if self._union_sum == 0:
          raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
      return self._intersection_sum/self._union_sum