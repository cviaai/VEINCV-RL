from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import torch
import numpy as np

class Precision(Metric):

    def __init__(self, threshold, output_transform=lambda x: x, device=None):
      """
      Precision metric calculation
      """
      self._threshold = threshold
      self._precision_sum = None
      self._num_examples = None
      super(Precision, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
      """
      Reset of values
      """
      self._precision_sum = 0
      self._num_examples = 0
      super(Precision, self).reset()

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

      true_positive = (prediction_bin * ground_truth_bin)
      false_positive = prediction_bin - true_positive
      #false_negative = ground_truth_bin - true_positive
      #true_negative = torch.ones_like(ground_truth_bin) - false_positive - false_negative - true_positive

      self._precision_sum += ((true_positive.sum() + eps) / (true_positive.sum() + false_positive.sum() + eps)).detach().item()
      self._num_examples += 1

    @sync_all_reduce("_num_examples")
    def compute(self):
      """
      Final calculation of average Precision metric over all batches
      :return: metric result over all batches
      """
      if self._num_examples == 0:
          raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
      return self._precision_sum/self._num_examples