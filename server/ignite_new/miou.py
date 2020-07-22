from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import torch
import numpy as np

class mIoU(Metric):

    def __init__(self, threshold, output_transform=lambda x: x, device=None):
      """
      Intersection over Union metric calculation
      """
      self._threshold = threshold
      self._miou_sum = None
      self._num_examples = None
      super(mIoU, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
      """
      Reset of values
      """
      self._miou_sum = 0
      self._num_examples = 0
      super(mIoU, self).reset()

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

      self._miou_sum += ((intersection.sum() + eps) / (union.sum() + eps)).detach().item()
      self._num_examples += 1

    @sync_all_reduce("_num_examples")
    def compute(self):
      """
      Final calculation of average IoU metric over all batches
      :return: metric result over all batches
      """
      if self._num_examples == 0:
          raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
      return self._miou_sum/self._num_examples