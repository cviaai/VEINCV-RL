from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from utils import *
#from skimage.measure import structural_similarity as ssim
from skimage import measure

import torch
import numpy as np

class SSIM(Metric):

    def __init__(self, threshold, output_transform=lambda x: x, device=None):
      """
      Precision metric calculation
      """
      self._threshold = threshold
      self._ssim_sum = None
      self._num_examples = None
      super(SSIM, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
      """
      Reset of values
      """
      self._ssim_sum = 0
      self._num_examples = 0
      super(SSIM, self).reset()

    @reinit__is_reduced
    def update(self, output):
      """
      Update metric for current output
      :param output: batch loading of pair (prediction, original mask)
      """
      y_pred, y = output

      self._ssim_sum += measure.compare_ssim(tensor2numpy(y[0]), tensor2numpy((y_pred[0] - self._threshold).clamp_min(0).sign()))
      self._num_examples += 1

    @sync_all_reduce("_num_examples")
    def compute(self):
      """
      Final calculation of average Precision metric over all batches
      :return: metric result over all batches
      """
      if self._num_examples == 0:
          raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
      return self._ssim_sum/self._num_examples