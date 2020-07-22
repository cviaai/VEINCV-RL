from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from torch import nn

import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy, softmax


def tversky_loss(logits, true, num_classes = 1, alpha=0.5, beta=0.5):
	true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    true_1_hot_f = true_1_hot[:, 0:1, :, :]
    true_1_hot_s = true_1_hot[:, 1:2, :, :]
    true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
    pos_prob = torch.sigmoid(logits)
    neg_prob = 1 - pos_prob
    probas = torch.cat([pos_prob, neg_prob], dim=1)
    
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)

class Tversky_Loss(Metric):

    _required_output_keys = None

    def __init__(self, output_transform=lambda x: x,
                 batch_size=lambda x: len(x), device=None):
      """
      Class calculates Binary Cross Entropy and Dice losses
      """
      super(Tversky_Loss, self).__init__(output_transform, device=device)
      self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self):
      """
      Reset of values
      """
      self._sum = 0
      self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
      """
      Update Loss for current output
      :param output: batch loading of pair (prediction, original mask)
      """
      
      y_pred, y = output

      n = self._batch_size(y)

      #self._sum += tversky_loss(y_pred.view(n, -1), y.view(n, -1).sign()).detach().item()
      self._sum += tversky_loss(y_pred, y.sign()).detach().item()
          
      self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
      """
      Final calculation of average Loss over all batches
      :return: Loss result over all batches
      """
      if self._num_examples == 0:
          raise NotComputableError(
              'Loss must have at least one example before it can be computed.')
      return self._sum / self._num_examples