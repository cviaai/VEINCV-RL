from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from torch import nn
import segmentation_models_pytorch as smp

from dice_helpers import *

class Loss(Metric):

    _required_output_keys = None

    def __init__(self, loss_name, output_transform=lambda x: x,
                 batch_size=lambda x: len(x), device=None):
      """
      Class calculates Binary Cross Entropy and Dice losses
      """
      super(Loss, self).__init__(output_transform, device=device)
      self._loss_name = loss_name
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
      if self._loss_name == "BCE":
          criterion = nn.BCELoss() 
          #criterion = nn.BCELoss(weight=torch.tensor(10, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
      if self._loss_name == "DICE":
          criterion = smp.utils.losses.DiceLoss()
      
      y_pred, y = output

      n = self._batch_size(y)
      if self._loss_name == "clDICE":
          self._sum += torch.mean(soft_cldice_loss(y_pred, y, target_skeleton=None)).detach().item()
      else:
          if self._loss_name == "COMB":
              self._sum += nn.BCELoss()(y_pred.view(n, -1), y.view(n, -1).sign()).detach().item() + smp.utils.losses.DiceLoss()(y_pred.view(n, -1), y.view(n, -1).sign()).detach().item()
      
          else:
              self._sum += criterion(y_pred.view(n, -1), y.view(n, -1).sign()).detach().item()
          
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