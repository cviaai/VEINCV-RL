from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from utils import *
import torch
import matplotlib.pyplot as plt
from torch import nn

from ignite.engine import Events, create_supervised_evaluator, \
    create_supervised_trainer

from iou import IoU
from miou import mIoU
from ssim import SSIM
from loss import Loss
#from tversky_loss import Tversky_Loss
from precision import Precision
from recall import Recall

from ignite.engine import Engine, _prepare_batch
from ignite.handlers import ModelCheckpoint

import segmentation_models_pytorch as smp

from dice_helpers import *

def run_experiment(model,
                   train_dataset,
                   val_dataset,
                   parameters):
    
    # params and data loading
    NUM_CLASSES = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = model.to(device)
    
    lr = parameters.get("lr")
    optimizer = Adam(model.parameters(), lr = lr)
    #optimizer = SGD(model.parameters(), lr = lr) # 0 RAM for batch = 1
    
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, (50,100,150), gamma=.5)

    num_epochs = parameters.get("epochs")
    batch_size = parameters.get("batch_size")
    threshold = parameters.get("threshold")
    loss_name = parameters.get("loss_name")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, num_workers=1)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                      shuffle=True, num_workers=1)
    
    if loss_name == "BCE":
        criterion = nn.BCELoss()
        #criterion = nn.BCELoss(weight=torch.tensor(10, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    if loss_name == "DICE":
        criterion = smp.utils.losses.DiceLoss()

    # main training cycle
    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        preds = model.forward(images)
        n = preds.shape[0]
        plt.imsave('/content/drive/My Drive/Colab Files/server/img_check/img.png', tensor2numpy(images[0]))
        plt.imsave('/content/drive/My Drive/Colab Files/server/img_check/mask.png', tensor2numpy(masks[0]))
        plt.imsave('/content/drive/My Drive/Colab Files/server/img_check/pred.png', tensor2numpy(preds[0]))
        plt.imsave('/content/drive/My Drive/Colab Files/server/img_check/pred_bin.png', tensor2numpy((preds[0] - threshold).clamp_min(0).sign()))
        if loss_name == "clDICE":
            loss = torch.mean(soft_cldice_loss(preds, masks, target_skeleton=None))
        else:    
            y = masks.view(n, -1).sign()
            y_pred = preds.view(n, -1)
            if loss_name == "COMB":
                loss = nn.BCELoss()(y_pred, y) + smp.utils.losses.DiceLoss()(y_pred, y)
            else:
                loss = criterion(y_pred, y)
        loss.backward() 
        optimizer.step()
        #scheduler.step()
        return loss.detach().item()
    
    trainer = Engine(process_function)
    
    # evaluation and metrics calculation
    metrics= {'IoU': IoU(threshold),
              'mIoU': mIoU(threshold),
              'SSIM': SSIM(threshold),
              'Precision': Precision(threshold),
              'Recall': Recall(threshold),
          loss_name: Loss(loss_name)}
          #loss_name: Tversky_Loss()}

    train_evaluator = create_supervised_evaluator(model, metrics, device=device)

    val_evaluator = create_supervised_evaluator(model, metrics, device=device)

    training_history = {'IoU':[], 'mIoU':[], 'SSIM':[], 'Precision':[], 'Recall':[], loss_name + '_loss':[]}
    validation_history = {'IoU':[], 'mIoU':[], 'SSIM':[], 'Precision':[], 'Recall':[], loss_name + '_loss':[]}
    last_epoch = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        IoU = metrics['IoU']
        mIoU = metrics['mIoU']
        SSIM = metrics['SSIM']
        Precision = metrics['Precision']
        Recall = metrics['Recall']
        loss = metrics[loss_name]
        last_epoch.append(0)
        training_history['IoU'].append(IoU)
        training_history['mIoU'].append(mIoU)
        training_history['SSIM'].append(SSIM)
        training_history['Precision'].append(Precision)
        training_history['Recall'].append(Recall)
        training_history[loss_name + '_loss'].append(loss)
        print("Epoch[{}] Train IoU: {:.3f} Train mIoU: {:.3f} Train SSIM: {:.3f} Train Precision: {:.3f} Train Recall: {:.3f} Train Loss: {:.3f}"
                          .format(trainer.state.epoch,
                                  IoU,
                                  mIoU,
                                  SSIM,
                                  Precision,
                                  Recall,
                                  loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        IoU = metrics['IoU']
        mIoU = metrics['mIoU']
        SSIM = metrics['SSIM']
        Precision = metrics['Precision']
        Recall = metrics['Recall']
        loss = metrics[loss_name]
        validation_history['IoU'].append(IoU)
        validation_history['mIoU'].append(mIoU)
        validation_history['SSIM'].append(SSIM)
        validation_history['Precision'].append(Precision)
        validation_history['Recall'].append(Recall)
        validation_history[loss_name + '_loss'].append(loss)
        print("Epoch[{}] Val IoU: {:.3f} Val mIoU: {:.3f} Val SSIM: {:.3f} Val Precision: {:.3f} Val Recall: {:.3f} Val Loss: {:.3f}"
                          .format(trainer.state.epoch,
                                  IoU,
                                  mIoU,
                                  SSIM,
                                  Precision,
                                  Recall,
                                  loss))

    # saving of models
    def score_function(engine):
        val_avg_accuracy = val_evaluator.state.metrics['IoU']
        return val_avg_accuracy

    best_model_saver = ModelCheckpoint("/content/drive/My Drive/Colab Files/server/best_models",  
                                       filename_prefix="AttU_Net",
                                       score_name="val_accuracy",  
                                       score_function=score_function,
                                       n_saved=3,
                                       save_as_state_dict=True,
                                       require_empty=False,
                                       create_dir=False)

    val_evaluator.add_event_handler(Events.COMPLETED, 
                                    best_model_saver, 
                                    {"best_model": model})
    
    trainer.run(train_loader, max_epochs=num_epochs)