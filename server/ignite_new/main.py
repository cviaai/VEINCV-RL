import torch
import sys
from utils import *
from train import *
from VeinsDataset import *
from architectures.attention_unet_architectures import *
from architectures.encoders_unet_architectures import *
from architectures.base_unet_architecture import *
from ax.service.managed_loop import optimize

# Paths loading
main_path = '/content/drive/My Drive/Colab Files/server/dataset_90/'
#main_path = '/content/drive/My Drive/Colab Files/server/dataset/'
train_img_fold = main_path + 'img/train/'
val_img_fold = main_path + 'img/val/'
train_mask_fold = main_path + 'mask/train/'
val_mask_fold = main_path + 'mask/val/'

# Dataloader call
train_loader = VeinsDataset(train_img_fold, train_mask_fold, augmentation = True)
val_loader = VeinsDataset(val_img_fold, val_mask_fold, augmentation = False)

# Load and train model
model = AttU_Net()
#model = R2U_Net()
#model = R2AttU_Net()
#model = U_Net()
#model = UNet11()
#model = AlbuNet()
#model = Unet(n_base_channels=32)

experiment = run_experiment(
    model = model,
    train_dataset = train_loader,
    val_dataset = val_loader,
    parameters={
        "lr": 1e-4,
        "optimizer_name": "Adam",
        "epochs": 300,
        "batch_size": 3,
      	"threshold": 0.4,
        "loss_name": "DICE",
    }
)

experiment

'''
"lr": [1e-2, 1e-3, 1e-4],
"optimizer_name": ["Adam", "SGD"]
"loss_name": ["BCE", "DICE", "COMB", "clDICE"]
'''