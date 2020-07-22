import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor

class VeinsDataset(Dataset):
    
    def __init__(self, images_fold, mask_fold, augmentation=False):
      """
      Vein data loading
      :param images_fold: path to vein images
      :param mask_fold: path to vein masks
      :param augmentation: augmentation of images from folder
      """
      self.images_fold = images_fold
      self.mask_fold = mask_fold

      self.augmentation = augmentation
      self.image_mask_paths = [] # list of image_path and its class pairs
      
      for image_name in os.listdir(images_fold):
          if image_name != '.DS_Store':
              mask_name = 'mask_' + image_name[6:][:-4] + '.png'
              self.image_mask_paths.append((image_name, mask_name))            
    
    def __getitem__(self, idx):
      """
      Get image and mask items
      :param idx: names of image and mask pairs
      :return image and mask tensors
      """
      image_name, mask_name = self.image_mask_paths[idx]
      
      img = Image.open(os.path.join(self.images_fold, image_name))
      mask = Image.open(os.path.join(self.mask_fold, mask_name))
      
      if self.augmentation:
          seed = np.random.randint(100000)
          transform = transforms.Compose([transforms.RandomRotation(degrees=180),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip()])
          random.seed(seed)
          img = transform(img)
          random.seed(seed)
          mask = transform(mask)

      img = ToTensor()(img)
      mask = ToTensor()(mask)

      return img, mask

    def __len__(self):
      """
      :return: length of (image, mask) pairs
      """
      return len(self.image_mask_paths)