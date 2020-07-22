import numpy as np

def tensor2numpy(img_tensor):
  """
  Helper method to transfer image from torch.tensor to numpy.array
  :param img_tensor: image in torch.tensor format
  :return: image in numpy.array format
  """
  img = img_tensor.detach().cpu().numpy().transpose(1,2,0) ### not to take grad of img
  
  img= np.clip(img, 0, 1) # less than 0 = 0, bigger than 1 = 1
  img = img.astype('float32')
  
  if img.shape[-1] == 1:
      img = np.squeeze(img) # e.x. (3,) and not (3, 1)
  
  return img