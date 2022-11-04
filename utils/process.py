import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch_radon import RadonFanbeam

def save_img(img_np, fname):
  ar = np.clip(img_np*255,0,255).astype(np.uint8)

  img = Image.fromarray(ar)
  img.save(fname)

def load_img(img_path, dtype):
  # Load data
  x = loadmat(img_path)
  data = x['CtDataLimited']

  # Define image size
  image_size = 512
  sino_shape = (721, 560)
  full_angles = np.arange(0, 360.5, 0.5).reshape(1,-1)
  full_angles_rad = full_angles * np.pi / 180

  # Get sinogram, normalize and convert to pytorch tensor
  sinogram = data['sinogram'][0,0]
  sinogram_torch = torch.from_numpy(sinogram).type(dtype)
  sinogram_torch = (sinogram_torch-sinogram_torch.min())/(sinogram_torch.max()-sinogram_torch.min())

  # Get equipment angles
  limited_angles = data['parameters'][0,0]['angles'][0,0]
  limited_angles_rad = limited_angles * np.pi / 180
  ini_ang = np.where(full_angles_rad.ravel() == limited_angles_rad[0,0])[0][0]
  fin_ang = np.where(full_angles_rad.ravel() == limited_angles_rad[-1,-1])[0][0]
  angles_index = (ini_ang, fin_ang)

  # Get parameters from the acquisition
  DSD = data['parameters'][0,0]['distanceSourceDetector'][0,0][0,0]
  DSO = data['parameters'][0,0]['distanceSourceOrigin'][0,0][0,0]
  effPixel = data['parameters'][0,0]['effectivePixelSizePost'][0,0][0,0]
  M = data['parameters'][0,0]['geometricMagnification'][0,0][0,0]
  numDetectors = data['parameters'][0,0]['numDetectorsPost'][0,0][0,0]

  DOD = DSD - DSO
  DSO = DSO / effPixel
  DOD = DOD /effPixel

  # Create radon fanbeam operator
  radon_fanbeam = RadonFanbeam(image_size, full_angles_rad.ravel(), source_distance=DSO, det_distance=DOD, det_count=numDetectors)

  return sinogram_torch, radon_fanbeam, angles_index

def normalize(x):
  return (x-x.min())/(x.max()-x.min())

def np_to_torch(img_np):
  '''Converts image in numpy.array to torch.Tensor.
  From C x W x H [0..1] to  C x W x H [0..1]
  '''
  return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]
