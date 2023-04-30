import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage import exposure
from skimage import filters
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from scipy.ndimage import grey_closing, binary_opening
import torch.nn.functional as F
from utils.process import normalize
from utils.dip import get_net, get_params
import odl
from odl.contrib.torch import OperatorFunction

def rec_mmp_seg(sinogram, radon_fanbeam, angles_index, dtype, mmp_iter=3,
                ini_iter=950, w_mmp=0.1):
  # Size
  image_size = 512
  rec_dim = (image_size, image_size)  
  ini_ang, fin_ang = angles_index
  
  # Create circle mask
  Y, X = np.mgrid[0:image_size,0:image_size]

  r = int(image_size/2)
  xc = int(image_size/2)
  yc = int(image_size/2)

  circle_mask = (X-xc)**2 + (Y-yc)**2 < r**2
  circle_mask = torch.from_numpy(circle_mask).unsqueeze(0).unsqueeze(0).type(dtype)

  # Load disk input
  rec_input = loadmat('utils/htc2022_solid_disc_full_recon_fbp.mat')['reconFullFbp']
  rec_input = normalize(rec_input)
  rec_input *= rec_input > 0.3
  rec_input = torch.from_numpy(rec_input).type(dtype).unsqueeze(0).unsqueeze(0)

  # Input Parameters
  input_depth = 1
  input_type = 'noise'
  img_size = (image_size, image_size)
  reg_noise = 5e-2 # Magnitude of the noise added to the input at each DIP iteration

  # Optimization Parameters
  OPTIMIZER = 'adam'
  OPT_OVER = 'net'
  pad = 'reflection'
  NET_TYPE = 'skip'
  LR = 0.00005

  # MMP parameters 
  M_l = [0.001, 0.005, 0.01]
  lmbd = 1.5
  i_mmp_iter = 35
  num_iter = [500, 500, 500] # Size defined by mmp_iter

  # Create mask of the angles outside the limited angle region
  sino_shape = (721, 560)
  cut_mask = torch.zeros(sino_shape).type(dtype)
  cut_mask[ini_ang:fin_ang+1] = 1
  cut_mask = 1 - cut_mask

  # Get known angles to use in the MMP method
  pk = torch.zeros_like(cut_mask).type(dtype)
  pk[ini_ang:fin_ang+1] = sinogram

  rec_net = get_net(input_depth, NET_TYPE, pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=12,
              n_channels=input_depth,
              num_scales=6,
              upsample_mode='bilinear').type(dtype)

  noise = rec_input.detach().clone()

  p = get_params(OPT_OVER,rec_net,rec_input)

  optimizer = torch.optim.Adam(p, lr=LR)

  out_img = DIP_rec(rec_net, rec_input, optimizer, sinogram, ini_iter, radon_fanbeam,
                  ini_ang, fin_ang, cut_mask, circle_mask, reg_noise, noise,
                  plot_interval=None, wl_reg=1e-9
                  )

  for i in range(mmp_iter):
    M = int(M_l[i] * 512**2)

    pu = cut_mask * OperatorFunction.apply(radon_fanbeam, out_img[0,0].detach().clone())
    img_mmp = MMP(M, i_mmp_iter, lmbd, sinogram, out_img, radon_fanbeam, rec_dim, cut_mask,
                 circle_mask, dtype, pk, outer_iter=1, plot_results=False)

    img_mmp_np = img_mmp.detach().cpu().numpy()
    img_mmp_np = grey_closing(denoise_nl_means(img_mmp_np), size=27)
    img_mmp = torch.from_numpy(img_mmp_np).type(dtype)
    sino_mmp = OperatorFunction.apply(radon_fanbeam, img_mmp)

    out_img = DIP_rec(rec_net, rec_input, optimizer, sinogram, num_iter[i], radon_fanbeam,
                      ini_ang, fin_ang, cut_mask, circle_mask, reg_noise, noise,
                      w_mmp=w_mmp, sino_mmp=sino_mmp, plot_interval=None, wl_reg=1e-9
                      )
  
  return out_img[0].clone().detach().cpu().numpy()

def DIP_rec(rec_net, rec_input, optimizer, sinogram_data, num_iter, radon_fanbeam, 
            ini_ang, fin_ang, cut_mask, circle_mask, reg_noise, noise,
            w_mmp=1e-1, sino_mmp=None, plot_interval=None, wl_reg=1e-9):
  
  for i in range(num_iter):
    optimizer.zero_grad()

    out_img = rec_net(rec_input + reg_noise*noise.normal_())
    out_img = circle_mask*(out_img-out_img.min())/(out_img.max()-out_img.min())

    out_sinogram = OperatorFunction.apply(radon_fanbeam, out_img[0,0])

    out_sinogram = (out_sinogram-out_sinogram.min())/(out_sinogram.max()-out_sinogram.min())

    total_loss = F.l1_loss(out_sinogram[ini_ang:fin_ang+1], sinogram_data)

    if(sino_mmp is not None):
      total_loss += w_mmp * F.l1_loss(cut_mask*out_sinogram, cut_mask*sino_mmp)

    if(wl_reg is not None):
      param_layers = [lp.view(-1) for lp in rec_net.parameters()]
      param_w = torch.cat([param_layers[wi] for wi in range(0,len(param_layers),2)])
      wl_loss = wl_reg * torch.norm(param_w, 2)
      total_loss += wl_loss

    total_loss.backward()
    optimizer.step()
    
    if(plot_interval is not None):
      if i%plot_interval == 0:
        img_np = (out_img[0,0]).clone().detach().cpu().numpy()
          
        plt.imshow(img_np,cmap='gray')
        plt.title(f"Step: {i}   |   Loss: {total_loss.item():.6f}")
        plt.colorbar() 
        plt.axis('off')

        plt.show()

  return out_img

def MMP(M, n_iter, lmbd, sinogram_data, init_rec, radon_fanbeam, rec_dim, cut_mask,
         circle_mask, dtype, pk=None, outer_iter=1, plot_results=False):
  if pk is None:
    pk = (1-cut_mask) * sinogram_data
  
  pu = cut_mask * OperatorFunction.apply(radon_fanbeam, init_rec)
  fbp = odl.tomo.fbp_op(radon_fanbeam,
                      filter_type='Shepp-Logan', frequency_scaling=0.8)

  for i in range(outer_iter):
    R = pk + lmbd*pu
    f = torch.zeros((1,rec_dim[0]*rec_dim[1])).type(dtype)

    for i in range(n_iter):
      rec_fbp = circle_mask * OperatorFunction.apply(fbp, R)
      rec_fbp = rec_fbp.reshape(1,-1)

      values_max, ind_max = torch.topk(rec_fbp, M)
      f[0,ind_max] = f[0,ind_max] + rec_fbp[0,ind_max]

      R = torch.clamp(pk + lmbd*pu - OperatorFunction.apply(radon_fanbeam, f.reshape(rec_dim)), min=0)

    pu = cut_mask * OperatorFunction.apply(radon_fanbeam, f.reshape(rec_dim))

  f = normalize(f.reshape(rec_dim))

  if(plot_results):
      img_np = f.clone().detach().cpu().numpy()
      plt.imshow(img_np,cmap='gray')
      plt.title(f"MMP Result")
      plt.colorbar() 
      plt.axis('off')
      plt.show()

  return f

def segment_reconstruction(rec_img):
  n_iterations = 10 
  smoothing_factor = 1 
  checkboard_param = 3 
  sig_cutoff_factor = 0.9

  preprocess_img = denoise_nl_means(rec_img)
  thresh = filters.threshold_otsu(preprocess_img)
  preprocess_img  = exposure.adjust_sigmoid(preprocess_img, cutoff=abs(sig_cutoff_factor-thresh))

  # Initial level set
  init_ls = checkerboard_level_set(preprocess_img.shape, checkboard_param)

  # Apply segmentation algorithm
  ls = morphological_chan_vese(preprocess_img, num_iter=n_iterations, init_level_set=init_ls,
                              smoothing=smoothing_factor)

  # Post processing
  seg = abs(ls - ls[25,25])
  seg = binary_opening(seg, iterations=5)

  return seg
