import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from scipy.ndimage import grey_closing
import torch.nn.functional as F
from utils.process import normalize
from utils.dip import get_net, get_params

def rec_mmp_seg(sinogram, radon_fanbeam, angles_index, dtype):
  # Size
  image_size = 512 
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
  ini_iter = 950

  # Optimization Parameters
  OPTIMIZER = 'adam'
  OPT_OVER = 'net'
  pad = 'reflection'
  NET_TYPE = 'skip'
  LR = 0.00005

  # MMP parameters 
  mmp_iter = 3
  w_mmp = 1e-1 # MMP weight
  thresh = [0.9, 0.85, 0.8] # Size defined by mmp_iter
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
    pu = cut_mask * radon_fanbeam.forward(out_img[0,0].detach().clone())
    img_mmp = MMP(pk, pu, radon_fanbeam, 500, thresh[i], 1, dtype, img_size, plot_results=False)
    img_mmp_np = img_mmp.detach().cpu().numpy().reshape(img_size)
    img_mmp_np = grey_closing(denoise_nl_means(img_mmp_np), size=15)
    img_mmp = torch.from_numpy(img_mmp_np).type(dtype)
    sino_mmp = radon_fanbeam.forward(img_mmp)

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
    out_sinogram = radon_fanbeam.forward(out_img[0,0])

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

def MMP(pk, pu, radon_fanbeam, M, thresh_perc, lmbd,
        dtype, rec_dim, plot_results=False):
  R = pk + lmbd*pu
  f = torch.zeros((1,rec_dim[0]*rec_dim[1])).type(dtype)
  thresh = thresh_perc * torch.norm(R)

  while(torch.norm(R) > thresh):
    filtered_sinogram = radon_fanbeam.filter_sinogram(R)
    rec_fbp = radon_fanbeam.backprojection(filtered_sinogram).reshape((1,-1))

    _, ind_max = torch.topk(rec_fbp, M)
    f[0,ind_max] = f[0,ind_max] + rec_fbp[0,ind_max]

    R = pk + lmbd*pu - radon_fanbeam.forward(f.reshape(rec_dim))

  f = (f-f.min()) / (f.max()-f.min())

  if(plot_results):
      img_np = f.clone().detach().cpu().numpy()
      plt.imshow(img_np.reshape(rec_dim),cmap='gray')
      plt.title(f"MMP Result")
      plt.colorbar() 
      plt.axis('off')
      plt.show()

  return f