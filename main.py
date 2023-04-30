import re
import os
import torch
import argparse

from utils import process, tools

# Get input arguments
parser = argparse.ArgumentParser(description=
            'Deblur images with different levels of blur.')

parser.add_argument('input_path', type=str,
                    help='Path with sinograms to be reconstructed.')
parser.add_argument('output_path', type=str,
                    help='Path to save the segmentation of the reconstructed images.')
parser.add_argument('category', type=int,
                    choices=range(1,8), metavar='[1-7]',
                    help='Difficulty category number.')

args = parser.parse_args()

# Get image names
img_names = sorted(os.listdir(args.input_path))
r = re.compile(".*\.mat")
img_names = list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
if torch.cuda.is_available():
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  dtype = torch.cuda.FloatTensor
  map_location = None
else:
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  dtype = torch.FloatTensor
  map_location = 'cpu'

for img_name in img_names:
  path_in = os.path.join(args.input_path, img_name)
  path_out = os.path.join(args.output_path, img_name)
  path_out = path_out[0:-3] + 'png'

  sinogram_torch, radon_fanbeam, angles_index = process.load_img(path_in, dtype)

  img_rec = tools.rec_mmp_seg(sinogram_torch, radon_fanbeam, angles_index, dtype)
  img_rec = process.normalize(img_rec)
  img_seg = tools.segment_reconstruction(img_rec[0])

  process.save_img(img_seg, path_out)
