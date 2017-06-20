import os
import cv2
import parmap
import argparse
import numpy as np
from   pathlib import Path
from   tqdm    import tqdm as tqdm
import matplotlib.pylab as plt
import dbm
import pickle
from PIL import Image
def format_image(img_path, size, nb_channels):
  """
  Load img with opencv and reshape
  """
  if nb_channels == 1:
      img = cv2.imread(img_path, 0)
      img = np.expand_dims(img, axis=-1)
  else:
      print( img_path )
      Image.open( img_path )
      img = cv2.imread(img_path.strip())
      img = img[:, :, ::-1]  # GBR to RGB
  w = img.shape[1]
  # Slice image in 2 to get both parts
  img_full   = img[:, :w // 2, :]
  img_sketch = img[:, w // 2:, :]

  img_full   = cv2.resize(img_full, (size, size), interpolation=cv2.INTER_AREA)
  img_sketch = cv2.resize(img_sketch, (size, size), interpolation=cv2.INTER_AREA)
  if nb_channels == 1:
      img_full = np.expand_dims(img_full, -1)
      img_sketch = np.expand_dims(img_sketch, -1)
  img = Image.fromarray( img_sketch )
  img.save("t.jpg")
  """ NOTICE この並びにする """
  img_full   = np.expand_dims(img_full, 0).transpose(0, 3, 1, 2)
  img_sketch = np.expand_dims(img_sketch, 0).transpose(0, 3, 1, 2)
  return img_full, img_sketch

def build_HDF5(jpeg_dir, nb_channels, size=256):
  """
  Gather the data in a single HDF5 file.
  """
  # Put train data in HDF5
  file_name = os.path.basename(jpeg_dir.rstrip("/"))
  for dtype in ["train", "test", "val"]:
    os.system( "mkdir {}".format(dtype) )
    files = list(map(str, Path(jpeg_dir).glob('%s/*.jpg'%dtype) ) )
    files.extend(list(map(str, Path(jpeg_dir).glob('%s/*.png'%dtype) ) ) )
    for e, file in tqdm(enumerate(files)):
      print( file )
      output = format_image(file, size, nb_channels)
      open("{}/{}.pkl".format(dtype,e), "wb").write( pickle.dumps(output) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to jpeg images')
    parser.add_argument('nb_channels', type=int, help='number of image channels')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', action="store_true",
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()
    data_dir = ""
    build_HDF5(args.jpeg_dir, args.nb_channels, size=args.img_size)
