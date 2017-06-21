
import os
import glob
import math
from PIL import Image
import numpy as np
import pickle
import sys

def resize_crop():
  for name in glob.glob("../../test/*.png"):
    print( name )
    img  = Image.open(name)
    w, h = img.size

    x    = h / 256 
    
    mini = img.resize( (int(w/x), 256) )

    file = name.split("/").pop()

    w, h    = mini.size
    center  = w // 2
    start   = center - 256 // 2
    end     = center + 256 // 2
    cropped = mini.crop((start, 0, end, 256))
    print(w, h, x)
    cropped.save("../../test.resize/{}".format(file))
  ...

def check():
  open("")  

def make_triple():
  files = sorted( glob.glob("../../test.resize/*") )
  print( files )
  for i in range(0, len(files) - 3, 3):
    x1 = files[i]
    x2 = files[i+2]
    y  = files[i+1]
    
    x1 = np.expand_dims(np.array(Image.open(x1)), 0).transpose(0, 3, 1, 2)
    x2 = np.expand_dims(np.array(Image.open(x2)), 0).transpose(0, 3, 1, 2)
    xs = np.concatenate( (x1, x2), axis=1 )
    y  = np.expand_dims(np.array(Image.open(y)), 0).transpose(0, 3, 1, 2)
    pair = (y, xs)
    open("datasets/{}.pkl".format(i),"wb").write( pickle.dumps(pair) )
    print( x1.shape )
    #print( x1, x2, y )
if __name__ == "__main__":
  if "--resize_crop" in sys.argv:
    resize_crop()

  if "--make_triple" in sys.argv:
    make_triple()
