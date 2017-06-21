import glob
import pickle
import numpy as np
import random

trains = []
valids = []
def init():
  for name in glob.glob("data/datasets/*.pkl"):
    print( name )
    x,y = pickle.loads( open(name, "rb").read() )
    """ normalize """
    x = x / 127.5 - 1
    y = y / 127.5 - 1
    trains.append( (x,y) )
  for name in glob.glob("data/datasets/*.pkl"):
    print( name )
    x,y = pickle.loads( open(name, "rb").read() )
    """ normalize """
    x = x / 127.5 - 1
    y = y / 127.5 - 1
    print( x.shape )
    valids.append( (x,y) )
  """ valids は最初一回のみシャッフルする """
  random.shuffle( valids ) 
init()

def getTrain(batch):
  random.shuffle( trains ) 
  for i in range(0, len(trains), batch):
    bs = trains[i:i+batch]
    xs = np.concatenate( [bs[i][0] for i in range(batch)] )
    xs = xs.transpose(0, 2, 3, 1)
    ys = np.concatenate( [bs[i][1] for i in range(batch)] )
    ys = ys.transpose(0, 2, 3, 1)
    #print( "now iter", i, xs.shape )
    #print( len(trains) )
    yield (xs, ys)

def getValids(batch):
  for i in range(0, len(valids), batch):
    bs = valids[i:i+batch]
    xs = np.concatenate( [bs[i][0] for i in range(batch)] )
    xs = xs.transpose(0, 2, 3, 1)
    ys = np.concatenate( [bs[i][1] for i in range(batch)] )
    ys = ys.transpose(0, 2, 3, 1)
    #print( "now iter", i, xs.shape )
    yield (xs, ys)

if __name__ == '__main__':
  getTrain(4)
