#!/usr/bin/env python2.7

import numpy as np
import pyeeg
import file_handler
from mpi4py import MPI

'''
The feature extractor should open a file, calculate feature names and values, store the calculations in a clearly named feature file, and continue. $NAME_features.csv EX Dog_3_preictal_segment_0030_features.csv
The feature files will be stored in a subdirectory of $REPO/feature
'''

class FeatureExtractor(object):
  def __init__(self, data):
    self.features = {}
    self.data = data
  def set_features(self):
    #self.set_hurst()
    self.set_foo()
  def set_hurst(self):
    for i in range(self.data.shape[0]):
      name = 'hurst_e' + str(i)
      value = pyeeg.hurst(self.data[i])
      self.features[name] = value
  def set_foo(self):
      name = 'foo'
      value = str(self.data[0][0])
      self.features[name] = value
  def record_features(self, file_in):
    feature_dir = '/home/kjs/repos/kaggle-aes-seizure-prediction/feature/'
    feature_file = file_in.split('.')[0] + '_features.csv'
    with open(feature_dir + feature_file, 'w') as f:
      for feature in self.features.keys():
        f.write(",".join([feature, self.features[feature]]))

def main():
  fh = file_handler.FileHandler()
  fh.set_train_interical_preictal_and_test_files()

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  train_file_buffer = np.array([''.zfill(37)])
  truth_buffer = np.array([0, ])
  human_train_array = np.array([np.array([f]) for f in fh.seg_train_files['Patient']])
  dog_train_array = np.array([np.array([f]) for f in fh.seg_train_files['Dog']])
  all_train_array = np.array([np.array([f]) for f in fh.all_train_files])

  if rank == 0:
    i = 0
    for f in all_train_array:
      i += 1
      proc = (i % (size - 1)) + 1
      comm.Send([f, MPI.SIGNED_CHAR], dest=proc)
      comm.Recv([truth_buffer, MPI.INT], source=MPI.ANY_SOURCE)
  else:    
    for i in range(len(human_train_array)):
      comm.Recv([train_file_buffer, MPI.SIGNED_CHAR], source=0)
      fh.file_in = train_file_buffer[0]
      fh.set_data()

      fe = FeatureExtractor(fh.data[0])
      fe.set_features()
      fe.record_features(train_file_buffer[0])
      truth_buffer[0] = 1
      comm.Send([truth_buffer, MPI.INT], dest=0)

if __name__ == '__main__':
  main()
