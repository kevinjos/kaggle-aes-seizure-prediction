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
  def set_features(self,filen):
    #self.set_hurst()
    self.set_foo()
    self.set_filen(filen)
  def set_hurst(self):
    for i in range(self.data.shape[0]):
      name = 'hurst_e' + str(i)
      value = pyeeg.hurst(self.data[i])
      self.features[name] = value
  def set_foo(self):
      name = 'foo'
      value = str(self.data[0][0])
      self.features[name] = value
  def set_filen(self, filen):
      self.features['filen'] = filen

def record_features(features):
  feature_dir = '/home/kjs/repos/kaggle-aes-seizure-prediction/feature/'
  feature_file = features['filen'].split('.')[0] + '_features.csv'
  with open(feature_dir + feature_file, 'w') as f:
    for feature in features.keys():
      if feature != 'filen':
        f.write(",".join([feature, features[feature]]))

def main():
  fh = file_handler.FileHandler()
  fh.set_train_interical_preictal_and_test_files()

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  train_file_buffer = np.array([''.zfill(37)])
  train_files = np.array([np.array([f]) for f in fh.all_train_files])
  train_data_buffer = np.array([np.zeros(4500000) for i in range(24)])

  if rank == 0:
    i = 0
    for f in train_files:
      i += 1
      proc = (i % (size - 1)) + 1

      fh.file_in = f[0]
      fh.set_data()

      data = fh.data[0]

      comm.Send([f, MPI.SIGNED_CHAR], dest=proc)
      comm.Send([data, MPI.SIGNED_INT], dest=proc)
      features = comm.recv(source=MPI.ANY_SOURCE)
      record_features(features)
  else:    
    for i in range(len(train_files)):
      comm.Recv([train_file_buffer, MPI.SIGNED_CHAR], source=0)
      comm.Recv([train_data_buffer, MPI.SIGNED_INT], source=0)

      fe = FeatureExtractor(train_data_buffer)
      fe.set_features(train_file_buffer[0])

      comm.send(fe.features, dest=0)

if __name__ == '__main__':
  MAIN()
