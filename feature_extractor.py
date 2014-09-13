#!/usr/bin/env python2.7

import numpy as np
import pyeeg
import file_handler
from time import sleep
from mpi4py import MPI

'''
The feature extractor should open a file, calculate feature names and values, store the calculations in a clearly named feature file, and continue. $NAME_features.csv EX Dog_3_preictal_segment_0030_features.csv
The feature files will be stored in a subdirectory of $REPO/feature
'''

class FeatureExtractor(object):
  def __init__(self, data):
    self.features = {}
    self.fprime = {}
    self.data = data

  def set_features(self,filen):
    self.set_filen(filen)
    self.set_foo()
    self.set_bar()
    self.set_first_order_diff()
    self.set_pfd()

  def set_hurst(self):
    for i in xrange(self.data.shape[0]):
      name = 'hurst_e' + str(i)
      value = pyeeg.hurst(self.data[i])
      self.features[name] = value

  def set_first_order_diff(self):
    for i in xrange(self.data.shape[0]):
      name = 'first_order_diff_e' + str(i)
      value = pyeeg.first_order_diff(self.data[i])
      self.fprime[name] = value

  def set_pfd(self):
    for i in xrange(self.data.shape[0]):
      name = 'pfd_e' + str(i)
      value = pyeeg.pfd(self.data[i].size, self.fprime['first_order_diff_e' + str(i)])
      self.features[name] = value

  def set_filen(self, filen):
      self.features['filen'] = filen
  
  def set_foo(self):
    for i in xrange(self.data.shape[0]):
      name = 'foo_e' + str(i)
      value = str(self.data[i][0])
      self.features[name] = value

  def set_bar(self):
    for i in xrange(self.data.shape[0]):
      name = 'bar_e' + str(i)
      value = str(self.data[i][-1])
      self.features[name] = value


def record_features(features):
  feature_dir = '/home/kjs/repos/kaggle-aes-seizure-prediction/feature/'
  feature_file = features['filen'].split('.')[0] + '_features.csv'
  with open(feature_dir + feature_file, 'w') as f:
    for feature in features.keys():
      if feature.find('filen') == -1 and feature.find('first_order_diff') == -1:
        f.write(",".join([feature, str(features[feature])]) + '\n')

def main():
  fh = file_handler.FileHandler()
  fh.set_train_interical_preictal_and_test_files()

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  fname = np.array([''.zfill(37)])
  stop_iteration = np.zeros(1)
  data_shape = np.zeros(2, dtype=np.int32)

  test = np.zeros(1024, dtype=np.int16)
  test_end = np.zeros(1024, dtype=np.int16)

  train_files = [np.array([f]) for f in fh.all_train_files]
  if rank == 0:
    i = 0
    for fname in train_files:
      i += 1
      proc = (i % (size - 1)) + 1

      fh.file_in = fname[0]
      fh.set_data()

      data = fh.data[0]
      data = np.array([np.array(data[j], dtype=np.int16) for j in range(data.shape[0])], dtype=np.int16)

      data_shape = np.array(data.shape, dtype=np.int32)
      count = data_shape[0] * data_shape[1]
      print "sending data of shape %s x %s for sample %s" % (data.shape[0], data.shape[1], fh.file_in)
      comm.Send([data_shape, MPI.INT], dest=proc, tag=1)
      comm.Send([stop_iteration, MPI.INT], dest=proc, tag=2)
      comm.Send([fname, MPI.SIGNED_CHAR], dest=proc, tag=3)

      print "SEND: %s" % data[0]
      comm.Send([data, MPI.INT], dest=proc, tag=4)

      comm.Recv([test, MPI.INT], source=MPI.ANY_SOURCE, tag=5)
      assert (test == data[0][:1024]).all()

      comm.Recv([test_end, MPI.INT], source=MPI.ANY_SOURCE, tag=6)
      assert (test_end == data[-1][:1024]).all()


      for r in range(1, size):
        if comm.Iprobe(source=r, tag=11):
          features = comm.recv(source=r, tag=11)
          record_features(features)
    stop_iteration[0] = 1
    for proc in range(1, size):
      comm.Send(stop_iteration, dest=proc)
  else:    
    while True:
      comm.Recv([data_shape, MPI.INT], source=0, tag=1)
      count = data_shape[0] * data_shape[1]
      data = np.array([np.zeros(data_shape[1], dtype=np.int16) for i in range(data_shape[0])], dtype=np.int16)
      print "made data array buffer"

      comm.Recv([stop_iteration, MPI.INT], source=0, tag=2)
      if stop_iteration[0]:
        break
      comm.Recv([fname, MPI.SIGNED_CHAR], source=0, tag=3)

      comm.Recv([data, MPI.INT], source=0, tag=4)
      print "RECV: %s" % data[0]

      test = data[0][:1024]
      comm.Send([test, MPI.INT], dest=0, tag=5)

      test_end = data[-1][:1024]
      comm.Send([test_end, MPI.INT], dest=0, tag=6)
      print "Data recieved by rank %s" % rank

      fe = FeatureExtractor(data)
      fe.set_features(fname[0])

      comm.send(fe.features, dest=0, tag=11)
      
  print "proc %s is complete" % rank

if __name__ == '__main__':
  main()
