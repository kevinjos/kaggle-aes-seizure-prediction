#!/usr/bin/env python2.7

import numpy as np
import pyeeg
import file_handler
import csv
import time
from decs import coroutine
from mpi4py import MPI

'''
The feature extractor creates a features.csv file
The file is organized row by column, sample by features
Names of features must be know before writing the file
For electrode specific features, we use the max_e
If a sample uses less electrodes that max_e,
  then it will have NA values for these electrodes
In cases where a calculation is recycled in 2+ features,
  ex first order differential
  the calc is stroed in a by electrode dictionary
'''

class FeatureExtractor(object):
  def __init__(self, data):
    self.features = {}
    self.fprime = {}
    self.data = data

  def set_features(self,filen):
    self.set_filen(filen)
    self.set_firstval()
    self.set_lastval()
    self.set_maxval()
    self.set_minval()
    self.set_meanval()
    self.set_samplesize()
    #self.set_hurst()
    #self.set_first_order_diff()
    #self.set_pfd()

  def set_filen(self, filen):
      self.features['filen'] = filen.split('.')[0]

  def set_firstval(self):
    for i in xrange(self.data.shape[0]):
      name = 'firstval_e' + str(i)
      value = str(self.data[i][0])
      self.features[name] = value

  def set_lastval(self):
    for i in xrange(self.data.shape[0]):
      name = 'lastval_e' + str(i)
      value = str(self.data[i][-1])
      self.features[name] = value

  def set_maxval(self):
    for i in xrange(self.data.shape[0]):
      name = 'maxval_e' + str(i)
      value = str(np.max(self.data[i]))
      self.features[name] = value

  def set_minval(self):
    for i in xrange(self.data.shape[0]):
      name = 'minval_e' + str(i)
      value = str(np.min(self.data[i]))
      self.features[name] = value

  def set_meanval(self):
    for i in xrange(self.data.shape[0]):
      name = 'meanval_e' + str(i)
      value = str(np.mean(self.data[i]))
      self.features[name] = value

  def set_samplesize(self):
    for i in xrange(self.data.shape[0]):
      name = 'samplesize_e' + str(i)
      value = str(self.data[i].size)
      self.features[name] = value

  def set_hurst(self):
    '''
    n squared -- takes long time
    we should try this with data compression
    '''
    for i in xrange(self.data.shape[0]):
      name = 'hurst_e' + str(i)
      value = pyeeg.hurst(self.data[i])
      self.features[name] = value

  def set_first_order_diff(self):
    '''
    used in other features
    itself is not a feature to be recorded
    '''
    for i in xrange(self.data.shape[0]):
      name = 'first_order_diff_e' + str(i)
      value = pyeeg.first_order_diff(self.data[i])
      self.fprime[name] = value

  def set_pfd(self):
    '''
    petrosian fractal dimension
    '''
    for i in xrange(self.data.shape[0]):
      name = 'pfd_e' + str(i)
      value = pyeeg.pfd(self.data[i].size, self.fprime['first_order_diff_e' + str(i)])
      self.features[name] = value

@coroutine
def record_features(fieldnames=[]):
  feature_file = '/home/kjs/repos/kaggle-aes-seizure-prediction/train_features.csv'
  try:
    f = open(feature_file, 'w')
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    while True:
      features = yield
      writer.writerow(features)
  except GeneratorExit:
    f.close()

def get_feature_names(max_e=24):
  fakedata = np.array([np.array([1,1,1]) for i in range(max_e)])
  fe = FeatureExtractor(fakedata)
  fe.set_features('fakefile')
  return fe.features.keys()

def main():
  #Prepare for file I/O
  fh = file_handler.FileHandler()
  fh.set_train_interical_preictal_and_test_files()
  train_files = np.array([np.array([f]) for f in fh.all_train_files])

  #Prepare for MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Make MPI buffers
  fname = np.array([''.zfill(37)])
  stop_iteration = np.zeros(1)
  data_shape = np.zeros(2, dtype=np.int32)

  if rank == 0:
    #Prepare tabular feature file header
    feature_names = get_feature_names()
    rf = record_features(feature_names)

    #Setup counters to be sure all files are accounted for
    total_responses = 0
    expected_responses = train_files.size
    for i in xrange(expected_responses):
      #Calculate the process rank to send the next file to
      proc = (i % (size - 1)) + 1

      #Prepare the data for MPI C type send
      fh.file_in = train_files[i][0]
      fh.set_data()
      data = np.array([np.array(fh.data[0][j], dtype=np.int32) for j in xrange(fh.data[0].shape[0])], dtype=np.int32)

      #Calculate the shape of the data to make the correctly sized buffer in the recieving processes
      data_shape = np.array(data.shape, dtype=np.int32)

      #Set the file name to send
      fname = train_files[i]

      #Send data
      comm.Send([stop_iteration, MPI.INT], dest=proc, tag=1)
      comm.Send([data_shape, MPI.INT], dest=proc, tag=2)
      comm.Send([fname, MPI.SIGNED_CHAR], dest=proc, tag=3)
      comm.Send([data, MPI.INT], dest=proc, tag=4)
      print "Data set number %s sent" % i

      #Probe processes for features and if recieved append features to feature file
      for r in xrange(1, size):
        if comm.Iprobe(source=r, tag=55):
          total_responses += 1
          features = comm.recv(source=r, tag=55)
          rf.send(features)

    #Cleanly exit execution by sending stop iteration signal to workers and closing feature record generator
    stop_iteration[0] = 1
    while total_responses != expected_responses:
      time.sleep(5)
      for proc in xrange(1, size):
        if comm.Iprobe(source=proc, tag=55):
          total_responses += 1
          features = comm.recv(source=proc, tag=55)
          rf.send(features)

    for proc in xrange(1, size):
      comm.Send([stop_iteration, MPI.INT], dest=proc, tag=1)
    rf.close()

  else:    
    while True:
      #Recieve signal to stop/contine execution
      comm.Recv([stop_iteration, MPI.INT], source=0, tag=1)
      if stop_iteration[0]:
        break

      #Recieve data shape and make buffer for data matrix
      comm.Recv([data_shape, MPI.INT], source=0, tag=2)
      data = np.array([np.zeros(data_shape[1], dtype=np.int32) for i in xrange(data_shape[0])], dtype=np.int32)
      comm.Recv([fname, MPI.SIGNED_CHAR], source=0, tag=3)
      comm.Recv([data, MPI.INT], source=0, tag=4)

      #Make and send feature set
      fe = FeatureExtractor(data)
      fe.set_features(fname[0])
      comm.send(fe.features, dest=0, tag=55)

  print "Closing process of rank %s" % rank
      
if __name__ == '__main__':
  main()
