#!/usr/bin/env python2.7

import numpy as np
import pyeeg
import file_handler
import csv
import time
import mne
import fftw3
import random
from decs import coroutine, timed
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
    self.frequency = 0.0
    self.features = {}
    self.fprime = {}
    self.bin_power = {}
    self.svd_embed_seq = {}
    self.fft = {}
    self.data = data

  @timed
  def set_features(self, filen):
    self.set_filen(filen)
    self.apply_frequency()
    self.apply_filters()
    self.apply_first_order_diff()
    self.apply_bin_power()
    self.apply_svd_embed_seq()
    #self.apply_fft()
    self.set_svd_entropy()
    self.set_fisher_info()
    self.set_meanval()
    self.set_dfa()
    self.set_pfd()
    self.set_hjorth()
    self.set_higuchi()
    self.set_spectral_entropy()
    #self.set_hurst()
    #self.set_samplesize()
    #self.set_firstval()
    #self.set_lastval()
    #self.set_maxval()
    #self.set_minval()

  def set_filen(self, filen):
    self.features['filen'] = filen.split('.')[0]

  def apply_frequency(self):
    if self.features['filen'].find('Dog') > -1:
      self.frequency = 400.0
    elif self.features['filen'].find('Patient') > -1:
      self.frequency = 5000.0
    elif self.features['filen'].find('fakefile') > -1:
      self.frequency = 128.0
    else:
      print "Oops, cannot find the correct frequencey for %s" % self.features['filen']

  def apply_filters(self):
    Fs = self.frequency
    Nf = int(Fs/2)
    lr = np.array([(x*60)-2 for x in range(1, Nf/60+1)], dtype=np.float64)
    hr = np.array([(x*60)+2 for x in range(1, Nf/60+1)], dtype=np.float64)
    for i in xrange(self.data.shape[0]):
      self.data[i] = mne.filter.band_stop_filter(
                      x = self.data[i], Fs = Fs, Fp1 = lr, Fp2 = hr,
                      copy=True, verbose='WARNING', n_jobs='cuda')
      self.data[i] = mne.filter.high_pass_filter(
                      x = self.data[i], Fs = Fs, Fp = 0.4, trans_bandwidth = 0.05, 
                      filter_length=int(np.floor(Fs*32)), copy = True, n_jobs = 'cuda',
                      verbose='WARNING')

  def apply_fft(self):
    '''
    Take FFT every second
    Returns np array of arrays [[fft_1s],
                                [fft_2s], ...
                                [fft_ns]]
    '''
    Fs = int(self.frequency)
    window = Fs
    Nf = window/2
    for i in xrange(self.data.shape[0]):
      name = 'fft_e' + str(i)
      value = np.array([], dtype=np.float64)
      for j in xrange(0, int(self.data[i].size), window):
        input = np.array(self.data[i][j:j+window], dtype=np.float64)
        output = np.zeros(input.size, dtype='complex')
        plan = fftw3.Plan(input, output)
        plan.execute()
        output = output[:int(window/2.0)]
        if j == 0 and output.size == Nf:
          value = np.append(value, output)
        elif j == window and output.size == Nf:
          value = np.append([value], [output], axis=0)
        elif output.size == Nf:
          value = np.append(value, [output], axis=0)
        else:
          continue
      self.fft[name] = np.abs(value)

  def apply_first_order_diff(self):
    for i in xrange(self.data.shape[0]):
      name = 'first_order_diff_e' + str(i)
      value = pyeeg.first_order_diff(self.data[i])
      self.fprime[name] = value

  def apply_bin_power(self):
    for i in xrange(self.data.shape[0]):
      name = 'bin_power_e' + str(i)
      power, power_ratio = pyeeg.bin_power(self.data[i], [0.5, 4, 7, 12, 30],
                                           int(self.frequency))
      self.bin_power[name] = (power, power_ratio)

  def apply_svd_embed_seq(self):
    for i in xrange(self.data.shape[0]):
      name = 'svd_embed_seq_e' + str(i)
      embed_seq = pyeeg.embed_seq(self.data[i], 2, 20)
      value = np.linalg.svd(embed_seq, compute_uv = 0)
      self.svd_embed_seq[name] = value

  def set_svd_entropy(self):
    for i in xrange(self.data.shape[0]):
      name = 'svd_entropy_e' + str(i)
      value = pyeeg.svd_entropy(W = self.svd_embed_seq['svd_embed_seq_e' + str(i)])
      self.features[name] = value

  def set_fisher_info(self):
    for i in xrange(self.data.shape[0]):
      name = 'fisher_e' + str(i)
      value = pyeeg.fisher_info(W = self.svd_embed_seq['svd_embed_seq_e' + str(i)])
      self.features[name] = value

  def set_meanval(self):
    for i in xrange(self.data.shape[0]):
      name = 'mean_e' + str(i)
      value = np.mean(self.data[i])
      self.features[name] = value
  
  def set_dfa(self):
    for i in xrange(self.data.shape[0]):
      name = 'dfa_e' + str(i)
      value = pyeeg.dfa(self.data[i], Ave = self.features['mean_e' + str(i)])
      self.features[name] = value
  
  def set_pfd(self):
    '''
    petrosian fractal dimension
    '''
    for i in xrange(self.data.shape[0]):
      name = 'pfd_e' + str(i)
      value = pyeeg.pfd(self.data[i], self.fprime['first_order_diff_e' + str(i)])
      self.features[name] = value

  def set_spectral_entropy(self):
    for i in xrange(self.data.shape[0]):
      name = 'spectral_entropy_e' + str(i)
      pr = self.bin_power['bin_power_e' + str(i)][1]
      band = [0.5, 4, 7, 12, 30]
      value = pyeeg.spectral_entropy(self.data[i], band, int(self.frequency), Power_Ratio=pr)
      self.features[name] = value

  def set_hjorth(self):
    for i in xrange(self.data.shape[0]):
      mob_name = 'hjorthmob_e' + str(i)
      com_name = 'hjorthcom_e' + str(i)
      mobility_value, complexity_value = pyeeg.hjorth(self.data[i], 
                                                    self.fprime['first_order_diff_e' + str(i)])
      self.features[mob_name] = mobility_value
      self.features[com_name] = complexity_value
    
  def set_higuchi(self):
    kmax = 4
    for i in xrange(self.data.shape[0]):
      name = 'higuchi_e' + str(i)
      value = pyeeg.hfd(self.data[i], kmax)
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

  '''
  test features for data consistency
  def set_samplesize(self):
    for i in xrange(self.data.shape[0]):
      name = 'samplesize_e' + str(i)
      value = str(self.data[i].size)
      self.features[name] = value

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
  '''

@coroutine
def record_features(fieldnames=[]):
  feature_file = '/home/kjs/repos/kaggle-aes-seizure-prediction/features.csv'
  try:
    f = open(feature_file, 'a')
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    while True:
      features = yield
      writer.writerow(features)
  except GeneratorExit:
    f.close()

def get_feature_names(max_e=24):
  fakedata = np.array([np.array([np.random.random_integers(-2**16, 2**16) 
                                for i in range(8192)], dtype=np.float64) 
                                for i in range(max_e)], dtype=np.float64)
  fe = FeatureExtractor(fakedata)
  fe.set_features('fakefile')
  return fe.features.keys()

def main():
  #Prepare for file I/O
  #train_files = fh.seg_train_files['Patient']
  fh = file_handler.FileHandler()
  fh.set_train_interical_preictal_and_test_files()
  train_files = fh.seg_train_files['Dog']
  train_files.extend(fh.seg_train_files['Patient'])
  test_files = fh.all_test_files
  train_files.extend(test_files)
  random.shuffle(train_files)
  train_files = np.array([np.array([f]) for f in train_files])

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
      #Prepare the data for MPI C type send
      fh.file_in = train_files[i][0]
      fh.set_data()
      data = np.array([np.array(fh.data[0][j], dtype=np.float64) 
                      for j in xrange(fh.data[0].shape[0])], dtype=np.float64)

      #Calculate the shape of the data to make the correctly sized buffer
      data_shape = np.array(data.shape, dtype=np.int32)

      #Set the file name to send
      fname = train_files[i]

      #Set proc to idle proc after initializing worker procs
      if i >= size-1:
        r = 1
        while not comm.Iprobe(source = r, tag = 55):
          time.sleep(1)
          r = (r % (size - 1)) + 1
        features = comm.recv(source = r, tag = 55)
        rf.send(features)
        total_responses += 1
        proc = r
      else:
        proc = i + 1

      #Send data
      comm.Send([stop_iteration, MPI.INT], dest=proc, tag=1)
      comm.Send([data_shape, MPI.INT], dest=proc, tag=2)
      comm.Send([fname, MPI.SIGNED_CHAR], dest=proc, tag=3)
      comm.Send([data, MPI.FLOAT], dest=proc, tag=4)
      print "%s, %s of %s, sent to process rank %s" % (fname[0], i, train_files.size, proc)

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
      data = np.array([np.zeros(data_shape[1], dtype=np.float64) 
                      for i in xrange(data_shape[0])], dtype=np.float64)
      comm.Recv([fname, MPI.SIGNED_CHAR], source=0, tag=3)
      comm.Recv([data, MPI.FLOAT], source=0, tag=4)

      #Make and send feature set
      fe = FeatureExtractor(data)
      fe.set_features(fname[0])
      comm.send(fe.features, dest=0, tag=55)

  print "Closing process of rank %s" % rank
      
if __name__ == '__main__':
  main()
