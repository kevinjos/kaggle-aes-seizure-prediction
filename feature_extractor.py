#!/usr/bin/env python2.7

from mne import utils
utils.set_log_file(fname = '/home/kjs/logs/mne.log', overwrite=False)
utils.set_log_level(verbose = True)
utils.logger.propagate = True
LOGGER = utils.logger

import numpy as np
from scipy.stats import skew, kurtosis
import pyeeg
import file_handler
import csv
import time
from mne.filter import high_pass_filter, band_stop_filter
import random
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
    self.frequency = 0.0
    self.features = {}
    self.fod = {}
    self.svd_embed_seq = {}
    self.fft = {}
    self.power_ratio = {}
    self.data = data

  def set_features(self, filen):
    self.set_filen(filen)
    self.apply_frequency()
    self.set_medianval()
    self.apply_first_order_diff()
    self.apply_artifact_removal()
    self.apply_fft()
    self.apply_filters()
    self.apply_fft()
    self.apply_first_order_diff()
    self.apply_svd_embed_seq()
    self.apply_power_ratio()
    self.set_hjorth()
    self.set_svd_entropy()
    self.set_fisher_info()
    self.set_meanval()
    self.set_dfa()
    self.set_pfd()
    self.set_hjorth()
    self.set_higuchi()
    self.set_spectral_entropy()
    self.set_line_length()
    self.set_pfr()
    self.set_skew()
    self.set_kurtosis()

  def set_filen(self, filen):
    self.features['filen'] = filen.split('.')[0]

  def apply_frequency(self):
    if self.features['filen'].find('Dog') > -1:
      self.frequency = 400.0
    elif self.features['filen'].find('Patient') > -1:
      self.frequency = 5000.0
    elif self.features['filen'].find('fakefile') > -1:
      self.frequency = 128.0

  def apply_filters(self):
    Fs = int(self.frequency)
    Nf = int(Fs/2)
    lr = np.array([(x*60)-2 for x in range(1, Nf/60+1)], dtype=np.float64)
    hr = np.array([(x*60)+2 for x in range(1, Nf/60+1)], dtype=np.float64)
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      self.data[e] = high_pass_filter(x = self.data[e], Fs = Fs, Fp = 0.4, 
                                      trans_bandwidth = 0.05, copy = False,
                                      filter_length=int(np.floor(Fs*32)), verbose=False)
      '''
      efft = self.fft['fft_e' + str(e)]
      line_ratio = (np.mean(efft[:,60])/np.mean(efft[:,50:58]) + 
                    np.mean(efft[:,60])/np.mean(efft[:,63:70])) / 2
      if line_ratio > 1.5:
      '''
      self.data[e] = band_stop_filter(x = self.data[e], Fs = Fs, Fp1 = lr, Fp2 = hr, 
                                        copy = False, verbose=False)

  def apply_fft(self):
    '''
    Take FFT every second
    Returns np array of arrays [[fft_1s],
                                [fft_2s], ...
                                [fft_ns]]
    '''
    window = int(self.frequency)
    Nf = window/2
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'fft_e' + str(e)
      value = np.array([], dtype=np.float64)
      for j in xrange(0, int(self.data[e].size), window):
        ffte = np.fft.rfft(np.array(self.data[e][j:j+window], dtype=np.float64))[1:]
        if j == 0 and ffte.size == Nf:
          value = np.append(value, ffte)
        elif j == window and ffte.size == Nf:
          value = np.append([value], [ffte], axis=0)
        elif ffte.size == Nf:
          value = np.append(value, [ffte], axis=0)
      self.fft[name] = np.abs(value)

  def apply_first_order_diff(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'e' + str(e)
      value = pyeeg.first_order_diff(self.data[e])
      self.fod[name] = value

  def apply_power_ratio(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'power_ratio_e' + str(e)
      fft = self.fft['fft_e' + str(e)]
      value = sum(fft)/np.sum(fft)
      self.power_ratio[name] = value
  
  def set_medianval(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'median_e' + str(e)
      value = np.median(self.data[e])
      self.features[name] = value

  def apply_artifact_removal(self):
    loc_plateau = int(self.frequency/6)
    for e in xrange(self.data.shape[0]):
      local_i = -1
      points_in_plateau = 0
      fod = self.fod['e' + str(e)]
      for i in xrange(len(fod)):
        local = 0
        if fod[i] == 0 and i > local_i:
          local_i = i + 1
          local += 2
          for local_i in xrange(local_i, len(fod) - 1):
            if fod[local_i] == 0:
              local += 1
            elif local >= loc_plateau:
              points_in_plateau += local
              self.data[e, i-1:local_i-1] = self.features['median_e' + str(e)]
              break
            else:
              break
      perc_plateau = float(points_in_plateau) / float(self.data.shape[1])
      if perc_plateau >= 0.05:
        print "Dropping Channel %s in %s" % (e, self.features['filen'])
        self.data[e] = np.zeros(1)
      else:
        name = 'percent_plateau_e' + str(e)
        value = perc_plateau
        self.features[name] = value

  def apply_svd_embed_seq(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'svd_embed_seq_e' + str(e)
      value = pyeeg.embed_seq(self.data[e], 4, 20)
      value = np.linalg.svd(value, compute_uv = 0)
      value /= sum(value)
      self.svd_embed_seq[name] = value

  def set_svd_entropy(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'svd_entropy_e' + str(e)
      value = pyeeg.svd_entropy(W = self.svd_embed_seq['svd_embed_seq_e' + str(e)])
      self.features[name] = value

  def set_fisher_info(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'fisher_e' + str(e)
      value = pyeeg.fisher_info(W = self.svd_embed_seq['svd_embed_seq_e' + str(e)])
      self.features[name] = value

  def set_meanval(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'mean_e' + str(e)
      value = np.mean(self.data[e])
      self.features[name] = value
  
  def set_dfa(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'dfa_e' + str(e)
      value = pyeeg.dfa(self.data[e], Ave = self.features['mean_e' + str(e)])
      self.features[name] = value
  
  def set_pfd(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'pfd_e' + str(e)
      value = pyeeg.pfd(self.data[e], self.fod['e' + str(e)])
      self.features[name] = value

  def set_spectral_entropy(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'spectral_entropy_e' + str(e)
      pr = self.power_ratio['power_ratio_e' + str(e)]
      value = pyeeg.spectral_entropy(self.data[e], range(len(pr)), int(self.frequency), pr)
      self.features[name] = value

  def set_hjorth(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      mob_name = 'hjorthmob_e' + str(e)
      com_name = 'hjorthcom_e' + str(e)
      mobility_value, complexity_value = pyeeg.hjorth(self.data[e], self.fod['e' + str(e)])
      self.features[mob_name] = mobility_value
      self.features[com_name] = complexity_value
    
  def set_higuchi(self):
    kmax = 4
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'higuchi_e' + str(e)
      value = pyeeg.hfd(self.data[e], kmax)
      self.features[name] = value

  def set_line_length(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'line_length_e' + str(e)
      fod = self.fod['e' + str(e)]
      value = np.sum(np.sqrt(1+x**2) for x in fod)/fod.size
      self.features[name] = value

  def set_pfr(self):
    nf = int(self.frequency/2)
    def isprime(number):  
      if number<=1:  
        return 0  
      check=2  
      maxneeded=number  
      while check<maxneeded+1:  
        maxneeded=number/check  
        if number%check==0:  
            return 0  
        check+=1  
      return 1 
    primes = [x for x in xrange(nf) if isprime(x)]
    nprimes = [x for x in xrange(nf) if not isprime(x)]
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'pfr_e' + str(e)
      power_ratio = self.power_ratio['power_ratio_e' + str(e)]
      value = np.sum(power_ratio[primes])/np.sum(power_ratio[nprimes])
      self.features[name] = value

  def set_amp_diff(self):
    data = np.delete(fe.data, [x for x in range(fe.data.shape[0]) if np.all(fe.data[x]==0)], 0)
    mn, mx, ave, sd, snr = self.calc_stats(np.std(data[:,x]) for x in xrange(data.shape[1]))
    name = 'mean_amp_diff'
    self.features[name] = ave
    name = 'max_amp_diff'
    self.features[name] = mx
    name = 'min_amp_diff'
    self.features[name] = mn
    name = 'std_amp_diff'
    self.features[name] = std
    name = 'snr_amp_diff'
    self.features[name] = snr

  def set_skew(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'skew_e' + str(e)
      value = skew(self.data[e])
      self.features[name] = value
    
  def set_kurtosis(self):
    for e in xrange(self.data.shape[0]):
      if np.all(self.data[e][:1000] == 0):
        continue
      name = 'kurtosis_e' + str(e)
      value = kurtosis(self.data[e])
      self.features[name] = value

  def calc_stats(gen):
    '''
    Input: generator
    Returns: min, max, mean, std and snr
    '''
    minimum, maximum, total, count = 2**16, -2**16, 0, 0
    for x in gen:
      minimum = min(minimum, x)
      maximum = max(maximum, x)
      total += x
      total_squared += x*x
      count += 1
    mean = total/count
    standard_deviation = np.sqrt(total_squared/count - total*total/count/count)
    snr = mean/standard_deviation
    return (minimum, maximum, mean, standard_deviation, snr)

@coroutine
def record_features(fieldnames=[]):
  feature_file = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/20141111features.csv'
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
                                for i in range(1024)], dtype=np.float64) 
                                for i in range(max_e)], dtype=np.float64)
  fe = FeatureExtractor(fakedata)
  fe.set_features('fakefile')
  names = fe.features.keys()
  del fe
  return names

def sort_files(fh):
  '''
  memory cannot handle doing 5x samples at 5000Hz at once.
  stagger 5000Hz samples
  '''
  #Prepare for file I/O
  fh.set_train_interical_preictal_and_test_files()
  small_train_files = fh.seg_train_files['Dog']
  small_train_files.extend(fh.seg_test_files['Dog'])
  large_train_files = fh.seg_train_files['Patient']
  large_train_files.extend(fh.seg_test_files['Patient'])
  train_files = []
  ratio = int(np.floor(len(small_train_files)/len(large_train_files))) - 1
  for i in range(len(small_train_files)):
    train_files.append(small_train_files[i])
    if i % ratio == 0 and int(i/ratio) < int(np.floor(len(large_train_files))):
      train_files.append(large_train_files[int(i/ratio)])
  with open('/home/kjs/repos/kaggle-aes-seizure-prediction/data/archive/20141101features.csv') as f:
    dw = csv.DictReader(f)
    done_files = set()
    for line in dw:
      done_files.add(line['filen'])
  unfinished_files = []
  for f in train_files:
    fm = f.split('.')[0].strip()
    if fm not in done_files:
      unfinished_files.append(f)
  train_files = unfinished_files    
  train_files = np.array([np.array([f]) for f in train_files])
  print len(train_files)
  return train_files

def main():

  #Prepare for MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Make MPI buffers
  fname = np.array([''.zfill(37)])
  stop_iteration = np.zeros(1)
  data_shape = np.zeros(2, dtype=np.int32)

  if rank == 0:
    fh = file_handler.FileHandler()
    train_files = sort_files(fh)
    #Prepare tabular feature file header
    feature_names = get_feature_names()
    rf = record_features(feature_names)

    #Setup counters to be sure all files are accounted for
    total_responses = 0
    expected_responses = train_files.size
    for i in xrange(expected_responses):
      del fh
      fh = file_handler.FileHandler()
      #Prepare the data for MPI C type send
      fh.file_in = train_files[i][0]
      fh.set_data()
      electrodes = fh.data[0].shape[0]
      data = np.array([np.array(fh.data[0][e], dtype=np.float64) for e in range(electrodes)],
                       dtype=np.float64)

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
      LOGGER.info("%s, %s of %s, sent to process rank %s" % (fname[0], i, train_files.size, proc))

    #Cleanly exit execution by sending stop iteration signal to workers and closing feature record generator
    stop_iteration[0] = 1
    while total_responses != expected_responses:
      time.sleep(1)
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
      features = fe.features
      del fe
      comm.send(features, dest=0, tag=55)

  LOGGER.info("Closing process of rank %s" % rank)
      
if __name__ == '__main__':
  main()

