#!/usr/bin/env python2.7

import feature_extractor as FE
import file_handler as FH
import pyeeg

fh = FH.FileHandler()
fh.file_in = 'Dog_2_interictal_segment_0010.mat'
fh.set_data()
fe = FE.FeatureExtractor(fh.data[0])
fe.set_features(fh.file_in)

def plot_hjorth():
  window = int(fe.frequency) * 30
  yc, ym = [], []
  for i in xrange(fe.data.shape[0]):
    Yc, Ym = [], []
    for j in range(window, fe.data.shape[1], window):
      dslice = fe.data[i, :j]
      fodsli = fe.fod['e' + str(i)][:j]
      hmob, hcom = pyeeg.hjorth(dslice, fodsli)
      Yc.append(hcom)
      Ym.append(hmob)
    yc.append(Yc)
    ym.append(Ym)
  X = range(window, fe.data.shape[1], window)
  plt.subplot(121)
  plt.title('hjorth complexity')
  plt.xlabel("Sample size window")
  for Y in yc:
    plt.plot(X, Y, hold=True)
  plt.subplot(122)
  plt.title('hjorth mobility')
  plt.xlabel("Sample size window")
  for Y in ym:
    plt.plot(X, Y, hold=True)
  plt.show()

def plot_pfs():
  plt.figure()
  Fs = int(fe.frequency)
  window = Fs
  nf = Fs/2
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
  for i in xrange(fe.data.shape[0]):
    Y = []
    X = range(window, fe.data.shape[1], window)
    for k in range(1, len(X)+1):
      fftsli = fe.fft['fft_e' + str(i)][:k, :]
      value = np.sum(fftsli[:,primes])/np.sum(fftsli[:,nprimes])
      Y.append(value)
    plt.plot(X, Y, hold = True)
  plt.title('prime frequency ratio')
  plt.xlabel("Sample size window")
  plt.show()

