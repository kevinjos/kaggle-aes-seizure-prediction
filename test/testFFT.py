#!/usr/bin/env python2.7

import mne
import fftw3
import numpy as np
import file_handler
from decs import timed

def main():
  fh = file_handler.FileHandler()
  fh.file_in = 'Patient_1_preictal_segment_0001.mat'
  fh.set_data()
  freq = fh.frequency[0]
  input = np.array(fh.data[0][0])
  return input
  

@timed
def fftw3_FFT(input, freq):
  res_f = np.array([], dtype=np.float64)
  res_c = np.array([], dtype='complex')
  for i in xrange(0, input.size, freq):
    input_c = np.array(input[i:i+freq], dtype='complex')
    output_c = np.zeros(freq, dtype='complex')
    plan = fftw3.Plan(input_c, output_c)
    plan.execute()
    input_f = np.array(input[i:i+freq], dtype=np.float64)
    output_f = np.zeros(freq, dtype='complex')
    plan = fftw3.Plan(input_f, output_f)
    plan.execute()
    if i == 0:
      res_f = np.append(res_f, output_f) #first time through
      res_c = np.append(res_c, output_c) #first time through
    elif i == freq:
      res_f = np.append([res_f], [output_f], axis=0) #second 
      res_c = np.append([res_c], [output_c], axis=0) #second 
    else:
      res_f = np.append(res_f, [output_f], axis=0) #third plus
      res_c = np.append(res_c, [output_c], axis=0) #third plus
  return (res_f, res_c)

def uber_fft_filter(npdatain, Fs):
  window = Fs * 2 #to get 0.5Hz resolution on the FFT
  output_a = np.zeros(window, dtype='complex')
  npdataout = np.array([], dtype=np.float64)
  for i in xrange(0, npdatain.size, window):
    input_a = np.array(npdatain[i:i+window], dtype=np.float64)
    plan_a = fftw3.Plan(input_a, output_a, direction='forward')
    plan_a.execute()
    input_b = output_a[:window/2]
    input_b[0], input_b[1], input_b[118], input_b[119], input_b[120], input_b[121] = 0, 0, 0, 0, 0, 0
    output_b = np.zeros(input_b.size, dtype=np.float64)
    plan_b = fftw3.Plan(input_b, output_b, direction='backward')
    plan_b.execute()
    npdataout = np.append(npdataout, output_b)
  return (np.log10(np.abs(npdataout)) * 20)

if __name__ == '__main__':
  main()

