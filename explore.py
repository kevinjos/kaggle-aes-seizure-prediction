#!/usr/bin/env python2.7

import scipy.io

def get_data(matfile):
  with open(matfile) as f:
    mat = scipy.io.loadmat(f)
    if mat.keys()[2].find('segment') > -1:
      data = mat[mat.keys()[2]]
    else:
      print "Key index is off for this data file. Write a better method to find the correct key for the data."
  return data[0,0]

pdfile = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/Dog_1/Dog_1_preictal_segment_0001.mat'
data = get_data(pdfile)
'''
data structure is a follows:
data[0] == channel # x sample count array of raw data from 
data[1] == data_length_sec: the time duration of each data row
data[2] == sampling_frequency: the number of data samples representing 1 second of EEG data
data[3] == channels: a list of electrode names corresponding to the rows in the data field
data[4] == sequence: the index of the data segment within the one hour series of clips. For example, preictal_segment_6.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data.
'''
print data[0].shape
