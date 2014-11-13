#!/usr/bin/env python2.7
'''
data structure is a follows:
data[0] == channel # x sample count array of raw data from 
data[1] == data_length_sec: the time duration of each data row
data[2] == sampling_frequency: the number of data samples representing 1 second of EEG data
data[3] == channels: a list of electrode names corresponding to the rows in the data field
data[4] == sequence: the index of the data segment within the one hour series of clips. For example, preictal_segment_6.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data.
'''

import scipy.io
import os
import numpy as np

class FileHandler(object):
  DATA_DIR = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/'
  if not os.path.exists(DATA_DIR):
    DATA_DIR = None
    print "Configure the data directory to match local directory structure"

  def __init__(self):
    #Also prompted by GUI to choose a file name (film)
    self.file_in = 'Patient_1_preictal_segment_0009.mat'
    self.file_in = 'Patient_1_test_segment_0001.mat'

  def set_data(self):
    self.file_in = "_".join(self.file_in.split("_")[0:2]) + "/" + self.file_in
    with open(self.DATA_DIR + self.file_in) as f:
      mat = scipy.io.loadmat(f)
      keys = mat.keys()
      i = [mat.keys().index(key) for key in mat.keys() if key.find('segment') > 0][0]
      data = mat[mat.keys()[i]]
    self.data = data[0,0]
    self.data_length_sec = self.data[1][0]
    self.frequency = self.data[2][0]
    self.electrode_names = self.data[3][0]
    if self.file_in.find('test') == -1:
      self.sequence_num = self.data[4]
    
  def append_file_name(self):
    self.data[3] = np.append(self.data[3], self.file_in.split("/")[1])

  def set_train_interical_preictal_and_test_files(self):
    '''
    self.segmented_train_files is a dictionary with two keys, interictal and preictal
    each key has a set for a value of all data from all subjects in the give category
    '''
    self.all_train_files, self.all_test_files = [], []
    self.segmented_train_files, self.seg_train_files = {}, {}
    self.segmented_test_files, self.seg_test_files = {}, {}
    P, I, D, H = 'preictal', 'interictal', 'Dog', 'Patient'
    self.segmented_train_files[I], self.segmented_train_files[P] = [], []
    self.seg_train_files[D], self.seg_train_files[H] = [], []
    self.segmented_test_files[I], self.segmented_test_files[P] = [], []
    self.seg_test_files[D], self.seg_test_files[H] = [], []
    
    subjects = os.listdir(self.DATA_DIR)
    for subject in subjects:
      if subject.find(D) > -1 or subject.find(H) > -1:
        self.all_train_files.extend([f for f in os.listdir(self.DATA_DIR + '/' + subject) if f.find('test') == -1])
        self.all_test_files.extend([f for f in os.listdir(self.DATA_DIR + '/' + subject) if f.find('test') != -1])
    for f in self.all_train_files:
      if f.find(I) > -1:
        self.segmented_train_files[I].append(f)
      elif f.find(P) > -1:
        self.segmented_train_files[P].append(f)
      if f.find(D) > -1:
        self.seg_train_files[D].append(f)
      elif f.find(H) > -1:
        self.seg_train_files[H].append(f) 
    for f in self.all_test_files:
      if f.find(D) > -1:
        self.seg_test_files[D].append(f)
      elif f.find(H) > -1:
        self.seg_test_files[H].append(f) 

  def arrange_files_by_subject_by_type(self):
    self.set_train_interical_preictal_and_test_files()
    subjects = set()
    for fname in self.all_train_files:
      subjects.add("_".join(fname.split("_")[:2]))
    res = {}
    all_files = self.all_train_files
    all_files.extend(self.all_test_files)
    for subject in subjects:
      res[subject] = {'train':set(), 'test':set()}
      for filen in all_files:
        if filen.find(subject) > -1:
          if filen.find('test') > -1:
            res[subject]['test'].add(filen)
          else:
            res[subject]['train'].add(filen)
    return res








