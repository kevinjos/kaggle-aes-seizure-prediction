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
import numpy as np
import os
import sys
import fftw3
import decs
from collections import deque
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

class FileHandler(object):
  DATA_DIR = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/'
  if not os.path.exists(DATA_DIR):
    DATA_DIR = None
    print "Configure the data directory to match local directory structure"

  def __init__(self):
    self.data = self.get_data(self.DATA_DIR + 'Dog_1/Dog_1_preictal_segment_0001.mat')

  def get_data(self, matfile):
    with open(matfile) as f:
      mat = scipy.io.loadmat(f)
      if mat.keys()[2].find('segment') > -1:
        data = mat[mat.keys()[2]]
      else:
        print "Key index is off for this data file. Write a better method to find the correct key for the data."
    return data[0,0]

  def get_channel(self, channels=[]):
    try:
      channel_data = self.data[0][channels[0], ]
    except IndexError:
      channel_data = None
      print "Channel does not exist, choose a channel betwee 0 and %s" % (data[0].shape[0] - 1)
    return channel_data

class Cine(object):
  def __init__(self):
    self.app = QtGui.QApplication(sys.argv)
    self.widget = QtGui.QWidget()
    self.start_btn = QtGui.QPushButton('Start')
    self.stop_btn = QtGui.QPushButton('Stop')
    self.rawplot = pg.PlotWidget()
    self.fftplot = pg.PlotWidget()
    self.rawplot.setRange(yRange=(500, -500))
    self.fftplot.setRange(yRange=(0, 90))
    self.layout = QtGui.QGridLayout()
    self.layout.addWidget(self.start_btn, 0, 0)
    self.layout.addWidget(self.stop_btn, 0, 1)
    self.layout.addWidget(self.rawplot, 1, 0, 2, 4)
    self.layout.addWidget(self.fftplot, 5, 0, 2, 4)
    self.widget.setLayout(self.layout)
    self.start_btn.clicked.connect(self.start)
    self.stop_btn.clicked.connect(self.stop)
    self.widget.show()
    self.x_time = deque([0], 1024)
    self.y_val = deque([0], 1024)

  def start(self):
    pass

  def stop(self):
    self.x_time.clear()
    self.y_val.clear()

  def do_fft(self, fft_mode):
    fft_size = 512
    bins = [i for i in range(fft_size/2)]
    while True:
      val, time_s = yield
      self.x_time.append(time_s)
      self.y_val.append(val)

  @decs.coroutine
  def plot(self, fft_mode):
    while True:
      val, time_s = yield
      self.x_time.append(time_s)
      self.y_val.append(val)
      if seq_num % 32 == 0 and len(self.raw_y) >= fft_size:
        y_val_list = list(self.raw_y)
        inputa = np.array(raw_y_list[-fft_size:], dtype=complex)
        #hann_window = np.hanning(fft_size)
        #inputa = inputa * hann_window
        #inputa = inputa * flattop_window
        outputa = np.zeros(fft_size, dtype=complex)
        if fft_mode == '1':
          fft = fftw3.Plan(inputa, outputa, direction='forward', flags=['estimate'])
          fft.execute()
          outputa = (np.log10(np.abs(outputa)) * 20)[:fft_size/2]
          self.fftplot.plot(bins, outputa, clear=True)
        self.rawplot.plot(self.raw_x, self.raw_y, clear=True)
        pg.QtGui.QApplication.processEvents()

if __name__ == '__main__':
  fh = FileHandler()
  channel_data = fh.get_channel(channels=[0])
  cine = Cine()
  cine.app.exec_()


