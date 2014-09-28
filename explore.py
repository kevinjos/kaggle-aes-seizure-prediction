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
from itertools import islice
from collections import deque
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import file_handler
import objgraph
import mne

mne.set_log_level('DEBUG')

class Cine(object):
  def __init__(self):
    self.filehandler = file_handler.FileHandler()
    self.app = QtGui.QApplication(sys.argv)
    self.widget = QtGui.QWidget()
    self.start_btn = QtGui.QPushButton('Start')
    self.stop_btn = QtGui.QPushButton('Stop')
    self.file_in = QtGui.QInputDialog()
    self.layout = QtGui.QGridLayout()
    self.layout.addWidget(self.start_btn, 0, 0)
    self.layout.addWidget(self.stop_btn, 0, 1)
    self.widget.setLayout(self.layout)
    self.start_btn.clicked.connect(self.start)
    self.stop_btn.clicked.connect(self.stop)
    self.widget.show()

  def cons_plots(self, num_channels):
    for i in range(num_channels):
      i = str(i)
      setattr(self, 'rawplot_' + i, pg.PlotWidget())
      getattr(self, 'rawplot_' + i).setRange(yRange=(200, -200))
      setattr(self, 'fftplot_' + i, pg.PlotWidget())
      getattr(self, 'fftplot_' + i).setRange(yRange=(0, 120))
      self.layout.addWidget(getattr(self, 'rawplot_' + i))
      self.layout.addWidget(getattr(self, 'fftplot_' + i))

  def start(self):
    file_in, file_provided = self.file_in.getText(self.widget, "", "what film?")
    if file_in:
      self.filehandler.file_in = str(file_in)
    self.filehandler.set_data()
    num_channels = self.filehandler.data[0].shape[0]
    num_samples = self.filehandler.data[0].shape[1]
    self.make_iterators(num_channels)
    self.plot(num_channels, num_samples)

  def stop(self):
    sys.exit()
    self.x_time.clear()
    self.y_val.clear()

  def make_iterators(self, num_channels):
    Fs = self.filehandler.frequency[0]
    Nf = int(Fs/2)
    lr = np.array([(x*60)-2 for x in range(1, Nf/60+1)], dtype=np.float64)
    hr = np.array([(x*60)+2 for x in range(1, Nf/60+1)], dtype=np.float64)
    for i in range(num_channels):
      self.filehandler.data[0][i] = mne.filter.band_stop_filter(
                           x = np.float64(self.filehandler.data[0][i]), 
                           Fs = np.float64(Fs), Fp1 = lr, Fp2 = hr, 
                           copy=True, verbose='DEBUG', n_jobs='cuda')
      self.filehandler.data[0][i] = mne.filter.high_pass_filter(
                           x = np.float64(self.filehandler.data[0][i]), 
                           Fs = np.float64(Fs), Fp = 0.4, trans_bandwidth = 0.05,
                           filter_length=int(np.floor(Fs*32)),
                           copy=True, verbose='DEBUG', n_jobs='cuda')
      setattr(self, 'rawchan_' + str(i), self.filehandler.data[0][i].flat)
      
  def do_fft(self, inputa):
    outputa = np.zeros(self.fft_size, dtype=complex)
    fft_plan = fftw3.Plan(inputa, outputa, direction='forward', flags=['estimate'])
    fft_plan.execute()
    return (np.log10(np.abs(outputa)) * 20)[:self.fft_size/2]

  def plot(self, num_channels, num_samples):
    #initialize parameters, graphs, and collections for plotting
    self.cons_plots(num_channels)

    self.fft_size = int(self.filehandler.frequency[0])
    time_scale = self.fft_size * 1
    bins = [i for i in xrange(self.fft_size/2)]
    x_time = deque([0], time_scale)
    y_val = [deque([0], time_scale) for i in range(num_channels)]
    #iterate over the data and make plots
    for time_s in xrange(num_samples):
      x_time.append(time_s)
      for i in range(num_channels):
        y_val[i].append(getattr(self, 'rawchan_' + str(i)).next())
      if time_s % self.fft_size == 0 and len(x_time) == self.fft_size:
        for i in range(num_channels):
          outputa = self.do_fft(np.array(y_val[i], dtype=complex))
          getattr(self, 'fftplot_' + str(i)).plot(bins[:500], outputa[:500], clear=True)
          getattr(self, 'rawplot_' + str(i)).plot([x/5000.0 for x in x_time], y_val[i], clear=True)
        self.app.processEvents()

if __name__ == '__main__':
  cine = Cine()
  cine.app.exec_()
    

