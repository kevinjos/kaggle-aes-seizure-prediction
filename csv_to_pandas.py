import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import TanhLayer

def get_df():
  f = '/home/kjs/repos/kaggle-aes-seizure-prediction/data/features.csv'
  df = pd.read_csv(f)
  return df

def convert_data(df):
  testset = df['filen'].str.contains('test')
  preictal = df['filen'].str.contains('preictal')
  subject=['_'.join(f.split('_')[:2]) for f in df.loc[:,'filen']]
  df.insert(0, 'target', preictal)
  df.insert(0, 'test', testset)
  df.insert(0, 'subject', subject)
  '''
  data == dictionary of dictionaries
  {'SUBJECT_0':{'TEST':df, 'TRAIN':df}, ...
   'SUBJECT_N:{'TEST':df, 'TRAIN':df}}
  '''
  data = {}
  for sub in set(subject):
    data[sub] = {'test':df.query('test and subject==@sub'), 
                 'train':df.query('~test and subject==@sub')}
    data[sub]['test'] = data[sub]['test'].dropna(axis=1).reset_index()
    data[sub]['train'] = data[sub]['train'].dropna(axis=1).reset_index()
  return data

def drop_meta_data(data):
  for sub in data:
    t = 'train'
    for c in data[sub][t]:
      if data[sub][t][c].dtype == 'O':
        data[sub][t].drop(c, axis=1, inplace=True)
        data[sub][t].reset_index()
  return data

def plt_gaussian(df, sub, t, col):
  '''
  X is a one dimensional np.array
  Y is a one dimensional np.array for the gaussian response
  mu is the mean of X
  sigma is the standard deviaion of X
  '''
  sqrtTwoPi = np.sqrt(2*np.pi)
  plt.figure()
  plt.title("Measure %s, Subject %s, dataset %s" % (col, sub, t))
  for col in df:
    X = np.array(df[col])
    X.sort()
    Y = np.zeros(X.size)
    mu = np.mean(X)
    sigma = np.std(X)
    variance = sigma**2
    for i in range(X.size):
      Y[i] = (1/(sigma * sqrtTwoPi)) * np.e**-((X[i] - mu)**2/(2 * variance))
    assert X.size == Y.size
    plt.plot(X, Y, hold=True)
  plt.show()
'''
col = 'fisher'
sub = 'Patient_1'
df = data[sub]['train'].query('interictal')
df_test = df[df.columns[df.columns.map(lambda x: col in x)]].dropna(axis = 1)
df = data[sub]['train'].query('~interictal')
df_train = df[df.columns[df.columns.map(lambda x: col in x)]].dropna(axis = 1)
plt_gaussian(df_test, sub, 'interictal', col)
plt_gaussian(df_train, sub, 'preictal', col)
'''

def main():
  df = get_df()
  data = convert_data(df)
  df = data['Dog_1']['train']
  nn = nn_routine(df)
  return nn

if __name__ == '__main__':
  main()
