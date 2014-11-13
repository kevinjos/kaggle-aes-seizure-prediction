import file_handler as fh
import cPickle as pickle
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from arac.pybrainbridge import _RecurrentNetwork, _FeedForwardNetwork
from pybrain.structure import FullConnection

def nn_routine_df(df):
  df_features = df.select_dtypes(include=['float64'])
  target = df['target']
  feasize = df_features.shape[1]
  ds = SupervisedDataSet(feasize, 1)
  for sample in df_features.iterrows():
    ds.addSample(np.array(sample[1]), np.array(target[sample[0]]))
  return nn_routine_ds(ds)

def nn_routine_ds(ds):
  feasize = ds['input'].shape[1]
  trainDS, testDS = ds.splitWithProportion(0.8)
  net = _FeedForwardNetwork()

  inLayer = LinearLayer(feasize) 
  net.addInputModule(inLayer)
  net, hl = add_fullyC_layer(net, inLayer, TanhLayer(feasize*8))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize*4))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize*2))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize/2))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize/4))
  net, hl = add_fullyC_layer(net, hl, TanhLayer(feasize/8))
  outLayer = SigmoidLayer(1)
  net.addOutputModule(outLayer)
  hidden_last_to_out = FullConnection(hl, outLayer)
  net.addConnection(hidden_last_to_out)

  net.sortModules()
  trainer = BackpropTrainer(net, trainDS, learningrate=0.001, momentum=.1, verbose=True)
  epochs = 10000
  for zzz in range(epochs):
    if zzz % 100 == 0:
      print "%2.3f percent complete" % (100.*zzz/epochs)
    trainer.train()
  res = np.append(testDS['target'], net.activateOnDataset(testDS), axis=1)
  return (net, trainer, trainDS, testDS, res)

def add_fullyC_layer(net, layer_p, layer_n):
  net.addModule(layer_n)
  p_to_n = FullConnection(layer_p, layer_n)
  net.addConnection(p_to_n)
  return net, layer_n

def train_xor():
  net = buildNetwork(2, 6, 1)
  ds = SupervisedDataSet(2, 1)
  ds.addSample((0, 0), (0,))
  ds.addSample((0, 1), (1,))
  ds.addSample((1, 0), (1,))
  ds.addSample((1, 1), (0,))
  trainer = BackpropTrainer(net, ds)
  for i in range(10000):
    trainer.train()
  return (net, ds, trainer)

def get_raw_train_ds_by_subject(s):
  FH = fh.FileHandler()
  filens = FH.arrange_files_by_subject_by_type()
  sfiles = filens[s]
  train_s = sfiles['train']
  f = train_s.pop()
  FH.file_in = f
  FH.set_data()
  data = FH.data[0]
  ds = SupervisedDataSet(data.shape[1], 1)
  for electrode in range(data.shape[0]):
    if FH.file_in.find('interictal') > -1:
      ds.addSample(np.array(data[electrode,:], dtype='int32'), np.array(0))
    elif FH.file_in.find('preictal') > -1:
      ds.addSample(np.array(data[electrode,:], dtype='int32'), np.array(1))
    else:
      "We've got a problem"
  i = 0
  while len(train_s) > 0:
    f = train_s.pop()
    FH.file_in = f
    FH.set_data()
    data = FH.data[0]
    for electrode in range(data.shape[0]):
      if FH.file_in.find('interictal') > -1:
        i += 1
        if i % 10 == 0:
          ds.addSample(np.array(data[electrode,:], dtype='int32'), np.array(0))
      elif FH.file_in.find('preictal') > -1:
        ds.addSample(np.array(data[electrode,:], dtype='int32'), np.array(1))
      else:
        "We've got a problem"
  return ds

def train_raw():
  ds = get_raw_train_ds_by_subject('Dog_1')
  return nn_routine_ds(ds)

if __name__ == '__main__':
  train_xor()
