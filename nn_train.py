import csv_to_pandas as cp
import nn
from mpi4py import MPI
import numpy as np
import time
import cPickle as pickle

def main():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  if rank == 0:
    df = cp.get_df()
    data = cp.convert_data(df)
    models = {}
    i = 1
    jobs_total = 7
    jobs_complete = 0
    subject = list(data.keys())
    while jobs_complete < jobs_total:

      if i < size:
        comm.send(False, dest = i, tag = 777)
        comm.send(data[subject.pop()]['train'], dest = i, tag = 1)
      else:
        r = 1
        while not comm.Iprobe(source = r, tag = 11):
          time.sleep(1)
          r = (r % (size - 1)) + 1
        trained_nn = comm.recv(source = r, tag = 11)
        nn_subject = comm.recv(source = r, tag = 22)
        models[nn_subject] = trained_nn
        jobs_complete += 1
        print "root jobs complete == %s" % jobs_complete
        if subject:
          comm.send(False, dest = r, tag = 777)
          comm.send(data[subject.pop()]['train'], dest = r, tag = 1)
      
      i += 1

    for i in range(1, size):
      comm.send(True, dest = i, tag = 777)
    f = open('/home/kjs/repos/kaggle-aes-seizure-prediction/data/nn.cpick', 'wb')
    pickle.dump(models, f)

  else:
    while True:
      while not comm.Iprobe(source = 0, tag = 777):
        time.sleep(5)
      stop_iteration = comm.recv(source = 0, tag = 777)
      if stop_iteration:
        break
      df = comm.recv(source = 0, tag = 1)
      trained_net = nn.nn_routine(df)
      comm.send(trained_net, dest = 0, tag = 11)
      comm.send(df.subject[0], dest = 0, tag = 22)

if __name__ == '__main__':
  main()
