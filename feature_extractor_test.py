#!/usr/bin/env python2.7

from mpi4py import MPI
import numpy as np

#Prepare MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Test sending a data matrix
data = np.array([np.zeros(1024, dtype=np.int64) for i in range(16)], dtype=np.int64)
data_back = np.array([np.zeros(1024, dtype=np.int64) for i in range(16)], dtype=np.int64)
if rank == 0:
  data = np.array([np.array(range(1024), dtype=np.int64) for i in range(16)], dtype=np.int64)
  comm.Send([data, MPI.INT], dest=1)
  comm.Recv([data_back, MPI.INT], source=1)
  assert (data == data_back).all()
  print "assert passed"
else:
  comm.Recv([data, MPI.INT], source=0)
  comm.Send([data, MPI.INT], dest=0)

'''
Test sending boolean
truth = np.array([True], dtype='bool')
if rank == 0:
  truth[0] = False
  comm.Send([truth, 1, MPI.BOOL], dest=1)
else:
  comm.Recv([truth, 1, MPI.BOOL], source=0)
  if not truth[0]:
    print "passed bool"
Fails. Should pass int instead if using Send/Recv. Can also pass True/False pickles with send/recv.
'''

#Recreate
#RuntimeWarning: overflow encountered in int_scalars pyeeg.py:333
data = np.array([2**32, 2**32], dtype = np.int64)
if data[1]*data[0] < 0:
  print "Overflows"


data = np.array([(-1+2**32), (2**31)], dtype = np.int64)
if data[1]*data[0] < 0:
  print "No overflow"
