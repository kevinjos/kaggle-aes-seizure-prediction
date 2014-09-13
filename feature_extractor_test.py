#!/usr/bin/env python2.7

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.array([np.zeros(1024, dtype=np.int16) for i in range(16)], dtype=np.int16)
data_back = np.array([np.zeros(1024, dtype=np.int16) for i in range(16)], dtype=np.int16)
print "arrays created in rank %s" % rank

if rank == 0:
  print "rank %s active" % rank
  data = np.array([np.array(range(1024), dtype=np.int16) for i in range(16)], dtype=np.int16)
  comm.Send([data, MPI.INT], dest=1)
  comm.Recv([data_back, MPI.INT], source=1)
  assert (data == data_back).all()
  print "assert passed"
  print data.dtype
  print data_back.dtype
else:
  print "rank %s active" % rank
  comm.Recv([data, MPI.INT], source=0)
  comm.Send([data, MPI.INT], dest=0)
