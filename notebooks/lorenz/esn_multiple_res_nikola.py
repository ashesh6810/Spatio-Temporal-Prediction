# run using:
# mpiexec -n 40 python esn_multiple_res_nikola.py
import matplotlib
matplotlib.use('Agg')

import pyESN
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from mpi4py import MPI
import time

start_time = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

per_proc = 40 / size
additional = 40 % size

data = genfromtxt('lorenz_data_F8_new_std.csv', delimiter=",")
dataT = np.transpose(data)
#np.shape(dataT)

esn = pyESN.ESN(
  n_inputs=1,
  n_outputs=1,
  n_reservoir=5000,
  # n_reservoir=50,  # lover number for debugging on laptop
  spectral_radius=0.9,
  sparsity=0.002,
  random_state=42)

trainN = 50000
# trainN = 10000  # lover number for debugging on laptop
testN = 2000
# testN = 400  # lover number for debugging on laptop

samples = per_proc + (1 if rank < additional else 0)
start_sample = per_proc * rank + min(additional, rank)

prediction = np.zeros([testN, samples])

for ptr in range(start_sample, start_sample + samples):
  pred_training = esn.fit(np.ones(trainN), dataT[0 : trainN, ptr])
  prediction[:,ptr - start_sample] = esn.predict(np.ones(testN)).flatten()
  print(rank, ptr, "test error: \n" + str(np.sqrt(np.mean((prediction[:,ptr - start_sample] - dataT[trainN:trainN+testN, ptr]) ** 2))))

prediction_trans = np.transpose(prediction)

res = np.array(comm.gather(prediction_trans, root=0))

if rank == 0:
  sol_matrix = np.zeros([testN, 40])
  ind = 0
  for mat in res:
    for arr in mat:
      sol_matrix[:,ind] = arr
      ind += 1

  ptr = 2
  plt.figure(figsize=(15, 1.5))
  plt.plot(range(0, trainN + testN), dataT[0 : trainN + testN, ptr], 'k', label="target system")
  plt.plot(range(trainN, trainN + testN), sol_matrix[:,ptr], 'r', label="free running ESN")
  lo, hi = plt.ylim()
  plt.plot([trainN, trainN], [lo + np.spacing(1), hi - np.spacing(1)], 'k:')
  plt.legend(loc=(0.61, 1.1), fontsize='x-small')

  plt.savefig('example_feature2.png')



  plt.figure(figsize=(400, 100))
  for ptr in range(0, 40):
    plt.subplot(8, 5, ptr + 1)
    
    plt.plot(range(trainN, trainN + testN), dataT[trainN : trainN + testN, ptr], 'k', label="target system")
    plt.plot(range(trainN, trainN + testN), sol_matrix[:,ptr], 'r', label="free running ESN")
    lo, hi = plt.ylim()
    plt.plot([trainN, trainN], [lo + np.spacing(1), hi - np.spacing(1)], 'k:')
    plt.legend(loc=(0.61, 1.1), fontsize='x-small')

  plt.savefig('panel_plot.png')

  print("execution time:", time.time() - start_time)

