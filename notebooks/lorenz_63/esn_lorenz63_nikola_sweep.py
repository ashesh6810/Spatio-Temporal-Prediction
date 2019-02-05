import pyESN
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import genfromtxt
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data=genfromtxt('lorenz_63_modified.csv', delimiter=",")
dataT=np.transpose(data)
dataT=np.transpose(dataT)

res_arr = [500, 1000, 1500, 2000, 2500]
trn_arr = [20000, 30000, 40000, 50000, 57000]
# trn_arr = [20, 30, 40, 50]

def run(res, trn):
  esn = pyESN.ESN(
    n_inputs=1,
    n_outputs=3,
    n_reservoir=res,
    spectral_radius=0.9,
    sparsity=0.002,
    random_state=42)

  spin_off = 0
  trainN = trn
  testN = 2000
  sol_matrix = np.zeros([testN, 3])
  np.shape(sol_matrix)


  esn.fit(np.ones(trainN - spin_off), dataT[spin_off:trainN,:])

  prediction = esn.predict(np.ones(testN))

  error_mat=prediction-dataT[trainN:trainN + testN]
  horizon = 0
  while (horizon < testN) and \
    (abs(error_mat[horizon, 0]) < 0.5) and \
    (abs(error_mat[horizon, 1]) < 0.5) and \
    (abs(error_mat[horizon, 2]) < 0.5):
    horizon += 1

  return (np.sum(np.sum(np.square(error_mat[:horizon])))) / float(np.sum(np.sum(np.square(dataT[trainN:trainN + horizon])))), horizon


res = res_arr[rank % 5]
trn = trn_arr[rank / 5]

rmse_avrg = 0
hor_avrg = 0
for _ in range(7):
  rmse, hor = run(res, trn)
  rmse_avrg += rmse
  hor_avrg += hor
rmse_avrg /= 7.0
hor_avrg /= 7.0
print("reservoirs:", res, "- trainN:", trn, "- RMSE:", rmse_avrg, "- horizon:", hor_avrg)

rank2 = rank + size
if rank2 < 25:
  res = res_arr[rank2 % 5]
  trn = trn_arr[rank2 / 5]

  rmse_avrg = 0
  hor_avrg = 0
  for _ in range(7):
    rmse, hor = run(res, trn)
    rmse_avrg += rmse
    hor_avrg += hor
  rmse_avrg /= 7.0
  hor_avrg /= 7.0
  print("reservoirs:", res, "- trainN:", trn, "- RMSE:", rmse_avrg, "- horizon:", hor_avrg)
