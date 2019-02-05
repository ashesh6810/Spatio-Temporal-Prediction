import matplotlib
matplotlib.use('Agg')
import pyESN_BE
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import time

start_time = time.time()

data=genfromtxt('lorenz_data_F8_new_std.csv',delimiter=",")
dataT=np.transpose(data)
np.shape(dataT)

esn = pyESN_BE.ESN(
  n_inputs=1,
  n_outputs=1,
  n_reservoir=5000,
  # n_reservoir=50,  # lover number for debugging on laptop
  spectral_radius=0.9,
  sparsity=0.002,
  reg=0.001,
  random_state=42)

trainN = 50000
testN = 2000
sol_matrix=np.zeros([testN,40])
np.shape(sol_matrix)

print("preparation time:", time.time() - start_time)
start_time = time.time()

for ptr in range(0,40):
 pred_training = esn.fit(np.ones(trainN),dataT[0:trainN,ptr])
 prediction = esn.predict(np.ones(testN))
 print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - dataT[trainN:trainN+testN,ptr])**2))))

 sol_matrix[:,ptr]=np.transpose(prediction)
 break


# np.shape(sol_matrix)

print("execution time:", time.time() - start_time)

"""
ptr=2;
plt.figure(figsize=(15,1.5))
plt.plot(range(0,trainN+testN),dataT[0:trainN+testN,ptr],'k',label="target system")
plt.plot(range(trainN,trainN+testN),sol_matrix[:,ptr],'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainN,trainN],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')

plt.savefig('example_feature2.png')



plt.figure(figsize=(400,100))
for ptr in range(0,40):
 plt.subplot(8,5,ptr)
 plt.plot(range(trainN,trainN+testN),dataT[trainN:trainN+testN,ptr],'k',label="target system")
 plt.plot(range(trainN,trainN+testN),sol_matrix[:,ptr],'r', label="free running ESN")
 lo,hi = plt.ylim()
 plt.plot([trainN,trainN],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
 plt.legend(loc=(0.61,1.1),fontsize='x-small')

plt.savefig('panel_plot.png')
"""