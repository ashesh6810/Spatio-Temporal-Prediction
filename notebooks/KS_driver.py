import matplotlib
matplotlib.use('Agg')
import pyESN
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from matplotlib import cm

data=genfromtxt('KS_data.csv',delimiter=",")
dataT=np.transpose(data)


esn = pyESN.ESN(n_inputs = 1,
          n_outputs = 4,
          n_reservoir = 5000,
          spectral_radius = 0.95,sparsity=0.002, reg=0.0001,
          random_state=42)

spin_off=2000
trainN = 22000
testN = 2000
sol_matrix=np.zeros([testN,64])


count=0;
for ptr in range(0,16):
 pred_training = esn.fit(np.ones(trainN-spin_off),dataT[spin_off:trainN,count:count+4])
 prediction = esn.predict(np.ones(testN))
 sol_matrix[:,count:count+4]=(prediction)
 count=count+4;

plt.figure(figsize=(15,15))
for ptr in range (0,32):

 plt.subplot(8,4,ptr+1)
 plt.plot(range(trainN-1000,trainN+testN),dataT[trainN-1000:trainN+testN,ptr],'k',label="target system")
 plt.plot(range(trainN,trainN+testN),sol_matrix[:,ptr],'y', label="free running ESN")
 lo,hi = plt.ylim()
 plt.plot([trainN,trainN],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
 plt.legend(loc=(0.61,1.1),fontsize='x-small')
 plt.ylabel('grid point'+ ' '+ str(ptr),fontsize=14)


plt.savefig('time_series.png')

[t,grid]=np.meshgrid((np.arange(trainN,trainN+testN)),np.arange(0,64))


diff_mat=dataT[trainN:trainN+testN,:]-sol_matrix[0:testN,:]
v = np.linspace(-3, 3, 10, endpoint=True)


plt.figure(figsize=(45,45))
plt.subplot(3,1,1)
v = np.linspace(-3, 3, 10, endpoint=True)
plt.contourf(np.transpose(t),np.transpose(grid),dataT[trainN:trainN+testN,:],cmap=cm.jet,vmin=-3,vmax=3)

plt.colorbar(ticks=v)
plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.xlabel('$dt$',fontsize=40)
plt.ylabel('Grid Point',fontsize=40)
plt.title('Truth',fontsize=40)

plt.subplot(3,1,2)

plt.contourf(np.transpose(t),np.transpose(grid),sol_matrix[0:testN,:],cmap=cm.jet,vmin=-3,vmax=3)
plt.colorbar(ticks=v)
#cbar.ax.set_ylabel('verbosity coefficient')
#plt.clim(-3,3)

plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.xlabel('$dt$',fontsize=40)
plt.ylabel('Grid Point',fontsize=40)
plt.title('Prediction',fontsize=40)

plt.subplot(3,1,3)
plt.contourf(np.transpose(t),np.transpose(grid),diff_mat,cmap=cm.jet,vmin=-3,vmax=3)
plt.colorbar(ticks=v)
#cbar.ax.set_ylabel('verbosity coefficient')
plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.xlabel('$dt$',fontsize=40)
plt.ylabel('Grid Point',fontsize=40)
plt.title('Error Matrix',fontsize=40)

plt.savefig('contourplots.png')
