import pyESN
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

data=genfromtxt('lorenz_63_modified.csv',delimiter=",")
dataT=np.transpose(data)
dataT=np.transpose(dataT)
#dataT=dataT[2500:,:]
np.shape(dataT)


esn = pyESN.ESN(n_inputs = 1,
          n_outputs = 3,
          n_reservoir = 2500,
          spectral_radius = 0.9, sparsity=0.002,
          random_state=42)

spin_off=0;
trainN = 40000
testN = 2000
sol_matrix=np.zeros([testN,3])
np.shape(sol_matrix)


pred_training = esn.fit(np.ones(trainN-spin_off),dataT[spin_off:trainN,:],inspect=True)

prediction = esn.predict(np.ones(testN))

error_mat=prediction-dataT[trainN:trainN+testN]

print('RMSE:')
print((np.sum(np.sum(np.square(error_mat))))/float(np.sum(np.sum(np.square(dataT[trainN:trainN+testN])))))

np.save('prediction_40000',prediction)

v=0.05*(np.arange(0,2000))

ptr=0;
plt.figure(figsize=(15,1.5))
plt.plot(v,dataT[trainN:trainN+testN,ptr],'k',label="target system")
plt.plot(v,prediction[:,ptr],'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([0.0,0.0],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.title('X coordinate')
plt.xlabel('MTU')


ptr=1;
plt.figure(figsize=(15,1.5))
plt.plot(v,dataT[trainN:trainN+testN,ptr],'k',label="target system")
plt.plot(v,prediction[:,ptr],'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([0,0],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.title('Y coordinate')
plt.xlabel('MTU')

ptr=2;
plt.figure(figsize=(15,1.5))
plt.plot(v,dataT[trainN:trainN+testN,ptr],'k',label="target system")
plt.plot(v,prediction[:,ptr],'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([0,0],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.title('Z coordinate')
plt.xlabel('MTU')

plt.show()
