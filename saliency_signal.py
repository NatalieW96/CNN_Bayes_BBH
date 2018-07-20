import subprocess
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import load_model

N = 8192
model = load_model('/home/2118404/cnn_matchfiltering/history/SNR10/run0/nn_model.hdf5')

with open('/home/2118404/cnn_matchfiltering/data_ts_signal_0.sav') as test:
    data=pickle.load(test)

timeseries= data[0][0][0]
print(len(timeseries))
print(timeseries)


plt.figure()
plt.plot(timeseries)
plt.savefig('/data/public_html/2118404/signal_SNR10.png')


timeseries= (timeseries[N/2:N+N/2]).astype(dtype=float).reshape(1,1,1,N)
target = np.zeros((1,2))
target[0,1] = 1
print(np.shape(timeseries))


# get baseline loss value
loss_baseline = model.test_on_batch(timeseries,target)
print(loss_baseline)

# loop over each timesample
timeseries_modify = np.tile(timeseries[0,0,0,:],(N,1)).reshape(N,N)
timeseries_modify += 0.1*np.eye(N)
timeseries_modify = timeseries_modify.reshape(N,1,1,N)
print(timeseries_modify.shape)
target_modify = np.zeros((N,2))
target_modify[:,1] = 1

i = 0
loss = np.zeros(N)
for ts,y in zip(timeseries_modify,target_modify):
    loss[i] = model.test_on_batch(ts.reshape(1,1,1,N),y.reshape(1,2))[-1]
    if i % 100 == 0:
        print(i)
    i += 1
print(loss)

plt.figure()
plt.plot(loss-loss_baseline[-1])
plt.savefig('/data/public_html/2118404/saliency_signal_SNR10.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(500)))
plt.savefig('/data/public_html/2118404/saliency_signal_SNR10_500.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(250)))
plt.savefig('/data/public_html/2118404/saliency_signal_SNR10_250.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(1000)))
plt.savefig('/data/public_html/2118404/saliency_signal_SNR10_1000.png')



exit(0)





