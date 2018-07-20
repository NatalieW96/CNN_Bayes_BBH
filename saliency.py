##################################################################################################
#Uses a CNN model and a BBH signal to output saliency maps. Here the output maps
#corresponds to no smoothing, and smoothing windows of 250, 500 and 1000.
##################################################################################################
import subprocess
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import load_model

N = 8192                        #Number of timesamples

#Load model to use for saliency
model = load_model('/home/2118404/cnn_matchfiltering/history/SNR10/run0/nn_model.hdf5')

#Load timeseries file to use for saliency
with open('/home/gabbard/CBC/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR10_Hdet_metricmass_119seed_ts_0.sav') as test:
    data=pickle.load(test)

#Selects the first timeseries of the file - note there must be a signal in this
timeseries= data[0][0][0]

#Takes the relevent window in the timeseries
timeseries= (timeseries[N/2:N+N/2]).astype(dtype=float).reshape(1,1,1,N)

#Create an empty target array
target = np.zeros((1,2))
target[0,1] = 1

# Get baseline prediction value
class_baseline = model.predict_on_batch(timeseries,target)
print(class_baseline)

#Make diagonalised array of small change for saliency
timeseries_modify = np.tile(timeseries[0,0,0,:],(N,1)).reshape(N,N)
timeseries_modify += 0.1*np.eye(N)
timeseries_modify = timeseries_modify.reshape(N,1,1,N)

#Make modified target array
target_modify = np.zeros((N,2))
target_modify[:,1] = 1

#Loop over timesamples to obtain saliency
i = 0
class_prob = np.zeros(N)
for ts,y in zip(timeseries_modify,target_modify):
    class_prob[i] = model.predict_on_batch(ts.reshape(1,1,1,N),y.reshape(1,2))[-1]
    if i % 100 == 0:
        print(i)
    i += 1

#Produce saliency maps for unsmoothed, smoothed windows of 500, 250 and 1000 respectivley
plt.figure()
plt.plot(class_prob-class_baseline[-1])
plt.savefig('/data/public_html/2118404/saliency_SNR10.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(500)))
plt.savefig('/data/public_html/2118404/saliency_SNR10_500.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(250)))
plt.savefig('/data/public_html/2118404/saliency_SNR10_250.png')

plt.figure()
plt.plot(np.convolve(loss-loss_baseline[-1], np.ones(1000)))
plt.savefig('/data/public_html/2118404/saliency_SNR10_1000.png')

exit(0)
