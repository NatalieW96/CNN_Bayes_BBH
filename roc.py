###################################################################################################
#Takes calculated bayes factors and machine learning class preditctions from the same dataset and
#produces a figure comparing them, and then later a ROC curve for the data. Note that the CNN
#dataset will be the smaller and correspond to the end of the bayes dataset of testing data size
###################################################################################################

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from numpy import maximum
import math

#Load in the bayes factors and classification values for the full dataset
n=10            #number of files
for j in np.arange(n):
    file = j+1
    if file == 1:
        with open('bayes_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            bayes = pickle.load(f)
        with open('classification_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            classification = pickle.load(f)
    else:
	with open('bayes_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            bayes2 = pickle.load(f)
        bayes = np.vstack((bayes, bayes2))
        with open('classification_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            classification2 = pickle.load(f)
        classification = np.vstack((classification, classification2))

#Load in CNN prediction values
CNN_result=np.load('/home/2118404/cnn_matchfiltering/history/run40/preds.npy')

#Create empty arrays for noise and signal for both bayes factors and CNN prediction values
noise = np.zeros((1,2))
signal = np.zeros((1,2))
MLnoise = np.zeros((1,2))
MLsignal = np.zeros((1,2))

#Set counters to zero
count1=0
count2=0

#Take last section of classification to correspond to classification of testing data for CNN
Nt=len(CNN_result)
CNN_classification=classification[-Nt:]
#Loop over bayes factors data to seperate into noise and signal data
Ns= len(bayes)
for i in np.arange(Ns):
    if classification[i] == 0:
        if count1 == 0:
            noise[0,:]= bayes[i,:]
        else:
            noise=np.vstack([noise,bayes[i,:]])
        count1 +=1
    else:
	if count2 == 0:
            signal[0,:] = bayes[i,:]
        else:
            signal = np.vstack([signal, bayes[i,:]])
        count2 +=1

#Reset counters
count1=0
count2=0

#Loop over CNN predictions to seperate into noise and signal data
for y in np.arange(len(CNN_classification)):
    if CNN_classification[y] == 0:
        if count1 == 0:
            MLnoise[0,:]= CNN_result[y,:]
        else:
            MLnoise = np.vstack([MLnoise, CNN_result[y,:]])
        count1 +=1
    else:
	if count2==0:
            MLsignal[0,:] = CNN_result[y,:]
        else:
            MLsignal = np.vstack([MLsignal, CNN_result[y,:]])
        count2 +=1

#Combine into one array for signal and noise respectivley and sort
signal_modify=signal[-len(MLsignal):]
noise_modify=noise[-len(MLnoise):]
graphsignal=np.column_stack((signal_modify[:,0],MLsignal[:,1]))

#Create figure to show relation between bayes factor and machine learning prediction
plt.figure()
plt.plot(graphsignal[:,0], graphsignal[:,1], color="red")
plt.plot(graphnoise[:,0], graphnoise[:,1], color = "blue")
plt.xscale('log')
plt.xlabel('Bayes Factor')
plt.ylabel('Machine Learning Statisitic')
plt.xlim((10**-3,10**3 ))
plt.savefig('/data/public_html/2118404/bayes_machinelearning.png')

#constants for ROC loop
N=1000  #number of sections
max = max(np.append(noise[:,0],signal[:,0]))
min = min(np.append(noise[:,0], signal[:,0]))
maxpower = int(math.ceil(math.log10(np.sort(signal[:,0])[-2]))) #in this example the maximum is inf, which cannot be used, so second largest power is taken
#maxpower = int(math.floor(math.log10(max)))
minpower = int(math.floor(math.log10(min)))

#Plot histogram of noise and signal data for the bayes factor
#plt.figure()
#plt.hist(np.log10(noise[:,0]), bins =np.arange(-5, 250,0.05) , normed = True, alpha = 0.5)
#plt.hist(np.log10(signal[:,0]), bins=np.arange(-5, 250,0.05) , normed = True, alpha = 0.5)
#plt.xlabel('Bayes')
#plt.xlim((-5,5))
#plt.savefig('/data/public_html/2118404/histograms.png')


#Create empty arrays for false alarm and true alarm for ROC curves
FA_bayes = np.zeros((N,1))
TA_bayes = np.zeros((N,1))
FA_ML = np.zeros((N,1))
TA_ML = np.zeros((N,1))

#Loop over data to create ROC curve data
for i, b in enumerate(np.logspace(minpower, maxpower, num=N)):
    FA_bayes[i]=float(len(np.where(noise[:,0]>b)[0]))/len(noise[:,0])
    TA_bayes[i]=float(len(np.where(signal[:,0]>b)[0]))/len(signal[:,0])

for j, x in enumerate(np.linspace(0,1,N)):
    FA_ML[j]=float(len(np.where(MLnoise[:,1]>x)[0]))/len(MLnoise[:,1])
    TA_ML[j]=float(len(np.where(MLsignal[:,1]>x)[0]))/len(MLsignal[:,1])

#Create ROC curve
plt.figure()
plt.plot(FA_bayes,TA_bayes)
plt.plot(FA_ML,TA_ML)
plt.xlabel('FA')
plt.ylabel('TA')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim((0.5, 1))
plt.savefig('/data/public_html/2118404/ROC.png')
exit()

