###################################################################################################
#Generates a randomised datafile of noisy sine-gaussians of various SNR and noise only signals, and
#calculates the Bayes factor for each one. Details on the calculation of the Bayes factor can be
#found in LaTeX document.
###################################################################################################
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
import random
import cPickle as pickle

#Define parameters as a class (R, S, Gaussian envelope width, central time)
class sgparams:
    def __init__(self,R,S,tau,t0):
        self.R = R
        self.S = S
        self.tau = tau
        self.t0 = t0

def sig(R,S,tau,t0):
    # signal - takes R and S as input parameters
    return R*np.sin(2*np.pi*f0*(t-t0))*np.exp(-((t-t0)/tau)**2) + S*np.cos(2*np.pi*f0*(t-t0))*np.exp(-((t-t0)/tau)**2)

def L(R,S,tau,t0):
    # likelihood ratio on R and S
    return np.exp(-0.5*np.sum(((x-sig(R,S,tau,t0))/sigma)**2 - (x/sigma)**2))

def p(R,S,tau,t0):
    # joint prior on R,S
    return (1.0/(2.0*np.pi*RSsig**2))*np.exp(-0.5*(R**2 + S**2)/RSsig**2)

def P(R,S,tau,t0):
    # integrand for Bayes factor
    return L(R,S,tau,t0)*p(R,S,tau,t0)

def integrand(tauval, t0val):
    fj = sig(1.0,0.0,tauval,t0val)	 # make fj timeseries - just call sig functiuon with R=1,S=0
    gj = sig(0.0,1.0,tauval,t0val)	 # make fj timeseries - just call sig functiuon with R=0,S=1

    # precompute coefficients
    a = (np.sum(fj**2))*0.5*k + Cn
    c = (np.sum(gj**2))*0.5*k + Cn
    b = (np.sum(fj*x))*k
    d = (np.sum(gj*x))*k

    # evaluate analytical integration result
    return (Cn/np.sqrt(a*c))*np.exp(0.25*((b**2)/a + (d**2)/c))


# simulation parameters
Nt = 512                # number of time samples
T = 1                   # duration of observation (sec)
dt = T/float(Nt)        # sampling time (sec)
t = dt*np.arange(Nt)    # vector of times
sigma = 1               # noise standard deviation
RSsig = 1.0             # prior width on signal amplitudes R and S
Ns = 50000              # number of data samples
f0 = 32.0               # the signal frequency (Hz)

nf=5                    # number of files to be made
q=Ns/nf

#Create empty arrays to be filled
data= np.zeros((q,Nt))
signal= np.zeros((q,Nt))
noise= np.zeros((q,Nt))
bayes = np.zeros((q,2))
classification = np.zeros((q,1))
par = []

#Set counter to zero
count = 0
Ntot= 100000            # total data series for use in file name

#Integration parmeters
dt0=0.01                # central time steps
dtau= 0.01              # tau steps
taumin=0.05             # minimum tau
taumax=0.25             # maximum tau
t0min=0.25              # minimum t0
t0max=0.75              # maximum t0

#Constants of analytical integral
k= 1.0/sigma**2
Cn= 0.5/RSsig**2
t1= time.time()         # start time
# Make data
for i in np.arange(Ns):
    y=random.random()   # randomly chosen signal and noise or noise only
    ns = np.random.normal(loc=0,scale=sigma,size=Nt)    #create noise

    if y>0.5:
	R0 =np.random.normal(loc=0, scale=RSsig, size = 1)          # the R amplitude
        S0 = np.random.normal(loc=0,scale=RSsig, size=1)            # the S amplitude
        tau0 = random.uniform(taumin, taumax)                       # the Gaussian envelope width (sec)
        t00 = random.uniform(t0min, t0max)                          # the central time (sec)
        h = sig(R0,S0, tau0, t00)                                   # make the signal
        classification[count]=1                                     # classify as a signal
        par.append(sgparams(R0, S0, tau0,t00))                      # save parameters

    else:
	h = np.zeros(Nt)                                            # create no signal
        classification[count]=0                                     # classify as noise only
        par.append(None)                                            # save 'None' as parameters

    x = h + ns                                                      # add signal and noise to make data
    data[count,:]=x                                                 # save data
    signal[count,:]=h                                               # save signal
    noise[count,:]=ns                                               # save noise

    #Calculate Bayes factor
    bayes[count,0], bayes[count,1] = integrate.dblquad(integrand,t0min , t0max, lambda x: taumin, lambda x: taumax, epsabs=1.49e-6, epsrel=1.49e-6)

    #Save files
    if count+1 == q:
        n=(i/q).astype(int) + 1
        pickle.dump(data, open('data_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved dataseries to data_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n)
        pickle.dump(signal, open('signal_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved signal to signal_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n)
        pickle.dump(noise, open('noise_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved noise to noise_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ns, Ntot,n)
        pickle.dump(bayes, open('bayes_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved bayes factors to bayes_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n)
        pickle.dump(par, open('params_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved parameters to params_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n)
        pickle.dump(classification, open('classification_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n), 'wb'))
        print 'saved classification to classification_{}Hz_{}samp_{}_n{}.sav'.format(f0, Ntot, Nt,n)
        count=0                                                     # reset counter
        par=[]                                                      # reset parameters
    else:
	count+=1

t2=time.time()          #end time
print t1-t2

exit(0)
