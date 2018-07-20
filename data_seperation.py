###################################################################################################
#Takes generated data, classifications and parameters and seperates into training, validation
#and testing data for use in CNN
###################################################################################################
import cPickle as pickle
import numpy as np
import itertools

class sgparams:
    def __init__(self,R,S,tau,t0):
        self.R = R
        self.S = S
        self.tau = tau
        self.t0 = t0

#Load in the parameters, classification values and data for the full dataset
n=10             #number of files
for j in np.arange(n):
    file = j+1
    if file == 1:
        with open('params_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            params = pickle.load(f)
        with open('classification_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            classification = pickle.load(f)
        with open('data_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            data = pickle.load(f)

    else:
	with open('params_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            params2 = pickle.load(f)
        params = params + params2
        with open('classification_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            classification2 = pickle.load(f)
        classification = np.vstack((classification, classification2))
        with open('data_32.0Hz_100000samp_512_n{}.sav'.format(file)) as f:
            data2 = pickle.load(f)
        data = np.vstack((data, data2))

#Define training size,validation size and testing size
training_size=90000
validation_size=5000
testing_size=5000

#Seperate data, parameters and classifications
data_training = data[0:training_size]
data_validation = data[training_size:training_size+validation_size]
data_testing = data[training_size+validation_size:training_size+validation_size+testing_size]

params_training = params[0:training_size]
params_validation = params[training_size:training_size+validation_size]
params_testing = params[training_size+validation_size:training_size+validation_size+testing_size]

classification_training = classification[0:training_size]
classification_validation = classification[training_size:training_size+validation_size]
classification_testing = classification[training_size+validation_size:training_size+validation_size+testing_size]

#Combine data and classification into one file
data_class_training= [np.expand_dims(data_training, axis=1),np.transpose(classification_training[:,0].astype(int))]
data_class_validation= [np.expand_dims(data_validation, axis=1),np.transpose(classification_validation[:,0].astype(int))]
data_class_testing=[np.expand_dims(data_testing, axis=1),np.transpose(classification_testing[:,0].astype(int))]

#Save files
pickle.dump(data_class_training, open('data_class_training_32Hz_100000samp_512.sav', 'wb'))
pickle.dump(data_class_validation, open('data_class_validation_32Hz_100000samp_512.sav', 'wb'))
pickle.dump(data_class_testing, open('data_class_testing_32Hz_100000samp_512.sav', 'wb'))

pickle.dump(params_training, open('params_training_32Hz_100000samp_512.sav', 'wb'))
pickle.dump(params_validation, open('params_validation_32Hz_100000samp_512.sav', 'wb'))
pickle.dump(params_testing, open('params_testing_32Hz_100000samp_512.sav', 'wb'))

exit()

