###################################################################################################
#Takes a CNN model and produces the class visualisation of a BBH signal
###################################################################################################
import matplotlib
matplotlib.use('agg')
from vis.utils import utils
from keras import activations
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter

#Load in the CNN model
model = load_model('/home/2118404/cnn_matchfiltering/history/SNR10/run0/nn_model.hdf5')
layer_idx= -1           #Chooses the last layer
#Swap softmax with linear
model.layers[layer_idx].activation=activations.linear
model = utils.apply_modifications(model)

#Choose catagory for signal (1 in this case)
img = visualize_activation(model, layer_idx, filter_indices=1, max_iter =500, input_modifiers=[Jitter(16)])

#Create visualisation
plt.plot(img[0][0])
plt.savefig('/data/public_html/2118404/visualization_signal.png')

exit()



