from lightSOM import SOM
import numpy as np
import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                    names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target'], usecols=[0, 5],
                   sep='\t+', engine='python')

data = data.values
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# Initialization and training
som_shape = (1, 3)

net=SOM().create(1, 3, data, pci=True,pbc=True)

net.train(0.5, 500, random_order=True)