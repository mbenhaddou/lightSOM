
import pandas as pd
import numpy as np
columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                    names=columns,
                   sep='\t+', engine='python')
target = data['target'].values
label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = data[data.columns[:-1]]


data_vals = data.values

import sys, os, errno
import numpy as np

sys.path.append('../..')

from lightSOM import SOM
path='./output/basic'

if path != './output/basic':
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove']
net=SOM().create(25, 25, data_vals, feature_names=columns, target=target,  pbc=True)

net.train(0.1, 1000, random_order=False)


from lightSOM.visualization.som_view import SOMView
vhts  = SOMView(net, 10,10, text_size=10)
#vhts.plot_nodes_maps(which_dim="all",col_size=3,denormalize=True)
vhts.plot_training_errors()

