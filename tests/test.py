import sys, os, errno
import numpy as np

sys.path.append('../')

from lightSOM import SOM
path='./'


if path != './':
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

raw_data = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.2, 0.2, 0.5]])
labels = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'indigo']

net=SOM().create(20, 20, raw_data,  neighborhood="mexican_hat", normalizer=None,feature_names=["red", 'green', 'blue'], target=labels,  pbc=True)

net.colorEx = True
net.train(0.05, 4000, random_order=False)
net.cluster()
#net.nodes_graph(colnum=0)


#
# #sm.codebook.lattice="rect"
#vhts  = MapView(12,12,"Features Map", text_size=10)
#vhts.show(net, which_dim=[0, 1,2],col_size=2, colorEx=False)

from lightSOM.visualization.som_view import SOMView
vhts  = SOMView(net, 12,12, text_size=10)
vhts.plot_nodes_maps(which_dim=[0, 1,2],col_size=2,denormalize=True, colorEx=True)
vhts.plot_frequencies()
#vhts.plot_hits_map()
# print("Saving weights and a few graphs...", end=' ')
# net.save('colorExample_weights', path=path)
# net.nodes_graph(path=path)
#
# net.diff_graph(path=path)
# test = net.project(raw_data, labels=labels, path=path)
#
# net.cluster(raw_data, type='KMeans', path=path, numcl=4)
