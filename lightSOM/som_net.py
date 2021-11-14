"""
SimpSOM (Simple Self-Organizing Maps) v1.3.4
F. Comitani @2017-2021

A lightweight python library for Kohonen Self-Organising Maps (SOM).
"""

from __future__ import print_function
from collections import defaultdict, Counter
import sys, time
from datetime import timedelta
import numpy as np
import os, errno
from kdmt.matrix import euclidean_distance
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sys import stdout
import lightSOM.clustering.densityPeak as dp
import lightSOM.clustering.qualityThreshold as qt
import logging

from lightSOM.normalization import NormalizerFactory
from lightSOM.nodes import Nodes
from lightSOM.distance import Distance
from lightSOM.neighborhood import get_neighberhood
from sklearn import cluster as sk_cluster
from lightSOM.decay_functions import *

class KSOM:
    """ Kohonen SOM Network class. """

    def __init__(self, height, width, data, decay_function="exponential_decay",  normalizer="var", distance="euclidean", neighborhood="gaussian", lattice="hexa",features_names=[], index=None, target=None, loadFile=None, PCI=0, PBC=0, random_seed=None):

        """Initialise the SOM network.

        Args:
            height (int): Number of nodes along the first dimension.
            width (int): Numer of nodes along the second dimension.
            data (np.array or list): N-dimensional dataset.
            loadFile (str, optional): Name of file to load containing information
                to initialise the network weights.
            PCI (boolean): Activate/Deactivate Principal Component Analysis to set
                the initial value of weights
            PBC (boolean): Activate/Deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC.
            n_jobs (int) [WORK IN PROGRESS]: Number of parallel processes (-1 use all available)
        """

        """ Switch to activate special workflow if running the colours example. """
        self.colorEx=False

        self.target=target
        self.index=index
        self.distance=distance

        """ Switch to activate periodic PCA weights initialisation. """
        self.PCI=bool(PCI)

        """ Switch to activate periodic boundary conditions. """
        self.PBC=bool(PBC)

        self.normalizer=None
        self._decay_function=eval(decay_function)
        if normalizer is not None:
            self.normalizer=NormalizerFactory.build(normalizer)
        """ Activate light parallelization. """
        #TODO:
        #self.n_jobs=n_jobs

        if self.PBC==True:
            print("Periodic Boundary Conditions active.")
        elif self.PBC==True and lattice=="rect":
            print("Periodic Boundary Conditions is still not implemented for rectangular lattice. Setting it to False")
            self.PBC=False

        else:
            print("Periodic Boundary Conditions inactive.")
        self.data_raw=data


        self.nodes=Nodes(width, height, dim= self.data_raw.shape[1], lattice=lattice, pbc=self.PBC)

        self._random_generator = np.random.RandomState(random_seed)
        self.data = self.normalizer.normalize(data) if normalizer else data
        self.data=data.reshape(np.array([data.shape[0], data.shape[1]]))
        self._features_names = self.build_features_names() if features_names is None else [features_names]
        self._distances=Distance()
        self._bmu=None
        """ Load the weights from file, generate them randomly or from PCA. """
        self._neighborhood=get_neighberhood(neighborhood)

        if loadFile==None:
            self.height = height
            self.width = width


            if self.PCI==True:
                print("The weights will be initialised with PCA.")


                self.nodes.pca_linear_initialization(self.data)
            else:
                print("The weights will be initialised randomly.")

                self.nodes.random_initialization(self.data, self._random_generator)

        else:
            print('The weights will be loaded from file.')

            if loadFile.endswith('.npy')==False:
                loadFile=loadFile+'.npy'
            weiArray=np.load(loadFile)
            #add something to check that data and array have the same dimensions,
            #or that they are mutually exclusive
            self.height = int(weiArray[0][0])
            self.width = int(weiArray[0][1])
            self.PBC= bool(weiArray[0][2])

        self._quantization_errors=[]
        self._topological_errors=[]

    def save(self, fileName='somNet_trained', path='./'):

        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            fileName (str, optional): Name of file where the data will be saved.

        """


        weiArray=[np.zeros(len(self.nodes.matrix[0][0]))]
        weiArray[0][0],weiArray[0][1],weiArray[0][2]= self.height, self.width, int(self.PBC)
        for w in self.nodes.matrix:
            weiArray.append(w)
        np.save(os.path.join(path,fileName), np.asarray(weiArray))



    def find_bmu(self, vec):

        """Find the best matching unit (BMU) for a given vector.

        Args:
            vec (np.array): The vector to match.

        Returns:
            bmu (Neuron): The best matching unit node2.

        """


        self._bmu_map_activation=self._distances.get_distance( vec, self.nodes.matrix, self.distance)


        bmu= np.unravel_index(self._bmu_map_activation.argmin(), self._bmu_map_activation.shape)
        return bmu


    def _verbose_iteration(self, iterations):
        """Yields the values in iterations printing the status on the stdout."""
        m = len(iterations)
        digits = len(str(m))
        progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
        progress = progress.format(m=m, d=digits, s=0)
        stdout.write(progress)
        beginning = time.time()
        stdout.write(progress)
        for i, it in enumerate(iterations):
            yield it
            sec_left = ((m - i + 1) * (time.time() - beginning)) / (i + 1)
            time_left = str(timedelta(seconds=sec_left))[:7]
            progress = '\r [ {i:{d}} / {m} ]'.format(i=i + 1, d=digits, m=m)
            progress += ' {p:3.0f}%'.format(p=100 * (i + 1) / m)
            progress += ' - {time_left} left '.format(time_left=time_left)
            stdout.write(progress)

    def _get_iteration_indexes(self, verbose=False, random_order=True):
        """Returns an iterable with the indexes of the samples
        to pick at each iteration of the training.

        If random_generator is not None, it must be an instance
        of numpy.random.RandomState and it will be used
        to randomize the order of the samples."""
        iterations = np.arange(self.epochs) % len(self.data)
        if random_order:
            self._random_generator.shuffle(iterations)
        if verbose:
            return self._verbose_iteration(iterations)
        else:
            return iterations


    def train(self, start_learning_rate=0.1, start_sigma=None, epochs=-1, verbose=True, random_order=True, keep_error_history=False):

        """Train the SOM.

        Args:
            startLearnRate (float): Initial learning rate.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints

        """

        logging.info(" Training...")
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            " -------------------------------------------------------------\n").format(data_len=self.data.shape[0],
                    data_dim=self.data.shape[1],
                    mpsz0=self.width,
                    mpsz1=self.height,
                    array_size=np.log10(
                        self.data.shape[0] * self.nodes.nnodes * self.data.shape[1])))

        if epochs==-1:
            epochs=self.data.shape[0]*10
        self.epochs=epochs

        iterations=self._get_iteration_indexes(verbose,random_order)

        self.startSigma=start_sigma
        if start_sigma is None:
            self.startSigma = max(self.height, self.width) / 2
        self.startLearnRate = start_learning_rate

        self.tau = self.epochs/np.log(self.startSigma) if self.startSigma>1 else self.epochs

        #TODO:
        #Parallel(n_jobs=self.n_jobs)(delayed(my_func)(c, K, N) for c in inputs)


        for i, it in enumerate(iterations):

            inputVec = self.data[it]

            bmu=self.find_bmu(inputVec)

            self.update_weights(inputVec, bmu, i)
#            self.update_weights2(inputVec, bmu, i)
            if keep_error_history:
                self._quantization_errors.append(self.quantization_error(self.data))
                self._topological_errors.append(self.topographic_error(self.data))
        if verbose:
            if self._quantization_errors !=[]:
                print('\n quantization error:', self._quantization_errors[-1])
            else:
                print('\n quantization error:', self.quantization_error(self.data))

            if self._topological_errors !=[]:
                print('\n topological error:', self._topological_errors[-1])
            else:
                print('\n topological error:', self.topographic_error(self.data))

            print("\rTraining SOM... done!")


    def update_weights(self, x, win, i):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        self.sigma = self._decay_function(self.startSigma, i, self.tau)#self.startSigma * np.exp(-i / self.tau)
        self.lrate = self._decay_function(self.startLearnRate, i, self.tau)#self.startLearnRate * np.exp(-i / self.tau)

        # improves the performances
        g = self._neighborhood(self.nodes, win, self.sigma)*self.lrate
        # w_new = eta * neighborhood_function * (x-w)
        self.nodes.matrix += np.einsum('ij, ijk->ijk', g, x-self.nodes.matrix)

    def bmu_map(self, data, return_indices=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""


        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.find_bmu(x)].append(i if return_indices else x)
        return winmap


    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding
        matrix x,y coordinates.

        :param bmu_ind: node index of the best matching unit
            (number of node from top left node)
        :returns: corresponding (x,y) coordinate
        """
        rows = self.nodes.mapsize[0]
        cols = self.nodes.mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """

        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.find_bmu(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap



    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""

        winners_coords = np.argmin(self._weight_distance(data), axis=1)
        return self.nodes.matrix[np.unravel_index(winners_coords,
                                           self.nodes.matrix.shape[:2])]

    def _weight_distance(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = np.array(data)
        weights_flat = self.nodes.matrix.reshape(-1, self.nodes.matrix.shape[2])
        input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(input_data, weights_flat.T)
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""

        return np.linalg.norm(data-self.quantization(data), axis=1).mean()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""

        # if self.nodes.lattice == 'hexa':
        #     msg = 'Topographic error not implemented for hexagonal topology.'
        #     raise NotImplementedError(msg)
        # total_neurons = np.prod(self._bmu_map_activation.shape)
        # if total_neurons == 1:
        #     warn('The topographic error is not defined for a 1-by-1 map.')
        #     return np.nan

        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = np.argsort(self._weight_distance(data), axis=1)[:, :2]
        b2my_xy = np.unravel_index(b2mu_inds, self.nodes.matrix.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = np.hstack([np.diff(b2mu_x), np.diff(b2mu_y)])
        distance = np.linalg.norm(dxdy, axis=1)
        return (distance > t).mean()


    def activation_frequency(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """

        a = np.zeros((self.nodes.matrix.shape[0], self.nodes.matrix.shape[1]))
        for x in data:
            a[self.find_bmu(x)] += 1
        return a

    @property
    def features_names(self):
        return self._features_names

    @features_names.setter
    def features_names(self, compnames):
        if self._dim == len(compnames):
            self._features_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ValueError('Component names should have the same '
                                      'size as the data dimension/features')


    def build_features_names(self):
        cc = ['Feature-' + str(i+1) for i in range(0, self._dim)]
        return np.asarray(cc)[np.newaxis, :]

    def build_u_matrix(self, distance=1.01, row_normalized=False):
        UD2 = np.array(self.nodes.node_distances)
        Umatrix = np.zeros((self.nodes.nnodes, 1))
        codebook = self.nodes.matrix.reshape(-1, self.data.shape[1])
        if row_normalized:
            vector = self.normalizer.normalize(codebook.T, codebook.T).T
        else:
            vector = codebook

        for i in range(self.nodes.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            neighborbor_ind = UD2[i][0:] <= distance
            neighborbor_codebooks = vector[neighborbor_ind]
            neighborbor_dists = euclidean_distance(
                codebook_i, neighborbor_codebooks)
            Umatrix[i] = np.float(np.sum(neighborbor_dists) / (neighborbor_dists.shape[1] - 1))

        return Umatrix.reshape(self.nodes.nnodes)

    def project(self, array, colnum=-1, labels=[], show=False, printout=True, path='./', colname = None):

        """Project the datapoints of a given array to the 2D space of the
            SOM by calculating the bmus. If requested plot a 2D map with as
            implemented in nodes_graph and adds circles to the bmu
            of each datapoint in a given array.

        Args:
            array (np.array): An array containing datapoints to be mapped.
            colnum (int): The index of the weight that will be shown as colormap.
                If not chosen, the difference map will be used instead.
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.

        Returns:
            (list): bmu x,y position for each input array datapoint.

        """

        if not colname:
            colname = str(colnum)

        if labels != []:
            colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
            		  '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
            class_assignment = {}
            counter = 0
            for i in range(len(labels)):
                if labels[i] not in class_assignment:
                    class_assignment[labels[i]] = colors[counter]
                    counter = (counter + 1)%len(colors)

        bmuList,cls=[],[]
        for i in range(array.shape[0]):
            bmuList.append(self.find_bmu(array[i,:]))
            if self.colorEx==True:
                cls.append(array[i,:])
            else:
                if labels!=[]:
                    cls.append(class_assignment[labels[i]])
                elif colnum==-1:
                    cls.append('#ffffff')
                else:
                    cls.append(array[i,colnum])

        """ Print the x,y coordinates of bmus, useful for the clustering function. """

        return bmuList

    def cluster_data(self, array=None, type='qthresh', cutoff=5, quant=0.2, percent=0.02, numcl=8):

        """Clusters the data in a given array according to the SOM trained map.
            The clusters can also be plotted.

        Args:
            array (np.array): An array containing datapoints to be clustered.
            type (str, optional): The type of clustering to be applied, so far only quality threshold (qthresh)
                algorithm is directly implemented, other algorithms require sklearn.
            cutoff (float, optional): Cutoff for the quality threshold algorithm. This also doubles as
                maximum distance of two points to be considered in the same cluster with DBSCAN.
            percent (float, optional): The percentile that defines the reference distance in density peak clustering (dpeak).
            numcl (int, optional): The number of clusters for K-Means clustering
            quant (float, optional): Quantile used to calculate the bandwidth of the mean shift algorithm.
            savefile (bool, optional): Choose to save the resulting clusters in a text file.
            filetype (string, optional): Format of the file where the clusters will be saved (csv or dat)
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.

        Returns:
            (list of int): A nested list containing the clusters with indexes of the input array points.

        """

        """ Call project to first find the bmu for each array datapoint, but without producing any graph. """
        if array is None:
            array=self.data
        bmuList = self.project(array, show=False, printout=False)
        clusters=[]

        if type=='qthresh':

            """ Cluster according to the quality threshold algorithm (slow!). """

            clusters = qt.qualityThreshold(bmuList, cutoff, self.PBC, self.height, self.width)

        elif type=='dpeak':

            """ Cluster according to the density peak algorithm. """

            clusters = dp.densityPeak(bmuList, PBC=self.PBC, netHeight=self.height, netWidth=self.width)

        elif type in ['MeanShift', 'DBSCAN', 'KMeans']:

            """ Cluster according to algorithms implemented in sklearn. """

            if self.PBC==True:
                print("Warning: Only Quality Threshold and Density Peak clustering work with PBC")

            try:

                if type=='MeanShift':
                    bandwidth = sk_cluster.estimate_bandwidth(np.asarray(bmuList), quantile=quant, n_samples=500)
                    cl = sk_cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmuList)

                if type=='DBSCAN':
                    cl = sk_cluster.DBSCAN(eps=cutoff, min_samples=5).fit(bmuList)

                if type=='KMeans':
                    cl= sk_cluster.KMeans(n_clusters=numcl).fit(bmuList)

                clLabs = cl.labels_

                for i in np.unique(clLabs):
                    clList=[]
                    tmpList=range(len(bmuList))
                    for j,k in zip(tmpList,clLabs):
                        if i==k:
                            clList.append(j)
                    clusters.append(clList)
            except:
                print(('Unexpected error: ', sys.exc_info()[0]))
                raise
        else:
            sys.exit("Error: unkown clustering algorithm " + type)
        clusters_=[0]* self.nodes.nnodes

        for i, cluster in enumerate(clusters):
            for c in cluster:
                clusters_[c]=i
        return clusters_

    def cluster(self, n_clusters=4, random_state=0):
        import sklearn.cluster as clust
        cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(self.nodes.matrix.reshape(-1, self.nodes.dim))
        self.cluster_labels = cl_labels
        return cl_labels

    def classify(self, data):
        """Classifies each sample in data in one of the classes definited
        using the method labels_map.
        Returns a list of the same length of data where the i-th element
        is the class assigned to data[i].
        """
        if self.target is None:
            raise ValueError("target option is missing for class initialization")

        winmap = self.labels_map(self.data, self.target)
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = self.find_bmu(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

def run_colorsExample(path='./'):
    """Example of usage of SimpSOM: a number of vectors of length three
        (corresponding to the RGB values of a color) are used to briefly train a small network.
        Different example graphs are then printed from the trained network.
    """

    """Try to create the folder"""
    if path != './':
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    raw_data = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.2, 0.2, 0.5]])
    labels = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'indigo']

    print(
        "Welcome to SimpSOM (Simple Self Organizing Maps) v1.3.4!\nHere is a quick example of what this library can do.\n")
    print("The algorithm will now try to map the following colors: ", end=' ')
    for i in range(len(labels) - 1):
        print((labels[i] + ", "), end=' ')
    print("and " + labels[-1] + ".\n")

    net = KSOM(20, 20, raw_data, PBC=True)

    net.colorEx = True
    net.train(0.01, 2000)

    print("Saving weights and a few graphs...", end=' ')
    net.save('colorExample_weights', path=path)
    net.nodes_graph(path=path)

    net.diff_graph(path=path)
    test = net.project(raw_data, labels=labels, path=path)

    net.cluster(raw_data, type='KMeans', path=path, numcl=4)

    print("done!")
if __name__ == "__main__":

    run_colorsExample()
