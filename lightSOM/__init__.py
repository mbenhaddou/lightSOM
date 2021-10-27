"""
SimpSOM (Simple Self-Organizing Maps) v1.3.4
F. Comitani @2017-2021 
 
A lightweight python library for Kohonen Self-Organising Maps (SOM).
"""

from __future__ import print_function

import sys
import numpy as np
import os, errno

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lightSOM.hexagons as hx
import lightSOM.densityPeak as dp
import lightSOM.qualityThreshold as qt
import logging
from sklearn.decomposition import PCA
from sklearn import cluster

#from joblib import Parallel, delayed

class somNet:
    """ Kohonen SOM Network class. """

    def __init__(self, netHeight, netWidth, data, loadFile=None, PCI=0, PBC=0, n_jobs=-1):

        """Initialise the SOM network.

        Args:
            netHeight (int): Number of nodes along the first dimension.
            netWidth (int): Numer of nodes along the second dimension.
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
        
        """ Switch to activate periodic PCA weights initialisation. """
        self.PCI=bool(PCI)

        """ Switch to activate periodic boundary conditions. """
        self.PBC=bool(PBC)

        """ Activate light parallelization. """
        #TODO:
        #self.n_jobs=n_jobs

        if self.PBC==True:
            print("Periodic Boundary Conditions active.")
        else:
            print("Periodic Boundary Conditions inactive.")

        self.nodeList=[]
        self.data=data.reshape(np.array([data.shape[0], data.shape[1]]))

        """ Load the weights from file, generate them randomly or from PCA. """

        if loadFile==None:
            self.height = netHeight
            self.width = netWidth

            minVal,maxVal=[],[]
            pcaVec=[]

            if self.PCI==True:
                print("The weights will be initialised with PCA.")
            
                self.pca_linear_initialization()
            
            else:
                print("The weights will be initialised randomly.")
            
                self.random_initialization()

            self.matrix = self.matrix.reshape(self.width, self.height, self.data.shape[1])

            for x in range(self.width):
                for y in range(self.height):
                    self.nodeList.append(Neuron(x, y, self.data.shape[1], self.height, self.width, self.PBC, init_vect=self.matrix[x, y]))
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

            #start from 1 because 0 contains info on the size of the network
            countWei=1
            for x in range(self.width):
                for y in range(self.height):
                    self.nodeList.append(Neuron(x, y, self.data.shape[1], self.height, self.width, self.PBC, weiArray=weiArray[countWei]))
                    countWei+=1


    def save(self, fileName='somNet_trained', path='./'):
    
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            fileName (str, optional): Name of file where the data will be saved.
            
        """
        
        
        weiArray=[np.zeros(len(self.nodeList[0].weights))]
        weiArray[0][0],weiArray[0][1],weiArray[0][2]= self.height, self.width, int(self.PBC)
        for node in self.nodeList:
            weiArray.append(node.weights)
        np.save(os.path.join(path,fileName), np.asarray(weiArray))
    
    @property
    def nb_nodes(self):
        return len(self.nodeList)

    def pca_linear_initialization(self):
        """
        We initialize the map, just by using the first two first eigen vals and
        eigenvectors
        Further, we create a linear combination of them in the new map by
        giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication
        // PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen
        // vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected
        eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        cols = self.height
        nb_nodes=self.height*self.width
        coord = np.zeros((nb_nodes, 2))
        pca_components = 2

        for i in range(0, nb_nodes):
            coord[i, 0] = int(i / cols)  # x
            coord[i, 1] = int(i % cols)  # y



        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(self.data, 0)
        data = (self.data - me)
        tmp_matrix = np.tile(me, (nb_nodes, 1))

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components) # RandomizedPCA is deprecated.
        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(nb_nodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)

        self.initialized = True


    def random_initialization(self):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(self.data, axis=0), (self.width*self.height, 1))
        mx = np.tile(np.max(self.data, axis=0), (self.width*self.height, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.width*self.height, self.data.shape[1]))
        self.initialized = True

    

    def find_bmu(self, vec):
    
        """Find the best matching unit (BMU) for a given vector.

        Args:           
            vec (np.array): The vector to match.
            
        Returns:            
            bmu (Neuron): The best matching unit node.
            
        """


        distSq = (np.square(self.matrix - vec)).sum(axis=2)
        bmu= np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

        return self.nodeList[bmu[0]*self.width+bmu[1]]

    def train(self, startLearnRate=0.01, epochs=-1):
    
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
                        self.data.shape[0] * self.nb_nodes * self.data.shape[1])))

        
        print("Training SOM... 0%", end=' ')
        self.startSigma = max(self.height, self.width) / 2
        self.startLearnRate = startLearnRate
        if epochs==-1:
            epochs=self.data.shape[0]*10
        self.epochs=epochs
        self.tau = self.epochs/np.log(self.startSigma)
    
        #TODO:
        #Parallel(n_jobs=self.n_jobs)(delayed(my_func)(c, K, N) for c in inputs)

        for i in range(self.epochs):

            if i%100==0:
                print(("\rTraining SOM... "+str(int(i*100.0/self.epochs))+"%" ), end=' ')

            self.sigma = self.startSigma * np.exp(-i / self.tau)
            self.lrate = self.startLearnRate * np.exp(-i / self.epochs)
            
            """ Train with the bootstrap-like method: 
                instead of using all the training points, a random datapoint is chosen with substitution
                for each iteration and used to update the weights of all the nodes.
            """
            
            inputVec = self.data[np.random.randint(0, self.data.shape[0]), :].reshape(np.array([self.data.shape[1]]))
            
            bmu=self.find_bmu(inputVec)

            self.update_weights(inputVec, self.lrate, self.sigma, bmu)

            # for node in self.nodeList:
            #     node.update_weights(inputVec, self.sigma, self.lrate, bmu)

        print("\rTraining SOM... done!")



    def update_weights(self, train_ex, learn_rate, radius_sq, bmu, step=6):
        g, h = bmu.pos
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.matrix[g, h, :] += learn_rate * (train_ex - self.matrix[g, h, :])

        # Change all cells in a small neighborhood of BMU
        for i in range(int(max(0, g - step)), int(min(self.matrix.shape[0], g + step))):
            for j in range(int(max(0, h - step)), int(min(self.matrix.shape[1], h + step))):
                dist_sq = np.sqrt(np.square(i - g) + np.square(j - h))
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.matrix[i, j, :] -= learn_rate * dist_func * (self.matrix[i, j, :]-train_ex)
                self.nodeList[i*self.width+j].weights= self.matrix[i, j]

    def nodes_graph(self, colnum=0, show=False, printout=True, path='./', colname=None):
    
        """Plot a 2D map with hexagonal nodes and weights values

        Args:
            colnum (int): The index of the weight that will be shown as colormap.
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.
        """

        if not colname:
            colname = str(colnum)

        centers = [[node.lattice_pos[0], node.lattice_pos[1]] for node in self.nodeList]

        widthP=100
        dpi=72
        xInch = self.width * widthP / dpi
        yInch = self.height * widthP / dpi
        fig=plt.figure(figsize=(xInch, yInch), dpi=dpi)

        if self.colorEx==True:
            cols = [[np.float(node.weights[0]),np.float(node.weights[1]),np.float(node.weights[2])]for node in self.nodeList]   
            ax = hx.plot_hex(fig, centers, cols)
            ax.set_title('Node Grid w Color Features', size=80)
            printName=os.path.join(path,'nodesColors.png')

        else:
            cols = [node.weights[colnum] for node in self.nodeList]
            ax = hx.plot_hex(fig, centers, cols)
            ax.set_title('Node Grid w Feature ' +  colname, size=80)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            cbar=plt.colorbar(ax.collections[0], cax=cax)
            cbar.set_label(colname, size=80, labelpad=50)
            cbar.ax.tick_params(labelsize=60)
            plt.sca(ax)
            printName=os.path.join(path,'nodesFeature_'+str(colnum)+'.png')
            
        if printout==True:
            plt.savefig(printName, bbox_inches='tight', dpi=dpi)
        if show==True:
            plt.show()
        if show!=False and printout!=False:
            plt.clf()


    def diff_graph(self, show=False, printout=True, returns=False, path='./'):
    
        """Plot a 2D map with nodes and weights difference among neighbouring nodes.

        Args:
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            returns (bool, optional): Choose to return the difference value.

        Returns:
            (list): difference value for each node.             
        """
        
        neighbours=[]
        for node in self.nodeList:
            nodelist=[]
            for nodet in self.nodeList:
                if node != nodet and node.get_nodeDistance(nodet) <= 1.001:
                    nodelist.append(nodet)
            neighbours.append(nodelist)     
            
        diffs = []
        for node, neighbours in zip(self.nodeList, neighbours):
            diff=0
            for nb in neighbours:
                diff=diff+node.get_distance(nb.weights)
            diffs.append(diff)  

        centers = [[node.lattice_pos[0], node.lattice_pos[1]] for node in self.nodeList]

        if show==True or printout==True:
        
            widthP=100
            dpi=72
            xInch = self.width * widthP / dpi
            yInch = self.height * widthP / dpi
            fig=plt.figure(figsize=(xInch, yInch), dpi=dpi)

            ax = hx.plot_hex(fig, centers, diffs)
            ax.set_title('Nodes Grid w Weights Difference', size=80)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            cbar=plt.colorbar(ax.collections[0], cax=cax)
            cbar.set_label('Weights Difference', size=80, labelpad=50)
            cbar.ax.tick_params(labelsize=60)
            plt.sca(ax)

            printName=os.path.join(path,'nodesDifference.png')
            
            if printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=dpi)
            if show==True:
                plt.show()
            if show!=False and printout!=False:
                plt.clf()

        if returns==True:
            return diffs 

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
            bmuList.append(self.find_bmu(array[i,:]).lattice_pos)
            if self.colorEx==True:
                cls.append(array[i,:])
            else: 
                if labels!=[]:   
                    cls.append(class_assignment[labels[i]])
                elif colnum==-1:
                    cls.append('#ffffff')
                else: 
                    cls.append(array[i,colnum])

        if show==True or printout==True:
        
            """ Call nodes_graph/diff_graph to first build the 2D map of the nodes. """

            if self.colorEx==True:
                printName=os.path.join(path,'colorProjection.png')
                self.nodes_graph(colnum, False, False)
                plt.scatter([pos[0] for pos in bmuList],[pos[1] for pos in bmuList], color=cls,  
                        s=500, edgecolor='#ffffff', linewidth=5, zorder=10)
                plt.title('Datapoints Projection', size=80)
            else:
                #a random perturbation is added to the points positions so that data 
                #belonging plotted to the same bmu will be visible in the plot      
                if colnum==-1:
                    printName=os.path.join(path,'projection_difference.png')
                    self.diff_graph(False, False, False)
                    plt.scatter([pos[0]-0.125+np.random.rand()*0.25 for pos in bmuList],[pos[1]-0.125+np.random.rand()*0.25 for pos in bmuList], c=cls, cmap=cm.viridis,
                            s=400, linewidth=0, zorder=10)
                    plt.title('Datapoints Projection on Nodes Difference', size=80)
                else:   
                    printName=os.path.join(path,'projection_'+ colname +'.png')
                    self.nodes_graph(colnum, False, False, colname=colname)
                    plt.scatter([pos[0]-0.125+np.random.rand()*0.25 for pos in bmuList],[pos[1]-0.125+np.random.rand()*0.25 for pos in bmuList], c=cls, cmap=cm.viridis,
                            s=400, edgecolor='#ffffff', linewidth=4, zorder=10)
                    plt.title('Datapoints Projection #' +  str(colnum), size=80)
                
            if labels!=[]:
                recs = []
                for i in class_assignment:
                    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_assignment[i]))
                plt.legend(recs,class_assignment.keys(),loc=0)

            # if labels!=[]:
            #     for label, x, y in zip(labels, [lattice_pos[0] for lattice_pos in bmuList],[lattice_pos[1] for lattice_pos in bmuList]):
            #         plt.annotate(label, xy=(x,y), xytext=(-0.5, 0.5), textcoords='offset points', ha='right', va='bottom', size=50, zorder=11) 
            
            if printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=72)
            if show==True:
                plt.show()
            plt.clf()
        
        """ Print the x,y coordinates of bmus, useful for the clustering function. """
        
        return [[pos[0],pos[1]] for pos in bmuList] 
        
        
    def cluster(self, array, type='qthresh', cutoff=5, quant=0.2, percent=0.02, numcl=8,\
                    savefile=True, filetype='dat', show=False, printout=True, path='./'):
    
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
                    bandwidth = cluster.estimate_bandwidth(np.asarray(bmuList), quantile=quant, n_samples=500)
                    cl = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmuList)
                
                if type=='DBSCAN':
                    cl = cluster.DBSCAN(eps=cutoff, min_samples=5).fit(bmuList)     
                
                if type=='KMeans':
                    cl= cluster.KMeans(n_clusters=numcl).fit(bmuList)

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

        
        if savefile==True:
            with open(os.path.join(path,type+'_clusters.'+filetype),
                      'w') as file:
                if filetype=='csv':
                    separator=','
                else: 
                    separator=' '
                for line in clusters:
                    for id in line: file.write(str(id)+separator)
                    file.write('\n')
        
        if printout==True or show==True:
            
            np.random.seed(0)
            printName=os.path.join(path,type+'_clusters.png')
            
            fig, ax = plt.subplots()
            
            for i in range(len(clusters)):
                randCl = "#%06x" % np.random.randint(0, 0xFFFFFF)
                xc,yc=[],[]
                for c in clusters[i]:
                    #again, invert y and x to be consistent with the previous maps
                    xc.append(bmuList[int(c)][0])
                    yc.append(self.height - bmuList[int(c)][1])
                ax.scatter(xc, yc, color=randCl, label='cluster'+str(i))

            plt.gca().invert_yaxis()
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)           
            ax.set_title('Clusters')
            ax.axis('off')

            if printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=600)
            if show==True:
                plt.show()
            plt.clf()   
            
        return clusters

        
class Neuron:

    """ Single Kohonen SOM Node class. """
    
    def __init__(self, x, y, numWeights, netHeight, netWidth, PBC, init_vect=[], weiArray=[]):
    
        """Initialise the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            numWeights (int): Length of the weights vector.
            netHeight (int): Network height, needed for periodic boundary conditions (PBC)
            netWidth (int): Network height, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            minVal(np.array, optional): minimum values for the weights found in the data
            maxVal(np.array, optional): maximum values for the weights found in the data
            pcaVec(np.array, optional): Array containing the two PCA vectors.
            weiArray (np.array, optional): Array containing the weights to give
                to the node if a file was loaded.

                
        """
    
        self.PBC=PBC
        self.pos=(x, y)
        self.lattice_pos = hx.coorToHex(x, y)
        self.weights = []

        self.netHeight=netHeight
        self.netWidth=netWidth

        if init_vect!=[]:
            #select randomly in the space spanned by the data
            self.weights= init_vect
        elif weiArray!=[]:
            for i in range(numWeights):
                self.weights.append(weiArray[i])

    
    def get_distance(self, vec):
    
        """Calculate the distance between the weights vector of the node and a given vector.

        Args:
            vec (np.array): The vector from which the distance is calculated.
            
        Returns: 
            (float): The distance between the two weight vectors.
        """
    
        sum=0
        if len(self.weights)==len(vec):
            for i in range(len(vec)):
                sum+=(self.weights[i]-vec[i])*(self.weights[i]-vec[i])
            return np.sqrt(sum)
        else:
            sys.exit("Error: dimension of nodes != input data dimension!")

    def get_nodeDistance(self, node):

        """Calculate the distance within the network between the node and another node.

        Args:
            node (Neuron): The node from which the distance is calculated.
            
        Returns:
            (float): The distance between the two nodes.
            
        """

        if self.PBC==True:

            """ Hexagonal Periodic Boundary Conditions """
            
            if self.netHeight%2==0:
                offset=0
            else: 
                offset=0.5

            return  np.min([np.sqrt((self.lattice_pos[0] - node.lattice_pos[0]) * (self.lattice_pos[0] - node.lattice_pos[0]) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1]) * (self.lattice_pos[1] - node.lattice_pos[1])),
                            #right
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth) * (self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1]) * (self.lattice_pos[1] - node.lattice_pos[1])),
                            #bottom 
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] + offset) * (self.lattice_pos[0] - node.lattice_pos[0] + offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                            #left
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth) * (self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1]) * (self.lattice_pos[1] - node.lattice_pos[1])),
                            #top 
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] - offset) * (self.lattice_pos[0] - node.lattice_pos[0] - offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                            #bottom right
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth + offset) * (self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth + offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                            #bottom left
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth + offset) * (self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth + offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] + self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                            #top right
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth - offset) * (self.lattice_pos[0] - node.lattice_pos[0] + self.netWidth - offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4)),
                            #top left
                            np.sqrt((self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth - offset) * (self.lattice_pos[0] - node.lattice_pos[0] - self.netWidth - offset) \
                                    + (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4) * (self.lattice_pos[1] - node.lattice_pos[1] - self.netHeight * 2 / np.sqrt(3) * 3 / 4))])
                        
        else:
            return np.sqrt((self.lattice_pos[0] - node.lattice_pos[0]) * (self.lattice_pos[0] - node.lattice_pos[0]) \
                           + (self.lattice_pos[1] - node.lattice_pos[1]) * (self.lattice_pos[1] - node.lattice_pos[1]))



    def update_weights(self, inputVec, sigma, lrate, bmu):

        """Update the node Weights.

        Args:
            inputVec (np.array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            lrate (float): The updated learning rate.
            bmu (Neuron): The best matching unit.
        """
    
        dist=self.get_nodeDistance(bmu)
        gauss=np.exp(-dist*dist/(2*sigma*sigma))

        #if gauss>0: #pointless
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - gauss*lrate*(self.weights[i]-inputVec[i])
        
def run_colorsExample(path='./'):   

    """Example of usage of SimpSOM: a number of vectors of length three
        (corresponding to the RGB values of a color) are used to briefly train a small network.
        Different example graphs are then printed from the trained network.     
    """ 

    """Try to create the folder"""
    if path!='./':
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
                
    raw_data =np.asarray([[1, 0, 0, 1, 0, 0, 1],[0,0, 0, 1,1,1,0],[1, 1, 0, 1, 0,0,1],[0,1,0,1,1,1,0],[1,0,1, 0, 0, 1, 1],[0,0,1, 1,0,1,1],[0.3, 0.2, 0.7, 0.1,0.2,0.2,0.5]])
    labels=['red','green','blue','yellow','magenta','cyan','indigo']

    print("Welcome to SimpSOM (Simple Self Organizing Maps) v1.3.4!\nHere is a quick example of what this library can do.\n")
    print("The algorithm will now try to map the following colors: ", end=' ')
    for i in range(len(labels)-1):
            print((labels[i] + ", "), end=' ') 
    print("and " + labels[-1]+ ".\n")
    
    net = somNet(20, 20, raw_data,PCI=True, PBC=False)
    
    net.colorEx=False
    net.train(0.01, 5000)

    print("Saving weights and a few graphs...", end=' ')
    net.save('colorExample_weights', path=path)
    net.nodes_graph(path=path)
    
    net.diff_graph(path=path)
    test=net.project(raw_data, labels=labels, path=path)
    print(test)
    net.cluster(raw_data, type='qthresh', path=path) 
    
    print("done!")


if __name__ == "__main__":

    run_colorsExample()
