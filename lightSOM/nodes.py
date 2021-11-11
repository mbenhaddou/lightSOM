import numpy as np

from sklearn.decomposition import PCA


class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass

from lightSOM.visualization.map_plot import generate_hex_lattice, generate_rect_lattice

class Nodes(object):

    def __init__(self, height, width, dim, lattice='rect', pbc=False):
        self.lattice = lattice
        mapsize=(height, width)

        self.PBC =pbc
        self.dim=dim
        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2)))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = self.mapsize[0]*self.mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False
        if self.lattice=='hexa':
            self.coordinates= generate_hex_lattice(height, width)
        elif self.lattice=='rect':
            self.coordinates=generate_rect_lattice(height, width)
        self.coordinates_x=self.coordinates[:,0].reshape(height, width)
        self.coordinates_y=self.coordinates[:,1].reshape(height, width)

        self.grid_x=np.arange(height)
        self.grid_y=np.arange(width)

        self.node_distances = [x[:] for x in [[0] * (width * height)] * (width * height)]
        for i, n1 in enumerate(self.coordinates):
            for j, n2 in enumerate(self.coordinates):
                if i < j:
                    self.node_distances[i][j] = self.get_nodeDistance(n1, n2)
                    self.node_distances[j][i] = self.node_distances[i][j]

        self.node_distances_reshaped=np.array(self.node_distances).reshape(*self.mapsize,*self.mapsize)

    def random_initialization(self, data, random_generator=None):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """

        # mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        # mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        # self.dim=data.shape[1]
        # if random_generator:
        # self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        # self.initialized = True
        # self.matrix = self.matrix.reshape(self.mapsize[0], self.mapsize[1], data.shape[1])
        if random_generator is None:
            random_generator = np.random.RandomState()

        self.matrix = random_generator.rand(*self.mapsize, data.shape[1])*2-1
        self.matrix /= np.linalg.norm(self.matrix, axis=-1, keepdims=True)

    def pca_linear_initialization(self, data):
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
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components) # RandomizedPCA is deprecated.
        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True
        self.matrix = self.matrix.reshape(self.mapsize[0], self.mapsize[1], data.shape[1])

    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        return self.lattice_distances[node_ind]

    def _rect_dist(self, node_ind):
        """
        Calculates the distance of the specified node2 to the other nodes in the
        matrix, generating a distance matrix

        Ej. The distance matrix for the node_ind=5, that corresponds to
        the_coord (1,1)
           array([[2, 1, 2, 5],
                  [1, 0, 1, 4],
                  [2, 1, 2, 5],
                  [5, 4, 5, 8]])

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        rows = self.mapsize[0]
        cols = self.mapsize[1]
        dist = None

        # bmu should be an integer between 0 to no_nodes
        if 0 <= node_ind <= (rows*cols):
            node_col = int(node_ind % cols)
            node_row = int(node_ind / cols)
        else:
            raise InvalidNodeIndexError(
                "Node index '%s' is invalid" % node_ind)

        if rows > 0 and cols > 0:
            r = np.arange(0, rows, 1)[:, np.newaxis]
            c = np.arange(0, cols, 1)
            dist2 = (r-node_row)**2 + (c-node_col)**2

            dist = dist2.ravel()
        else:
            raise InvalidMapsizeError(
                "One or both of the map dimensions are invalid. "
                "Cols '%s', Rows '%s'".format(cols=cols, rows=rows))

        return dist

    def get_nodeDistance(self, node1, node2):

        """Calculate the distance within the network between the node2 and another node2.

        Args:
            node2 (somNode): The node2 from which the distance is calculated.

        Returns:
            (float): The distance between the two nodes.

        """
        height, width = self.mapsize
        if self.PBC == True:

            """ Hexagonal Periodic Boundary Conditions """
            hexa_height=height * 2 / np.sqrt(3) * 3 / 4
            if height % 2 == 0:
                offset = 0
            else:
                offset = 0.5

            return np.min([np.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0]) \
                                   + (node1[1] - node2[1]) * (node1[1] - node2[1])),
                           # right
                           np.sqrt(
                               (node1[0] - node2[0] + width) * (node1[0] - node2[0] + width) \
                               + (node1[1] - node2[1]) * (node1[1] - node2[1])),
                           # bottom 
                           np.sqrt((node1[0] - node2[0] + offset) * (node1[0] - node2[0] + offset) \
                                   + (node1[1] - node2[1] + hexa_height) * (
                                           node1[1] - node2[1] + hexa_height)),
                           # left
                           np.sqrt(
                               (node1[0] - node2[0] - width) * (node1[0] - node2[0] - width) \
                               + (node1[1] - node2[1]) * (node1[1] - node2[1])),
                           # top 
                           np.sqrt((node1[0] - node2[0] - offset) * (node1[0] - node2[0] - offset) \
                                   + (node1[1] - node2[1] - hexa_height) * (
                                           node1[1] - node2[1] - hexa_height)),
                           # bottom right
                           np.sqrt((node1[0] - node2[0] + width + offset) * (
                                   node1[0] - node2[0] + width + offset) \
                                   + (node1[1] - node2[1] + hexa_height) * (
                                           node1[1] - node2[1] + hexa_height)),
                           # bottom left
                           np.sqrt((node1[0] - node2[0] - width + offset) * (
                                   node1[0] - node2[0] - width + offset) \
                                   + (node1[1] - node2[1] + hexa_height) * (
                                           node1[1] - node2[1] + hexa_height)),
                           # top right
                           np.sqrt((node1[0] - node2[0] + width - offset) * (
                                   node1[0] - node2[0] + width - offset) \
                                   + (node1[1] - node2[1] - hexa_height) * (
                                           node1[1] - node2[1] - hexa_height)),
                           # top left
                           np.sqrt((node1[0] - node2[0] - width - offset) * (
                                   node1[0] - node2[0] - width - offset) \
                                   + (node1[1] - node2[1] - hexa_height) * (
                                           node1[1] - node2[1] - hexa_height))])

        else:
            return np.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0]) \
                           + (node1[1] - node2[1]) * (node1[1] - node2[1]))


