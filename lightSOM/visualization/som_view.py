from matplotlib import pyplot as plt
import numpy as np
import math, os
from lightSOM.visualization.map_plot import plot_map, plot_projection_map
from kdmt.matrix import euclidean_distance
from collections import Counter
import matplotlib
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from lightSOM.visualization.resources import markers, colors
markers={i:m for i,m in enumerate(markers.keys())}
class SOMView():

    def __init__(self, som, width, height, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, *args, **kwargs):

        self.width = width
        self.height = height

        self.show_axis = show_axis
        self.packed = packed
        self.text_size = text_size
        self.show_text = show_text
        self.som=som

        self._fig = None

    def __del__(self):
        self._close_fig()


    def build_diff_matrix(self, som, distance=1.01, denormalize=False):
        nodes_distances = np.array(som.nodes.node_distances)
        Diffmatrix = np.zeros((som.nodes.nnodes, 1))
        codebook = som.nodes.matrix.reshape(-1, som.data.shape[1])
        if denormalize and self.som.normalizer is not None:
            vector = som.normalizer.denormalize(som.data_raw, codebook)
        else:
            vector = codebook

        for i in range(som.nodes.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            neighborbor_ind = nodes_distances[i][0:] <= distance
            neighborbor_weights = vector[neighborbor_ind]
            neighborbor_dists = euclidean_distance(
                codebook_i, neighborbor_weights)
            Diffmatrix[i] = np.float(np.sum(neighborbor_dists) / (neighborbor_dists.shape[1] - 1))

        return Diffmatrix.reshape(som.nodes.nnodes)

    def _set_labels(self, cents, ax, labels, onlyzeros, fontsize, lattice="hexa"):

        hex = lattice=="hexa"

        for i, txt in enumerate(labels):
            if onlyzeros == True:
                if txt > 0:
                    txt = ""
            c = cents[i] if hex else (cents[i, 1], cents[-(i + 1), 0])
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)

    def _close_fig(self):
        if self._fig:
            plt.close(self._fig)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        self._fig = plt.figure(figsize=(self.width, self.height))
        self._fig.patch.set_facecolor('white')
        plt.title(self.title, size=50)
        plt.axis('off')
        plt.rc('font', **{'size': self.text_size})

    def save(self, filename, transparent=False, bbox_inches='tight', dpi=400):
        self._fig.savefig(filename, transparent=transparent, dpi=dpi, bbox_inches=bbox_inches)
        return


    def _calculate_figure_params(self, which_dim, col_sz):

        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            dim = self.som.nodes.dim
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = self.som.nodes.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.arange(0, dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap

        elif type(which_dim) == int:
            dim = 1
            msz_row, msz_col = self.som.nodes.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = 16, 16 * ratio_hitmap

        elif type(which_dim) == list:
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = self.som.nodes.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap
        elif which_dim == 'none':
            dim = 1
            row_sz = 1
            col_sz=1
            msz_row, msz_col = self.som.nodes.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap
        no_row_in_plot = math.ceil(dim / col_sz)  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)


    def show(self, matrix, file_name='nodesColors.png', which_dim='all', cmap = plt.get_cmap('viridis'),
             col_size=1, save=True, path='.', show_colorbar=True, colorEx=False, alpha=1):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(which_dim, col_size)
        self.prepare()
        centers = [[node[0], node[1]] for node in self.som.nodes.coordinates]
        names = []
        colors = []
        if colorEx:
            colors.append([list(node) for node in matrix.reshape(-1, 3)])
            ax = plot_map(self._fig, centers, colors, 'Node Grid w Color Features')
        else:
            weights = matrix
            if which_dim == 'all':
#                names = ["Feature_"+ f for f in self.som.features_names[0]]
                for which_dim in range(len(weights.reshape(-1, self.som.nodes.dim)[0])):
                    names.append("Feature_" + self.som.features_names[0][which_dim])
                    colors.append([[node[which_dim]] for node in weights.reshape(-1, self.som.nodes.dim)])
            elif type(which_dim) == int:
                names.append(["Feature_"+self.som.features_names[0][which_dim]])
                colors.append([[node[which_dim]] for node in weights.reshape(-1, self.som.nodes.dim)])
            elif type(which_dim) == list:
                for dim in which_dim:
                    names.append("Feature_" + self.som.features_names[0][dim])
                    colors.append([[node[dim]] for node in weights.reshape(-1, self.som.nodes.dim)])
            elif which_dim=='none':
                names=[]
                colors.append(matrix)
            ax=plot_map(fig=self._fig, centers=centers, weights=colors, titles=names,
                            shape=[no_row_in_plot, no_col_in_plot], cmap=cmap, show_colorbar=show_colorbar, lattice=self.som.nodes.lattice, alpha=alpha)



        if save==True:
            if colorEx:
                printName = os.path.join(path, 'nodesColors.png')
            else:
                printName=os.path.join(path, file_name)
            self.save(printName, bbox_inches='tight', dpi=300)

        return ax

    def plot_nodes_maps(self,  file_name='nodes_features.png',which_dim='all', cmap = plt.get_cmap('viridis'),
             col_size=1, denormalize=False, save=True, path='.', show=False, colorEx=False):
        self.title="Futures Map"
        weights=self.som.nodes.matrix
        if denormalize and self.som.normalizer is not None:
            weights=self.som.normalizer.denormalize(self.som.data_raw, weights)

        return self.show(weights, file_name,which_dim, cmap,
             col_size, save, path, show, colorEx)


    def plot_features_map(self,file_name='features_map.png', cmap = plt.get_cmap('viridis'),
             col_size=1, save=True, path='.', show_colorbar=False, colorEx=False):
        self.title="Features Map"
        self.prepare()
        features=np.zeros(self.som.nodes.mapsize)

        for i in np.arange(self.som.nodes.matrix.shape[0]):
            for j in np.arange(self.som.nodes.matrix.shape[1]):
                features[i][j] = np.argmax(self.som.nodes.matrix[i, j, :])

        centers = [[node[0], node[1]] for node in self.som.nodes.coordinates]
        return plot_projection_map(self._fig, centers, features.ravel(), target=self.som.features_names[0])

    def plot_diffs(self, cmap = plt.get_cmap('viridis'),
                   col_size=1, denormalize=False, save=True, path='.', show_colorbar=False, annotate="none", file_name='nodes_differences', colorEx=False):

        diffs=self.build_diff_matrix(self.som, denormalize=denormalize)
        self.title="Difference Map"
        alpha=1
        if annotate.lower() !="none":
            alpha=0.5
        ax=self.show(diffs, file_name, "none", cmap, col_size, save, path, show_colorbar, colorEx, alpha=alpha)

        if annotate=="target" and self.som.target is not None:
            self._set_label_markers(ax, self.som.data, self.som.target)
        elif annotate=="samples" and self.som.target is not None:
            self._annotate_samples(ax, self.som.data, self.som.target)
        elif annotate=="names" and self.som.target is not None and self.som.index is not None:
            self._annotate_samples_with_names(ax, self.som.data, self.som.target, self.som.index)

    def plot_cluster_map(self, data=None, n_clusters=4, anotate=False, onlyzeros=False, labelsize=7, cmap = plt.get_cmap('Pastel1')):
        org_w = self.width
        org_h = self.height
        self.title="Cluster Map"
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(1, 1)
        self.width /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        self.height /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        centers = [[node[0], node[1]] for node in self.som.nodes.coordinates]
        clusters=[]
        clusters.append(self.som.cluster(n_clusters))


        # codebook = getattr(som, 'cluster_labels', som.cluster())
        msz = self.som.nodes.mapsize

        self.prepare()
        if data:
            proj = self.som.project(data)
            cents = self.som.bmu_ind_to_xy(proj)


        ax=plot_map(self._fig, centers, clusters, cmap=cmap, show_colorbar=False, lattice=self.som.nodes.lattice)
        if anotate:
            self._set_labels(centers, ax, clusters[0], onlyzeros, labelsize, lattice=self.som.nodes.lattice)
        self.save("clusres.png", bbox_inches='tight', dpi=300)


    def plot_hits_map(self, anotate=True, onlyzeros=False, labelsize=7, cmap=plt.get_cmap("jet")):
        org_w = self.width
        org_h = self.height
        self.title="Hits Map"
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(1, 1)
        self.width /=  (self.width/org_w) if self.width > self.height else (self.height/org_h)
        self.height /=  (self.width / org_w) if self.width > self.height else (self.height / org_h)

        cnts = Counter(self.som.project(self.som.data))
        cnts = [cnts.get((x, y), 0) for x in range(self.som.nodes.mapsize[0]) for y in range(self.som.nodes.mapsize[1])]
        counts=[]
        counts.append(cnts)
        # mp = np.array(counts).reshape(self.som.nodes.mapsize[0],
        #                               self.som.nodes.mapsize[1])
        centers = [[node[0], node[1]] for node in self.som.nodes.coordinates]

        norm = matplotlib.colors.Normalize(
                vmin=0,
                vmax=np.max(np.array(counts).flatten()),
                clip=True)

        self.prepare()

        if self.som.nodes.lattice == "rect":

            ax = plt.gca()
            if anotate:
                self._set_labels(centers, ax, counts, onlyzeros, labelsize)

            pl = plt.pcolor(mp[::-1], norm=norm, cmap=cmap)

            plt.axis([0, self.som.codebook.mapsize[1], 0, self.som.codebook.mapsize[0]])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)

            #plt.show()
        elif self.som.nodes.lattice == "hexa":
            ax=plot_map(self._fig, centers, counts, cmap=cmap, show_colorbar=False)

#            ax, cents = plot_hex_map(mp[::-1], colormap=cmap, fig=self._fig)
            if anotate:
                self._set_labels(centers, ax, counts[0], onlyzeros, labelsize, hex=True)

#                self._set_labels(cents, ax, counts, onlyzeros, labelsize, hex=True)
            self.save("hitmap.png", bbox_inches='tight', dpi=300)


    def _set_label_markers(self, ax, data, target):
        # Plotting the response for each pattern in the iris dataset
        # different colors and markers for each label
        target_dict={u:i for i, u in enumerate(np.unique(target))}
        for cnt, xx in enumerate(data):
            w = [self.som.nodes.coordinates_x[self.som.find_bmu(xx)[0]][self.som.find_bmu(xx)[1]], self.som.nodes.coordinates_y[self.som.find_bmu(xx)[0]][self.som.find_bmu(xx)[1]]]  # getting the winner
            # palce a marker on the winning position for the sample xx
            ax.plot(w[0], w[1], markers[target_dict[target[cnt]]], markerfacecolor='None',
                     markeredgecolor=colors[target_dict[target[cnt]]], markersize=12, markeredgewidth=2)

    def plot_label_proportions(self, data=None, target=None, target_map=None):
        import matplotlib.gridspec as gridspec

        if data is None:
            data=self.som.data
        if target is None:
            target=self.som.target
#        target_dict={u:i for i, u in enumerate(np.unique(target))}
        labels_map = self.som.labels_map(data, [t for t in target])
        if target_map is None:
            target_map={i:u for i, u in enumerate(np.unique(target))}
        fig = plt.figure(figsize=(9, 9))
        the_grid = gridspec.GridSpec(*self.som.nodes.mapsize, fig)
        for position in labels_map.keys():
            label_fracs = [labels_map[position][l] for l in target_map.values()]
            plt.subplot(the_grid[self.som.nodes.mapsize[0] - 1 - position[1],
                                 position[0]], aspect=1)
            patches, texts = plt.pie(label_fracs)

        plt.legend(patches, target_map.values(), bbox_to_anchor=(3.5, 6.5), ncol=3)
        plt.savefig('./som_seed_pies.png')
        return plt

    def _annotate_samples(self, ax, data, target=None):

        w_x, w_y = zip(*[[self.som.nodes.coordinates_x[self.som.find_bmu(xx)[0]][self.som.find_bmu(xx)[1]], self.som.nodes.coordinates_y[self.som.find_bmu(xx)[0]][self.som.find_bmu(xx)[1]]] for xx in data])
        w_x = np.array(w_x)
        w_y = np.array(w_y)
        if target is not None:
            target=np.array(target)
            for c in np.unique(target):
                idx_target = target == c
                ax.scatter(w_x[idx_target] + (np.random.rand(np.sum(idx_target))) * .5,
                            w_y[idx_target] + (np.random.rand(np.sum(idx_target))) * .5,
                            s=50, c=colors[c], label=target[c])
        ax.legend(loc='upper right')
        ax.grid()


    def _annotate_samples_with_names(self, ax, data, target=None, names=None):
        data_map={}
        if names is not None:
            data_map = self.som.labels_map(data, names)

        color_dict={u:colors[i] for i, u in enumerate(np.unique(target))}
        target_color_dict={name:color_dict[target[i]] for i,name in enumerate(names)}

        for p, data_item in data_map.items():
            data_item = list(data_item)
            x = p[0] + .1
            y = p[1] - .8
            for i, c in enumerate(data_item):
                off_set = (i + 1) / len(data_item) - 0.05
                ax.text(x, y + off_set, c, c=target_color_dict[c], fontsize=10)
        legend_elements = [Patch(facecolor=clr,
                                 edgecolor='w',
                                 label=l) for l, clr in color_dict.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
        ax.grid()

    def plot_frequencies(self, data=None, cmap=plt.get_cmap('OrRd'), file_name='frequency_plot.png', col_size=1, save=True, path='.', show=False):


        self.title = "Frequency Map"
        if data is None:
            data=self.som.data
        frequencies = self.som.activation_frequency(data).reshape(self.som.nodes.nnodes,)

        ax = self.show(frequencies, file_name, "none", cmap, col_size, save, path, show, colorEx=False)


    def plot_training_errors(self):
        if self.som._quantization_errors !=[]:
            plt.plot(np.arange(self.som.epochs), self.som._quantization_errors, label='quantization error')
            plt.plot(np.arange(self.som.epochs), self.som._topological_errors, label='topographic error')
            plt.ylabel('quantization error')
            plt.xlabel('iteration index')
            plt.title('Training quality evolution')
            plt.legend()
            plt.show()
