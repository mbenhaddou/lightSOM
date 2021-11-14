"""
Hexagonal tiling library

F. Comitani @2017 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import math
from lightSOM.visualization.resources import markers, colors
from matplotlib.collections import RegularPolyCollection

def coorToHex(x,y):

    """Convert Cartesian coordinates to hexagonal tiling coordinates.

        Args:
            x (float): position along the x-axis of Cartesian coordinates.
            y (float): position along the y-axis of Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
                
    """


    newy=y*2/np.sqrt(3)*3/4
    newx=x
    if y%2: newx+=0.5
    return [newx,newy]    


def generate_hex_lattice(n_rows, n_columns):
    x_coord, y_coord = [], []
    for i in range(n_rows):
        for j in range(n_columns):
            hex_coords=coorToHex(i, j)
            x_coord.append(hex_coords[0])
            y_coord.append(hex_coords[1])
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates

def generate_rect_lattice(n_rows, n_columns):

    x_coord, y_coord = [], []
    for i in range(n_rows):
        for j in range(n_columns):
            x_coord.append(i)
            y_coord.append(j)
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates

def plot_map(fig, centers, weights, titles=[], shape=[1, 1], cmap = plt.get_cmap('viridis'), show_colorbar=True, lattice="hexa", alpha=1):
    
    """Plot an hexagonal grid based on the nodes positions and color the tiles
       according to their weights.

        Args:
            fig (matplotlib figure object): the figure on which the hexagonal grid will be plotted.
            centers (list, float): array containing couples of coordinates for each cell 
                to be plotted in the Hexagonal tiling space.
            weights (list, float): array contaning informations on the weigths of each cell, 
                to be plotted as colors.
            
        Returns:
            ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
                
    """

    if isinstance(cmap, str):
        cmap=plt.get_cmap(cmap)

    orientation=0
    radius=.95/np.sqrt(3)
    num_verices=6
    if lattice=="rect":
        orientation = np.pi/4
        radius = .95 / np.sqrt(2)
        num_verices = 4

    xpoints = [x[0]  for x in centers]
    ypoints = [x[1]  for x in centers]
    patches = []
    weights=np.array(weights)
    if len(weights.shape)<3:
        weights=np.expand_dims(weights, 2)

    if not isinstance(titles, list):
        titles=[titles]
    if len(titles) != weights.shape[0]:
        titles = [""] * weights.shape[2]
    for i, title in zip(range(weights.shape[0]), titles):

        ax = fig.add_subplot(int(shape[0]), int(shape[1]), i + 1, aspect='equal')
        # Get pixel size between two data points
        ax.scatter(xpoints, ypoints, s=0.0, marker='s')
        ax.axis([min(xpoints) - 1., max(xpoints) + 1.,
                 min(ypoints) - 1., max(ypoints) + 1.])

        weight = weights[i, :, :]
        if any(isinstance(el, np.ndarray) for el in weight) and len(weight[0])==3:

            for x,y,w in zip(xpoints,ypoints,weight):
                hexagon = RegularPolygon((x,y), numVertices=num_verices, radius=radius,
                                    orientation=orientation,
                                    facecolor=w)
                ax.add_patch(hexagon)

        else:
            for x,y,w in zip(xpoints,ypoints,weight):
                if isinstance(w, np.ndarray):
                    w=w[0]
                hexagon = RegularPolygon((x,y), numVertices=num_verices, radius=radius,
                                    orientation=orientation)
                patches.append(hexagon)

            p = PatchCollection(patches)
            p.cmap=cmap
            p.set_alpha(alpha=alpha)
            p.set_array(np.array([w[0] for w in weight]))
            ax.add_collection(p)


            ax.axis('off')
            ax.autoscale_view()
            if show_colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                cbar = plt.colorbar(p, cax=cax)
    #            cbar.set_label('Weights Difference', size=30, labelpad=30)
                cbar.ax.tick_params(labelsize=20/(np.log2(len(titles)) if len(titles)>1 else 1))
        ax.axis('off')
        if title !="":
            ax.set_title(title, size=30/(np.log2(len(titles)) if len(titles)>1 else 1))
        ax.autoscale_view()
    return ax



def plot_projection_map(fig, centers, matrix, data_map=None, target=None, names=None, shrink=1.0, lattice="rect", cmap='viridis'):



    orientation=0
    radius=.95/np.sqrt(3)
    num_verices=6
    if lattice=="rect":
        orientation = np.pi/4
        radius = .95 / np.sqrt(2)
        num_verices = 4


    if isinstance(cmap, str):
        cmap=plt.get_cmap(cmap)

    xpoints = [x[0]  for x in centers]
    ypoints = [x[1]  for x in centers]
    patches = []

    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(xpoints, ypoints, s=0.0, marker='s')
    ax.axis([min(xpoints) - 1., max(xpoints) + 1.,
                 min(ypoints) - 1., max(ypoints) + 1.])
    xy_pixels = ax.transData.transform(np.vstack([xpoints, ypoints]).T)
    xpix, ypix = xy_pixels.T

    # discover radius and hexagon
    apothem = shrink * (xpix[1] - xpix[0]) / math.sqrt(3)
    area_inner_circle = math.pi * (apothem ** 2)
    used_colors=[]
    for x, y, w in zip(xpoints, ypoints, matrix):
        used_colors.append(int(w))
        hexagon = RegularPolygon((x, y), numVertices=num_verices, radius=radius,
                                 orientation=orientation, facecolor=colors[int(w)])
        patches.append(hexagon)
    used_colors=list(set(used_colors))
    p = PatchCollection(patches, match_original=True)
    p.cmap = cmap

#    p.set_array(np.array(matrix))

    ax.add_collection(p)
    ax.autoscale_view()
    legend_elements=[Patch(facecolor=colors[clr], edgecolor='w', label=l) for clr, l in enumerate(target) if clr in used_colors]

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))

    ax.axis('off')


    return ax






    #
    #
    # for x, y in zip(xpoints, ypoints):
    #     hexagon = RegularPolygon((x, y), numVertices=num_verices, radius=radius,
    #                              orientation=orientation)
    #     patches.append(hexagon)
    #
    # p = PatchCollection(patches)
    # p.cmap = cmap
    #
    # p.set_array(np.array(weights))
    # ax.add_collection(p)
    #
    # ax.axis('off')
    # ax.autoscale_view()
    # legend_elements = [Patch(facecolor=clr,
    #                              edgecolor='w',
    #                              label=l) for l, clr in zip(weights, target)]
    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    # ax.grid()
    #
