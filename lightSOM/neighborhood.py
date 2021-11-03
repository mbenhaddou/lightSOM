import numpy as np


def get_neighberhood(name):


    neig_functions = {'gaussian': gaussian,
                      'mexican_hat': mexican_hat,
                      'bubble': bubble,
                      'triangle': triangle}
    if name not in neig_functions:
        raise ValueError("neighberhood function "+name+" is not supported. Options are ['gaussian', 'mexican_hat', "
                                                       "'bubble', 'triangle']")
    return neig_functions[name]
def gaussian(nodes, c, sigma):
    """Returns a Gaussian centered in c."""
    d = 2* sigma * sigma
    # ax = np.exp(-np.power(nodes.coordinates_x - nodes.coordinates_x.T[c], 2) / d)
    # ay = np.exp(-np.power(nodes.coordinates_y - nodes.coordinates_y.T[c], 2) / d)
    # return (ax * ay).T  # the external product gives a matrix
    return np.exp(-np.power(nodes.node_distances_reshaped[c], 2)/d)

def mexican_hat(nodes, c, sigma):
    """Mexican hat centered in c."""
    p = np.power(nodes.coordinates_x - nodes.coordinates_x.T[c], 2) + np.power(nodes.coordinates_y - nodes.coordinates_y.T[c], 2)
    d = 2 * sigma * sigma
    return (np.exp(-p / d) * (1 - 2 / d * p)).T




def bubble(nodes, c, sigma):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    ax = np.logical_and(nodes.grid_x > c[0] - sigma,
                     nodes.grid_x < c[0] + sigma)
    ay = np.logical_and(nodes.grid_y > c[1] - sigma,
                     nodes.grid_y < c[1] + sigma)
    return np.outer(ax, ay) * 1.


def triangle(nodes, c, sigma):
    """Triangular function centered in c with spread sigma."""
    triangle_x = (-abs(c[0] - nodes.grid_x)) + sigma
    triangle_y = (-abs(c[1] - nodes.grid_y)) + sigma
    triangle_x[triangle_x < 0] = 0.
    triangle_y[triangle_y < 0] = 0.
    return np.outer(triangle_x, triangle_y)
