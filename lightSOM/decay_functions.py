import numpy as np

def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


def exponential_decay(learning_rate, t, max_iter):
    return learning_rate * np.exp(-t / max_iter)
