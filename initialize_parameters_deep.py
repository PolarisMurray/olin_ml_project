import numpy as np
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimenstions of each layer
    in our network

    Returns:
    parameters -- python dictionary containing parameters "W1", 
    "b1", ..., "WL", "bL":

    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims)
    """
    np.rando.seed(3)

    parameters = {}

    l = len(layer_dims) # number of layers in the network

    for i in range(1, l):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    
    return parameters

