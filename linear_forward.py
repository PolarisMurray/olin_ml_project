
def linear_forward(A, W, b):
    """
    The linear part of a layer's forward propagation.

    Arguments:

    A -- activations from previous layer (or input data): (size of the previous
    layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b"; stored for computing 
    the backward
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache