'''
This implementation is based on the rbf_graph implementation in the SGTL python package. 
More information can be found here: https://sgtl.readthedocs.io/en/latest/getting-started.html
'''

import math
from typing import List

import scipy
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import pandas as pd



class Graph: 
    # def __init__(self, similarity_function, variance, dropping_edges = True): 
    #     # Name of the selected similarity function
    #     self.sim_function = similarity_function
    #     # Edges are dropped when weight below a threshold to avoid near zero edges.
    #     self.dropping_edges = dropping_edges

    #     # if similarity_function == "rbf": 
    #     #     return anySparseGraph(rbf_kernel)



    def __init__(self, adj_mat):
        """
        Initialise the graph with an adjacency matrix.

        :param adj_mat: A sparse scipy matrix.
        """
        # The graph is represented by the sparse adjacency matrix. We store the adjacency matrix in two sparse formats.
        # We can assume that there are no non-zero entries in the stored adjacency matrix.
        self.adj_mat = adj_mat.tocsr()
        self.adj_mat.eliminate_zeros()
        self.lil_adj_mat = adj_mat.tolil()

        # For convenience, and to speed up operations on the graph, we precompute the degrees of the vertices in the
        # graph.
        self.degrees = adj_mat.sum(axis=0).tolist()[0]
        self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))
        
    def __add__(self, other):
        """
        Adding two graphs requires that they have the same number of vertices. the sum of the graphs is simply the graph
        constructed by adding their adjacency matrices together.

        You can also just add a sparse matrix to the graph directly.
        """
        if isinstance(other, scipy.sparse.spmatrix):
            if other.shape[0] != self.number_of_vertices():
                raise AssertionError("Graphs must have equal number of vertices.")
            # return Graph(self.adjacency_matrix() + other)
            return Graph(self.adjacency_matrix() + other)

        if other.number_of_vertices() != self.number_of_vertices():
            raise AssertionError("Graphs must have equal number of vertices.")

        # return Graph(self.adjacency_matrix() + other.adjacency_matrix())
        return Graph(self.adjacency_matrix() + other.adjacency_matrix())

    def number_of_vertices(self) -> int:
        """The number of vertices in the graph."""
        return self.adjacency_matrix().shape[0]

    def inverse_sqrt_degree_matrix(self):
        """Construct the square root of the inverse of the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(self.inv_sqrt_degrees, [0], self.number_of_vertices(), self.number_of_vertices(),
                                    format="csr")

    def adjacency_matrix(self):
        """
        Return the Adjacency matrix of the graph.
        """
        return self.adj_mat
    
    def laplacian_matrix(self):
        """
        Construct the Laplacian matrix of the graph. The Laplacian matrix is defined to be

        .. math::
           L = D - A

        where D is the diagonal degree matrix and A is the adjacency matrix of the graph.
        """
        return self.degree_matrix() - self.adjacency_matrix()
    
    def normalised_laplacian_matrix(self):
        """
        Construct the normalised Laplacian matrix of the graph. The normalised Laplacian matrix is defined to be

        .. math::
            \\mathcal{L} = D^{-1/2} L D^{-1/2} =  I - D^{-1/2} A D^{-1/2}

        where I is the identity matrix and D is the diagonal degree matrix of the graph.
        """
        return self.inverse_sqrt_degree_matrix() @ self.laplacian_matrix() @ self.inverse_sqrt_degree_matrix()
    # def sparseGraph(self, adj_mat):
    #     """
    #     Initialise the graph with an adjacency matrix.

    #     :param adj_mat: A sparse scipy matrix.
    #     """
    #     # The graph is represented by the sparse adjacency matrix. We store the adjacency matrix in two sparse formats.
    #     # We can assume that there are no non-zero entries in the stored adjacency matrix.
    #     self.adj_mat = adj_mat.tocsr()
    #     self.adj_mat.eliminate_zeros()
    #     self.lil_adj_mat = adj_mat.tolil()

    #     # For convenience, and to speed up operations on the graph, we precompute the degrees of the vertices in the
    #     # graph.
    #     self.degrees = adj_mat.sum(axis=0).tolist()[0]
    #     self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
    #     self.sqrt_degrees = list(map(math.sqrt, self.degrees))
    #     self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

    def degree_matrix(self):
        """Construct the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(
            self.degrees, [0], self.number_of_vertices(), self.number_of_vertices(), format="csr")




    


# def fullyConnected(data, kernelName, variance=1, threshold=0.1):
def fullyConnected(data, kernelName, threshold=0.1):
    """
    Construct a graph from the given data using a radial basis function.
    The weight of the edge between each pair of data points is given by the Gaussian kernel function

    .. math::
        k(x_i, x_j) = \\exp\\left( - \\frac{||x_i - x_j||^2}{2 \\sigma^2} \\right)

    where :math:`\\sigma^2` is the variance of the kernel and defaults to 1. Any weights less than the specified
    threshold (default :math:`0.1`) are discarded and no edge is added to the graph.

    :param data: a sparse matrix with dimension :math:`(n, d)` containing the raw data
    :param variance: the variance of the gaussian kernel to be used
    :param threshold: the threshold under which to ignore the weights of an edge. Set to 0 to keep all edges.
    :return: an `sgtl.Graph` object
    """
    kernel, hyperParam = parseKernelName(kernelName)
    print(hyperParam)
    # Get the maximum distance which corresponds to the threshold specified.
    if threshold <= 0:
        # Handle the case when threshold is equal to 0 - need to create a fully connected graph.
        max_distance = float('inf')
    else:
        # max_distance = math.sqrt(-2 * variance * math.log(threshold))
        max_distance = math.sqrt(-2 * hyperParam * math.log(threshold))


    print(data.shape)
    # Create the nearest neighbours for each vertex using sklearn - create a data structure with all neighbours
    # which are close enough to be above the given threshold.
    distances, neighbours = NearestNeighbors(radius=max_distance).fit(data).radius_neighbors(data)

 


    # Now, let's construct the adjacency matrix of the graph iteratively
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]))
    for vertex in range(data.shape[0]):
        # Get the neighbours of this vertex
        for i, neighbour in enumerate(neighbours[vertex]):
            if neighbour != vertex:
                distance = distances[vertex][i]
                # weight = math.exp(- (distance**2) / (2 * variance))
                weight = kernel(distance, hyperParam)
                adj_mat[vertex, neighbour] = weight
                adj_mat[neighbour, vertex] = weight

    # self.sparseGraph(adj_mat)
    # return self
    return Graph(adj_mat)



def parseKernelName(kernelName):
    print(kernelName)
    if kernelName[:3] == "rbf":
        return rbf, int(kernelName[4:])
    elif kernelName[:3] == "lpl":
        return laplacian, int(kernelName[4:])
    # elif kernelName[:3] == "sigmoid":
    #     return sigmoid, int(kernelName[4:])
    # elif kernelName[:3] == "chi2":
    #     return chi2, int(kernelName[4:])
    elif kernelName[:3] == "inv":
        return inverse_euclidean, int(kernelName[4:])
    else:
        print("Wrong kernel name used")
        return None

#############################################
### KERNELS

def rbf(distance, variance):
    return math.exp(- (distance**2) / (2 * variance))

def laplacian(distance,variance):
    return math.exp(- (distance) / (math.sqrt(variance)))


def inverse_euclidean(distance,variance):
    return 1 / (1 + distance**variance)


# def sigmoid(x, y, gamma: float = 1.0, coef0: float = 0.0) -> float:
#     # Needs γ and c inside 'nice' ranges to stay in the valid kernel regime.
#     x, y = np.asarray(x), np.asarray(y)
#     return math.tanh(gamma * float(np.dot(x, y)) + coef0)

# def chi2(distance, gamma):
#     # Implemented using distance instead of vectors
#     return math.exp(-gamma * distance)


# need to access the actual neighbours for these below

# Cosine 
# Polynomial