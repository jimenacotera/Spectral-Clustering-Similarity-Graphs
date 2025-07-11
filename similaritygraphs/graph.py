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
    

    def total_volume(self) -> float:
        """The total volume of the graph."""
        return sum(self.degrees)

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


    def degree_matrix(self):
        """Construct the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(
            self.degrees, [0], self.number_of_vertices(), self.number_of_vertices(), format="csr")



#############################################
### FULLY CONNECTED SIMILARITY GRAPH AND KERNELS

def fullyConnected(data, kernelName, threshold=0.1):
    """
    :param data: a sparse matrix with dimension :math:`(n, d)` containing the raw data
    :param kernelName: kernel description
    :param threshold: the threshold under which to ignore the weights of an edge. Set to 0 to keep all edges.
    :return: an `sgtl.Graph` object
    """
    # kernel, hyperParam, max_distance = parseKernelName(kernelName, threshold)
    kernel, hyperParam = parseKernelName(kernelName)
    # Get the maximum distance which corresponds to the threshold specified.
    if threshold <= 0:
        # Handle the case when threshold is equal to 0 - need to create a fully connected graph.
        max_distance = float('inf')
    elif kernel == inverse_euclidean: 
        max_distance = math.sqrt(-2 * 10 * math.log(threshold))
    else:
        # max_distance = math.sqrt(-2 * variance * math.log(threshold))
        max_distance = math.sqrt(-2 * hyperParam * math.log(threshold))

    # Create the nearest neighbours for each vertex using sklearn 
    # Neighbours closer than max_distance from given threshold
    distances, neighbours = NearestNeighbors(radius=max_distance).fit(data).radius_neighbors(data)


    # Construct the adjacency matrix of the graph iteratively
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

    return Graph(adj_mat)



def parseKernelName(kernelName):
    """
    Helper to get kernel name and hyperparameters
    """
    print(kernelName)
    if kernelName[:3] == "rbf":
        return rbf, float(kernelName[4:])
    elif kernelName[:3] == "lpl":
        return laplacian, int(kernelName[4:])
    elif kernelName[:3] == "inv":
        return inverse_euclidean, kernelName[4:]
    else:
        print("Wrong kernel name used")
        return None

def rbf(distance, hyperparameter ):
    """
    hyperparam <- variance
    """
    return math.exp(- (distance**2) / (2 * hyperparameter))

def laplacian(distance,hyperparameter):
    """
    hyperparam <- variance
    """
    return math.exp(- (distance) / (math.sqrt(hyperparameter)))

def inverse_euclidean(distance,hyperparameter):
    """
    hyperparam <- "power-epsilon"
    """
    power=int(hyperparameter.split("-")[0])
    # print("power ", power)
    epsilon=float(hyperparameter.split("-")[1])
    # print("epsilon " , epsilon)
    return 1 / (epsilon + distance**power)


#############################################
### SPARSIFIERS


def spectralSparsifier(data):
    """
    Following Algorithm 1 by Kent Quanrud as described in
    "Spectral Sparsification of Metrics and Kernels"

    Epsilon Sparsifier
    data: (n,d) dimension sparse matrix
    """
    # Epsilon Sparsifier
    eps = 1 
    # print("Constructing spectral sparsifier...")
    d = 5
    #n= num of pixels
    n = data.shape[0]

    # Random vector where each coordinate is independently 
    # distributed as a standard Gaussian
    # Using newer numpy random generator package 
    vector = np.random.default_rng().standard_normal(d)
    # Compute embedding of data on vector
    y = data.dot(vector)
    # Rank each embedding in vector a
    ordered = np.lexsort((np.arange(n), y))
    ranks = np.empty(n,dtype=int)
    ranks[ordered] = np.arange(1, n+1)
    print(ranks)


    # Repeat (n logn eps^-2) times
    for i in range (n * math.log(n) * eps):
        # Sample edge (i,j) with prob proportional to inv rank difference
        # to do so, sample an interval length with set prob
        # then sample any interval with that length
        pass

    
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]))
    return Graph(adj_mat)


