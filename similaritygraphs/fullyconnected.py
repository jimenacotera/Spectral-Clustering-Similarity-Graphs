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



class FullyConnected: 
    def __init__(self, similarity_function, dropping_edges = True): 
        # Name of the selected similarity function
        self.sim_function = similarity_function
        # Edges are dropped when weight below a threshold to avoid near zero edges.
        self.dropping_edges = dropping_edges

        # if similarity_function == "rbf": 
        #     return anySparseGraph(rbf_kernel)
        




    def sparseGraph(self, adj_mat):
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




    def rbf_graph(self, data, variance=1, threshold=0.1):
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
        # Get the maximum distance which corresponds to the threshold specified.
        if threshold <= 0:
            # Handle the case when threshold is equal to 0 - need to create a fully connected graph.
            max_distance = float('inf')
        else:
            max_distance = math.sqrt(-2 * variance * math.log(threshold))

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
                    weight = math.exp(- (distance**2) / (2 * variance))
                    adj_mat[vertex, neighbour] = weight
                    adj_mat[neighbour, vertex] = weight


        self.sparseGraph(adj_mat)
        return self

    def anySparseGraph(self, data, kernel, variance=1, threshold=0.1):
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
        # Get the maximum distance which corresponds to the threshold specified.
        if threshold <= 0:
            # Handle the case when threshold is equal to 0 - need to create a fully connected graph.
            max_distance = float('inf')
        else:
            max_distance = math.sqrt(-2 * variance * math.log(threshold))

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
                    weight = kernel(distance, variance)
                    adj_mat[vertex, neighbour] = weight
                    adj_mat[neighbour, vertex] = weight

        self.sparseGraph(adj_mat)
        return self



    #############################################
    ### KERNELS

    def rbf_kernel(distance, variance):
        return math.exp(- (distance**2) / (2 * variance))
    
    def laplacian_kernel(distance,variance):
        return math.exp(- (distance) / (math.sqrt(variance)))