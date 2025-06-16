import math
from typing import List

import scipy
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import pandas as pd



class KNN: 
    def __init__(self, k = 10):
        self.k = k