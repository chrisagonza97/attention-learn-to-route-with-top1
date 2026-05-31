from .node import Node
import numpy as np

class CoreSwitch(Node):
    def __init__(self, id, k):
        super().__init__(id)  # Call the parent constructor (Node class)
        self.k = k
        self.aggr_edges = np.zeros(k, dtype=int)  # Use NumPy array initialized with zeros

    def add_aggr_edge(self, id):
        # Add the aggregate edge to the first available slot (where the value is 0)
        for i in range(self.k):
            if self.aggr_edges[i] == 0:
                self.aggr_edges[i] = id
                break  # Ensure we only add the edge once
