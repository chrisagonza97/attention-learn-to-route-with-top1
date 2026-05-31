from .node import Node
import numpy as np

class AggregateSwitch(Node):
    def __init__(self, id, pod, k):
        super().__init__(id)  # Call the parent constructor (Node class)
        self.pod = pod
        self.core_edges = np.empty(k // 2, dtype=int)  # Using NumPy array for core edges

