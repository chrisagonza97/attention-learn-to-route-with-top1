from .node import Node

class EdgeSwitch(Node):
    def __init__(self, id, pod):
        super().__init__(id)  # Call the parent constructor (Node class)
        self.pod = pod  # Store the pod information
