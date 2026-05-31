from .node import Node

class PhysicalMachine(Node):
    def __init__(self, id, pod, edge_id, capacity_left):
        super().__init__(id)  # Call the parent constructor (Node class)
        self.pod = pod  # Store pod information
        self.edge_id = edge_id  # Store edge switch ID
        self.capacity_left = capacity_left  # Initialize available capacity
        # vm_pairs would be implemented if necessary in the future
    
    def add_vm(self):
        if self.capacity_left == 0:
            raise Exception("Physical Machine is at capacity!")
        self.capacity_left -= 1  # Reduce capacity when VM is added

    def remove_vm_pair(self):
        self.capacity_left += 1  # Increase capacity when VM is removed
