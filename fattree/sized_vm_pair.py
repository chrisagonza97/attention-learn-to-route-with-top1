from .vm_pair import VmPair


class SizedVmPair(VmPair):
    def __init__(self, first, second, traffic_rate, vm_size):
        super().__init__(first, second, traffic_rate)
        self.vm_size = vm_size
