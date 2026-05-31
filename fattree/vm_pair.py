class VmPair:
    def __init__(self, first, second, traffic_rate):
        self.first_vm_location = first
        self.second_vm_location = second
        self.traffic_rate = traffic_rate
        self.mcfMigrVm1Pm = None
        self.mcfMigrVm2Pm = None
