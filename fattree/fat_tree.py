import heapq
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from .core_switch import CoreSwitch
from .agg_switch import AggregateSwitch
from .edge_switch import EdgeSwitch
from .phys_machine import PhysicalMachine
from .vm_pair import VmPair
from .sized_vm_pair import SizedVmPair
from .ac_migrate_pytorch import Actor, Critic
try:
    import pulp as pl
except ImportError:
    pl = None


class FatTree:
    def __init__(self, k, vm_pair_count, vnf_capacity, vnf_count, pm_capacity):
        self.uuid = 0
        self.k = k
        self.vm_pair_count = vm_pair_count
        self.vnf_capacity = vnf_capacity
        self.vnf_count = vnf_count
        self.pm_capacity = pm_capacity
        self.migration_coefficient= 50

        self.discount_factor = 0.5
        self.episodes = 20
        self.temperature = 10000
        self.epsilon = 0.01
        self.q_table = {}
        self.episode_costs = []

        self.first_pm = (k * k) // 4 + (k * k // 2) + (k * k // 2)
        self.last_pm = self.first_pm + (k * k * k) // 4 - 1
        self.pm_count = self.last_pm - self.first_pm + 1
        
        tree_size = (k * k // 4) + (k // 2 * k) + (k // 2 * k) + (k * k * k // 4)
        self.tree = np.empty(tree_size, dtype=object)  # Optimized using NumPy arrays
        self.vnfs = np.empty(vnf_count, dtype=int)
        self.vm_pairs = np.empty(vm_pair_count, dtype=object)
        
        self.traffic_low = 0
        self.traffic_high = 0
        
        self.build_tree()  # Initializing the tree
        self.place_vnfs()  # Placing VNFs
    
    def place_vnfs(self):
        # random placement using NumPy
        for i in range(self.vnf_count):
            flag = True
            while flag:
                random_node = np.random.randint(0, self.first_pm)
                if random_node not in self.vnfs:  # NumPy's fast lookup
                    self.vnfs[i] = random_node
                    flag = False

    def get_sorted_pairs(self):
        return sorted(self.vm_pairs, key=lambda vm_pair: vm_pair.traffic_rate, reverse=True)

    def build_tree(self):
        # Add core switches
        for i in range(self.k * self.k // 4):
            self.add_to_tree("core", -1, -1)
        
        # Add aggregate switches
        pod_count = -1
        core_count = 0
        for i in range(self.k // 2 * self.k):
            if i % (self.k // 2) == 0:
                pod_count += 1
                core_count = 0
            self.add_to_tree("aggregate", pod_count, -1)
            
            # Create edges between aggregate and core switches
            for j in range(self.k // 2):
                temp = self.tree[self.uuid - 1]
                temp.core_edges[j] = core_count

                core = self.tree[core_count]
                core.add_aggr_edge(self.uuid - 1)
                core_count += 1
        
        # Add edge switches
        pod_count = -1
        edge_id = self.uuid - 1
        for i in range(self.k // 2 * self.k):
            if i % (self.k // 2) == 0:
                pod_count += 1
            self.add_to_tree("edge", pod_count, -1)
        
        # Add physical machines
        pod_count = -1
        for i in range(self.k * self.k * self.k // 4):
            if i % (self.k * self.k // 4) == 0:
                pod_count += 1
            if i % (self.k // 2) == 0:
                edge_id += 1
            self.add_to_tree("pm", pod_count, edge_id)

    def add_to_tree(self, node_type, pod, edge_id):
        # Using NumPy for efficient array management
        if node_type == "core":
            self.tree[self.uuid] = CoreSwitch(self.uuid, self.k)
        elif node_type == "aggregate":
            self.tree[self.uuid] = AggregateSwitch(self.uuid, pod, self.k)
        elif node_type == "edge":
            self.tree[self.uuid] = EdgeSwitch(self.uuid, pod)
        elif node_type == "pm":
            self.tree[self.uuid] = PhysicalMachine(self.uuid, pod, edge_id, self.pm_capacity)
        self.uuid += 1

    def get_pair_cost(self, pm1, pm2):
        cost=0
        cost+=self.distance(pm1, self.vnfs[0], True)
        cost+=self.distance(self.vnfs[self.vnf_count-1], pm2, True)
        return cost
    
    
    def distance(self,one, two, flag):
        if flag==True:
            one = self.tree[one]
            two = self.tree[two]
            one_id, two_id = one.id, two.id

            # If the two nodes are the same
            if one_id == two_id:
                return 0

            # Core to Core
            if isinstance(one, CoreSwitch) and isinstance(two, CoreSwitch):
                for aggr_id in one.aggr_edges:
                    if aggr_id in two.aggr_edges:
                        return 2  # Both core switches are connected to the same aggregate switch
                return 4  # Different aggregate switches

            # Aggregate to Aggregate
            if isinstance(one, AggregateSwitch) and isinstance(two, AggregateSwitch):
                if one.pod == two.pod:
                    return 2  # Both aggregate switches are in the same pod
                for core_id in one.core_edges:
                    if core_id in two.core_edges:
                        return 2  # Connected to the same core switch
                return 4  # Different core switches

            # Edge to Edge
            if isinstance(one, EdgeSwitch) and isinstance(two, EdgeSwitch):
                if one.pod == two.pod:
                    return 2  # Both edge switches are in the same pod
                return 4  # Different pods

            # Physical Machine to Physical Machine
            if isinstance(one, PhysicalMachine) and isinstance(two, PhysicalMachine):
                if one.edge_id == two.edge_id:
                    return 2  # Both physical machines are under the same edge switch
                if one.pod == two.pod:
                    return 4  # Both physical machines are in the same pod
                return 6  # Different pods

            # Core to Aggregate or Aggregate to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, AggregateSwitch)) or (isinstance(one, AggregateSwitch) and isinstance(two, CoreSwitch)):
                if isinstance(one, CoreSwitch):
                    return 1 if two.id in one.aggr_edges else 3
                return 1 if one.id in two.aggr_edges else 3

            # Core to Edge or Edge to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, CoreSwitch)):
                return 2  # Distance between any core switch and any edge switch is always 2

            # Core to Physical Machine or Physical Machine to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, CoreSwitch)):
                return 3  # Distance between any core switch and any physical machine is always 3

            # Aggregate to Edge or Edge to Aggregate
            if (isinstance(one, AggregateSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, AggregateSwitch)):
                if isinstance(one, AggregateSwitch):
                    return 1 if one.pod == two.pod else 3
                return 1 if two.pod == one.pod else 3

            # Aggregate to Physical Machine or Physical Machine to Aggregate
            if (isinstance(one, AggregateSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, AggregateSwitch)):
                if isinstance(one, AggregateSwitch):
                    return 2 if one.pod == two.pod else 4
                return 2 if two.pod == one.pod else 4

            # Edge to Physical Machine or Physical Machine to Edge
            if (isinstance(one, EdgeSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, EdgeSwitch)):
                if isinstance(one, EdgeSwitch):
                    return 1 if one.id == two.edge_id else (3 if one.pod == two.pod else 5)
                return 1 if two.id == one.edge_id else (3 if two.pod == one.pod else 5)

            # Default case (should never reach this)
            return -1
        else:
            one_id, two_id = one.id, two.id

            # If the two nodes are the same
            if one_id == two_id:
                return 0

            # Core to Core
            if isinstance(one, CoreSwitch) and isinstance(two, CoreSwitch):
                for aggr_id in one.aggr_edges:
                    if aggr_id in two.aggr_edges:
                        return 2  # Both core switches are connected to the same aggregate switch
                return 4  # Different aggregate switches

            # Aggregate to Aggregate
            if isinstance(one, AggregateSwitch) and isinstance(two, AggregateSwitch):
                if one.pod == two.pod:
                    return 2  # Both aggregate switches are in the same pod
                for core_id in one.core_edges:
                    if core_id in two.core_edges:
                        return 2  # Connected to the same core switch
                return 4  # Different core switches

            # Edge to Edge
            if isinstance(one, EdgeSwitch) and isinstance(two, EdgeSwitch):
                if one.pod == two.pod:
                    return 2  # Both edge switches are in the same pod
                return 4  # Different pods

            # Physical Machine to Physical Machine
            if isinstance(one, PhysicalMachine) and isinstance(two, PhysicalMachine):
                if one.edge_id == two.edge_id:
                    return 2  # Both physical machines are under the same edge switch
                if one.pod == two.pod:
                    return 4  # Both physical machines are in the same pod
                return 6  # Different pods

            # Core to Aggregate or Aggregate to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, AggregateSwitch)) or (isinstance(one, AggregateSwitch) and isinstance(two, CoreSwitch)):
                if isinstance(one, CoreSwitch):
                    return 1 if two.id in one.aggr_edges else 3
                return 1 if one.id in two.core_edges else 3

            # Core to Edge or Edge to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, CoreSwitch)):
                return 2  # Distance between any core switch and any edge switch is always 2

            # Core to Physical Machine or Physical Machine to Core
            if (isinstance(one, CoreSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, CoreSwitch)):
                return 3  # Distance between any core switch and any physical machine is always 3

            # Aggregate to Edge or Edge to Aggregate
            if (isinstance(one, AggregateSwitch) and isinstance(two, EdgeSwitch)) or (isinstance(one, EdgeSwitch) and isinstance(two, AggregateSwitch)):
                if isinstance(one, AggregateSwitch):
                    return 1 if one.pod == two.pod else 3
                return 1 if two.pod == one.pod else 3

            # Aggregate to Physical Machine or Physical Machine to Aggregate
            if (isinstance(one, AggregateSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, AggregateSwitch)):
                if isinstance(one, AggregateSwitch):
                    return 2 if one.pod == two.pod else 4
                return 2 if two.pod == one.pod else 4

            # Edge to Physical Machine or Physical Machine to Edge
            if (isinstance(one, EdgeSwitch) and isinstance(two, PhysicalMachine)) or (isinstance(one, PhysicalMachine) and isinstance(two, EdgeSwitch)):
                if isinstance(one, EdgeSwitch):
                    return 1 if one.id == two.edge_id else (3 if one.pod == two.pod else 5)
                return 1 if two.id == one.edge_id else (3 if two.pod == one.pod else 5)

            # Default case (should never reach this)
            return -1

    def set_traffic_range(self, traffic_low, traffic_high):
        self.traffic_low = traffic_low
        self.traffic_high = traffic_high

    def create_vm_pairs(self):
        # Using random placement for VMs on physical machines
        for i in range(self.vm_pair_count):
            flag = True
            while flag:
                first = random.randint(self.first_pm, self.last_pm)
                second = random.randint(self.first_pm, self.last_pm)
                if first == second:
                    continue
                first_pm = self.tree[first]
                second_pm = self.tree[second]
                if first_pm.capacity_left <= 0 or second_pm.capacity_left <= 0:
                    continue
                flag = False

            first_pm.add_vm()
            second_pm.add_vm()
            rand_rate = random.randint(self.traffic_low, self.traffic_high)
            self.vm_pairs[i] = VmPair(first, second, rand_rate)

    def save_curr_locations(self):
        # Save current VM pair locations to old_locations
        self.old_locations = []
        for i in range(self.vm_pair_count):
            self.old_locations.append(self.vm_pairs[i].first_vm_location)
            self.old_locations.append(self.vm_pairs[i].second_vm_location)

    def restore_old_locations(self):
        # Restore VM pair locations from old_locations
        if hasattr(self, 'old_locations'):
            for i in range(self.vm_pair_count):
                self.vm_pairs[i].first_vm_location = self.old_locations[i * 2]
                self.vm_pairs[i].second_vm_location = self.old_locations[i * 2 + 1]
        

    def create_sized_vm_pairs(self, lower_bound, upper_bound):
        # Using random placement for VMs on physical machines
        for i in range(self.vm_pair_count):
            flag = True
            while flag:
                first = random.randint(self.first_pm, self.last_pm)
                second = random.randint(self.first_pm, self.last_pm)
                if first == second:
                    continue
                first_pm = self.tree[first]
                second_pm = self.tree[second]
                if first_pm.capacity_left <= 0 or second_pm.capacity_left <= 0:
                    continue
                flag = False

            first_pm.add_vm()
            second_pm.add_vm()
            rand_rate = random.randint(self.traffic_low, self.traffic_high)
            #create a random vm size between lower_bound and upper_bound
            vm_size = random.randint(lower_bound, upper_bound)
            self.vm_pairs[i] = SizedVmPair(first, second, rand_rate, vm_size)
        
    def create_sized_vm_pairs_fb(self, lower_bound, upper_bound):
        # Using random placement for VMs on physical machines
        for i in range(self.vm_pair_count):
            flag = True
            while flag:
                first = random.randint(self.first_pm, self.last_pm)
                second = random.randint(self.first_pm, self.last_pm)
                if first == second:
                    continue
                first_pm = self.tree[first]
                second_pm = self.tree[second]
                if first_pm.capacity_left <= 0 or second_pm.capacity_left <= 0:
                    continue
                flag = False

            first_pm.add_vm()
            second_pm.add_vm()
             # Assign traffic rate based on weighted probability
            prob = random.random()
            if prob < 0.25:
                rand_rate = random.randint(0, 299)  # light
            elif prob < 0.95:
                rand_rate = random.randint(300, 700)  # medium
            else:
                rand_rate = random.randint(701, 1000)  # heavy
            #rand_rate = random.randint(self.traffic_low, self.traffic_high)
            #create a random vm size between lower_bound and upper_bound
            vm_size = random.randint(lower_bound, upper_bound)
            self.vm_pairs[i] = SizedVmPair(first, second, rand_rate, vm_size)


    def create_sized_pairs_ff_place(self, lower_bound, upper_bound):
        #placing VM pairs based on PAL algorithm
        #first call functions that create the VM pairs
        #they are also placed randomly but, they are correctly placed in this function
        
        self.create_sized_vm_pairs_fb(lower_bound, upper_bound)
        self.save_curr_locations()
        pm_slots = []
        for i in range(self.pm_count):
            tempICost = self.distance(self.vnfs[0], self.first_pm + i, True)
            tempECost = self.distance(self.vnfs[self.vnf_count - 1], self.first_pm + i, True)
            slot = {
                "pm_id": self.first_pm + i,
                "i_cost": tempICost,
                "e_cost": tempECost,
                "powered_on": False,
                "open_slots": self.pm_capacity,
            }
            pm_slots.append(slot)

        sorted_by_icost = sorted(pm_slots, key=lambda slot: slot["i_cost"])
        sorted_by_ecost = sorted(pm_slots, key=lambda slot: slot["e_cost"])

        powered_on_pms_i = []
        powered_on_pms_e = []

        #for slot in pm_slots:
            #if slot["powered_on"]:
                #heapq.heappush(self.powered_on_iheap, (slot["i_cost"], slot))
                #heapq.heappush(self.powered_on_eheap, (slot["e_cost"], slot))
        used_pms = set()

        sorted_pairs = sorted(self.vm_pairs, key=lambda vm_pair: vm_pair.traffic_rate, reverse=True)
        for i in range(self.vm_pair_count):
            found_i_pm = False
            for idx, (cost, pm_id, slot) in enumerate(powered_on_pms_i):
                if slot["open_slots"] >= sorted_pairs[i].vm_size:
                    sorted_pairs[i].first_vm_location = slot["pm_id"]
                    slot["open_slots"] -= sorted_pairs[i].vm_size
                    used_pms.add(slot["pm_id"])
                    found_i_pm = True
                    if slot["open_slots"] == 0:
                        powered_on_pms_i.pop(idx)
                        heapq.heapify(powered_on_pms_i)
                    break
            if not found_i_pm:
                slot = sorted_by_icost.pop(0)
                slot["powered_on"] = True
                slot["open_slots"] -= sorted_pairs[i].vm_size
                sorted_pairs[i].first_vm_location = slot["pm_id"]
                used_pms.add(slot["pm_id"])
                if slot["open_slots"] > 0:
                    heapq.heappush(powered_on_pms_i, (slot["i_cost"], slot["pm_id"], slot))


            found_e_pm = False
            for idx, (cost, pm_id, slot) in enumerate(powered_on_pms_e):
                if slot["open_slots"] >= sorted_pairs[i].vm_size:
                    sorted_pairs[i].second_vm_location = slot["pm_id"]
                    slot["open_slots"] -= sorted_pairs[i].vm_size
                    used_pms.add(slot["pm_id"])
                    found_e_pm = True
                    if slot["open_slots"] == 0:
                        powered_on_pms_e.pop(idx)
                        heapq.heapify(powered_on_pms_e)
                    break
            if not found_e_pm:
                slot = sorted_by_ecost.pop(0)
                slot["powered_on"] = True
                slot["open_slots"] -= sorted_pairs[i].vm_size
                sorted_pairs[i].second_vm_location = slot["pm_id"]
                used_pms.add(slot["pm_id"])
                if slot["open_slots"] > 0:
                    heapq.heappush(powered_on_pms_e, (slot["e_cost"], slot["pm_id"], slot))

                    
        #print out total cost of configuration
        total_cost = 0
        for i in range(self.vm_pair_count):
            total_cost += self.calc_pair_cost(sorted_pairs[i])
        print(f"Total cost of configuration for sized first fit placement: {total_cost}")
        self.restore_old_locations()
        used_pm_count = len(used_pms)
        return total_cost, used_pm_count

    def create_pairs_sized_pal_place(self, lower_bound, upper_bound):
        #self.create_sized_vm_pairs(lower_bound, upper_bound)
        pm_slots = []
        for i in range(self.pm_count):
            tempICost = self.distance(self.vnfs[0], self.first_pm + i, True)
            tempECost = self.distance(self.vnfs[self.vnf_count - 1], self.first_pm + i, True)
            slot = {
                "pm_id": self.first_pm + i,
                "i_cost": tempICost,
                "e_cost": tempECost,
                "powered_on": False,
                "open_slots": self.pm_capacity,
            }
            pm_slots.append(slot)

        sorted_by_icost = sorted(pm_slots, key=lambda slot: slot["i_cost"])
        sorted_by_ecost = sorted(pm_slots, key=lambda slot: slot["e_cost"])
        
        sorted_pairs = sorted(self.vm_pairs, key=lambda vm_pair: vm_pair.traffic_rate, reverse=True)
        #pair vm pair placed on first PM it fits in
        # Initialize next-fit pointers
        current_i_pm_idx = 0
        current_e_pm_idx = 0

        used_pms = set()
        # Place vm pairs using next-fit
        for i in range(self.vm_pair_count):
            # Place first VM
            found_i_pm = False
            while current_i_pm_idx < len(sorted_by_icost):
                if sorted_by_icost[current_i_pm_idx]["open_slots"] >= sorted_pairs[i].vm_size:
                    sorted_pairs[i].first_vm_location = sorted_by_icost[current_i_pm_idx]["pm_id"]
                    sorted_by_icost[current_i_pm_idx]["open_slots"] -= sorted_pairs[i].vm_size
                    used_pms.add(sorted_by_icost[current_i_pm_idx]["pm_id"])
                    found_i_pm = True
                    # If current PM is full, move to next PM
                    if sorted_by_icost[current_i_pm_idx]["open_slots"] == 0:
                        current_i_pm_idx += 1
                    break
                else:
                    # Current PM doesn't have enough space, move to next
                    current_i_pm_idx += 1
            
            if not found_i_pm:
                raise ValueError(f"No PM found for VM pair {i} with size {sorted_pairs[i].vm_size}")
            
            # Place second VM
            found_e_pm = False
            while current_e_pm_idx < len(sorted_by_ecost):
                if sorted_by_ecost[current_e_pm_idx]["open_slots"] >= sorted_pairs[i].vm_size:
                    sorted_pairs[i].second_vm_location = sorted_by_ecost[current_e_pm_idx]["pm_id"]
                    sorted_by_ecost[current_e_pm_idx]["open_slots"] -= sorted_pairs[i].vm_size
                    used_pms.add(sorted_by_ecost[current_e_pm_idx]["pm_id"])
                    found_e_pm = True
                    # If current PM is full, move to next PM
                    if sorted_by_ecost[current_e_pm_idx]["open_slots"] == 0:
                        current_e_pm_idx += 1
                    break
                else:
                    # Current PM doesn't have enough space, move to next
                    current_e_pm_idx += 1
            
            if not found_e_pm:
                raise ValueError(f"No PM found for VM pair {i} with size {sorted_pairs[i].vm_size}")

        #print out total cost of configuration
        total_cost = 0
        for i in range(self.vm_pair_count):
            total_cost += self.calc_pair_cost(sorted_pairs[i])
        print(f"Total cost of configuration for sized PAL placement: {total_cost}")

        used_pm_count = len(used_pms)

        return total_cost, used_pm_count
   
    '''
   def create_pairs_sized_pal_place(self, lower_bound, upper_bound):
    # Build PM slots; I and E will reference the same dict objects
    pm_slots = []
    for j in range(self.pm_count):
        pm_id = self.first_pm + j
        pm_slots.append({
            "pm_id": pm_id,
            "i_cost": self.distance(self.vnfs[0], pm_id, True),
            "e_cost": self.distance(self.vnfs[self.vnf_count - 1], pm_id, True),
            "open_slots": self.pm_capacity,
        })

    # Ordered PM lists (Next-Fit over these)
    I = sorted(pm_slots, key=lambda s: s["i_cost"])
    E = sorted(pm_slots, key=lambda s: s["e_cost"])

    # VM pairs by non-ascending traffic
    pairs = sorted(self.vm_pairs, key=lambda p: p.traffic_rate, reverse=True)

    i = j = 0
    used_pms = set()

    for k in range(self.vm_pair_count):
        dv = getattr(pairs[k], "vm_size", 1)
        dvp = dv  # same size for both VMs in a pair

        # find feasible I[i] and E[j]
        while i < len(I) and I[i]["open_slots"] < dv:
            i += 1
        while j < len(E) and E[j]["open_slots"] < dvp:
            j += 1
        if i >= len(I) or j >= len(E):
            raise ValueError(f"No feasible PMs for pair {k} (size={dv}).")

        # enforce different PMs — advance the side with the cheaper next option
        # until we get I[i].pm_id != E[j].pm_id
        guard = 0
        while i < len(I) and j < len(E) and I[i]["pm_id"] == E[j]["pm_id"]:
            # try to move the side whose next candidate is "better available"
            move_I = False
            # if E can't move or I can and looks promising, move I; else move E
            if (j + 1 >= len(E)) or (i + 1 < len(I) and I[i + 1]["open_slots"] >= dv and I[i + 1]["i_cost"] <= E[j]["e_cost"]):
                move_I = True
            if move_I:
                i += 1
                while i < len(I) and I[i]["open_slots"] < dv:
                    i += 1
            else:
                j += 1
                while j < len(E) and E[j]["open_slots"] < dvp:
                    j += 1
            guard += 1
            if guard > self.pm_count * 2:
                raise ValueError("Could not find two distinct PMs with enough capacity.")

        if i >= len(I) or j >= len(E):
            raise ValueError(f"No feasible distinct PMs for pair {k} (size={dv}).")

        # place on different PMs
        pairs[k].first_vm_location  = I[i]["pm_id"]
        pairs[k].second_vm_location = E[j]["pm_id"]
        I[i]["open_slots"] -= dv
        E[j]["open_slots"] -= dvp
        used_pms.add(I[i]["pm_id"]); used_pms.add(E[j]["pm_id"])

        # Next-Fit: advance pointer if this PM can no longer fit the next item
        if i < len(I) and I[i]["open_slots"] < dv:
            i += 1
        if j < len(E) and E[j]["open_slots"] < dvp:
            j += 1

    # cost
    total_cost = sum(self.calc_pair_cost(p) for p in pairs)
    print(f"Total cost of configuration for sized PAL placement: {total_cost}")
    return total_cost, len(used_pms)
        '''
    def make_t(self):
        #t will be a 2d array
        #first dimension size is number of VMs (vm pairs *2)
        #second dimension size is number of PMS
        n_v = self.vm_pair_count * 2
        n_p = self.pm_count
        t = [[0 for _ in range(n_p)] for _ in range (n_v)]

        ingress_sw = self.vnfs[0]
        egress_sw = self.vnfs[self.vnf_count - 1]

        for i in range(self.vm_pair_count):
            pair = self.vm_pairs[i]
            lam = pair.traffic_rate

            for j in range(n_p):
                pm_id = self.first_pm + j

                # ingress VM (index 2*i)
                mig_i  = self.migration_coefficient * self.distance(pair.first_vm_location, pm_id, True)
                comm_i = lam * self.distance(pm_id, ingress_sw, True)
                t[2 * i][j] = mig_i + comm_i

                # egress VM (index 2*i+1)
                mig_e  = self.migration_coefficient * self.distance(pair.second_vm_location, pm_id, True)
                comm_e = lam * self.distance(pm_id, egress_sw, True)
                t[2 * i + 1][j] = mig_e + comm_e

        return t

    def make_d(self):
        d = []
        for i in range(self.vm_pair_count):
            pair = self.vm_pairs[i]
            size = getattr(pair, "vm_size", 1)
            d.extend([int(size), int(size)])
        return d
    
    def make_rc(self):
        return [int(self.pm_capacity)] * self.pm_count

    def migrate_pamh_ilp(
        self,
        log: bool = True,
        time_limit: float | None = None,
        gap: float | None = None,       # relative MIP gap (e.g., 0.01 for 1%)
        threads: int | None = None,
        write_lp: bool = False,
        keep_files: bool = False,
    ):
        # 1) Build data
        t  = self.make_t()
        d  = self.make_d()
        rc = self.make_rc()
        n_v = self.vm_pair_count * 2
        n_p = self.pm_count

        # 2) Model
        prob = pl.LpProblem("PAMH_ILP", pl.LpMinimize)

        # (8) Vars
        x = {(v, j): pl.LpVariable(f"x_{v}_{j}", 0, 1, pl.LpBinary)
            for v in range(n_v) for j in range(n_p)}

        # (9) Each VM exactly once
        for v in range(n_v):
            prob += pl.lpSum(x[v, j] for j in range(n_p)) == 1, f"assign_once_v{v}"

        # (10) PM capacity
        for j in range(n_p):
            prob += pl.lpSum(d[v] * x[v, j] for v in range(n_v)) <= rc[j], f"cap_pm{j}"

        # (7) Objective
        prob += pl.lpSum(t[v][j] * x[v, j] for v in range(n_v) for j in range(n_p))

        # Optional: write .lp for inspection
        if write_lp:
            prob.writeLP("PAMH_ILP.lp")

        # --- Solver (CBC) with progress logging and portable gap control ---
        cbc_opts = []
        if log:
            # more verbose log (2)
            cbc_opts += ["-log", "2"]
        if gap is not None:
            # CBC's relative MIP gap (e.g., 0.01 for 1%)
            cbc_opts += ["-ratio", str(gap)]

        solver = pl.PULP_CBC_CMD(
            msg=log,                 # print CBC progress
            timeLimit=time_limit,    # seconds (None = no limit)
            threads=threads,         # None = CBC default
            keepFiles=keep_files,    # keep temp files for inspection
            mip=True,
            options=cbc_opts,        # <-- pass -ratio / -log here
        )

        status_code = prob.solve(solver)
        status = pl.LpStatus[status_code]

        # 4) Extract solution
        assignment = [-1] * n_v
        for v in range(n_v):
            for j in range(n_p):
                if pl.value(x[v, j]) > 0.5:
                    assignment[v] = j
                    break

        used_pms = sorted({a for a in assignment if a != -1})
        obj_value = pl.value(prob.objective)

        return assignment, obj_value, used_pms, status
    
    def migrate_pamh_plan(self, apply=True):
        """
        Greedy PAM-H repack into empty PMs.
        Returns:
            total_cost (float): sum_v t[v][assigned_j]
            used_pm_count (int): number of PMs with ≥1 VM
        """
        # Costs / sizes / capacities
        t  = self.make_t()        # [n_v][n_p]
        d  = self.make_d()        # [n_v]
        rc = self.make_rc()       # [n_p]
        n_v = self.vm_pair_count * 2
        n_p = self.pm_count

        # PMs start empty (local capacities)
        cap = rc[:]

        # Old communication cost per VM (pre-migration)
        ingress_sw = self.vnfs[0]
        egress_sw  = self.vnfs[self.vnf_count - 1]
        old_comm = [0.0] * n_v
        for i in range(self.vm_pair_count):
            pair = self.vm_pairs[i]
            lam  = pair.traffic_rate
            old_comm[2*i]   = lam * self.distance(pair.first_vm_location,  ingress_sw, True)       # ingress VM
            old_comm[2*i+1] = lam * self.distance(egress_sw,               pair.second_vm_location, True)  # egress VM

        assignment_idx = [-1] * n_v
        remaining = set(range(n_v))
        used_pm_idxs = set()

        def best_feasible(v):
            best_j, best_cost = None, None
            need = d[v]
            for j in range(n_p):
                if need <= cap[j]:
                    c = t[v][j]
                    if best_cost is None or c < best_cost:
                        best_j, best_cost = j, c
            if best_j is None:
                raise ValueError(f"No feasible PM for VM {v} with demand {need} under current capacities.")
            return best_j, best_cost

        # Greedy selection rounds
        while remaining:
            pick_v = pick_j = None
            pick_score = None
            for v in remaining:
                j_star, c_star = best_feasible(v)
                util = old_comm[v] - c_star                 # improvement
                denom = d[v] if d[v] > 0 else 1
                score = util / denom                        # utility per unit VM size
                if (pick_score is None) or (score > pick_score):
                    pick_v, pick_j, pick_score = v, j_star, score

            assignment_idx[pick_v] = pick_j
            cap[pick_j] -= d[pick_v]
            used_pm_idxs.add(pick_j)
            remaining.remove(pick_v)

        total_cost = sum(t[v][assignment_idx[v]] for v in range(n_v))
        assignment_pm_ids = [self.first_pm + j for j in assignment_idx]
        used_pm_count = len(used_pm_idxs)

        if apply:
            # write the chosen plan back into vm_pairs
            for i in range(self.vm_pair_count):
                self.vm_pairs[i].first_vm_location  = assignment_pm_ids[2*i]
                self.vm_pairs[i].second_vm_location = assignment_pm_ids[2*i+1]

        # Return exactly what your plotting code expects
        return total_cost, used_pm_count

    def create_pairs_pal_place(self):
        #placing VM pairs based on PAL algorithm
        #first call functions that create the VM pairs
        #they are also placed randomly but, they are correctly placed in this function
        self.create_vm_pairs()
        resource_slots =[]
        #there is a total of pm_count * pm_capacity slots available
        for i in range(self.pm_count):
            for j in range(self.pm_capacity):
                tempICost = self.distance(self.vnfs[0], self.first_pm + i, True)
                tempECost = self.distance(self.vnfs[self.vnf_count - 1], self.first_pm + i, True)

                slot = {
                    "pm_id": self.first_pm + i,
                    "slot_id": (self.pm_count * i) + j,
                    "i_cost": tempICost,
                    "e_cost": tempECost,
                    "selected": False
                }
                resource_slots.append(slot)
        
        iota = []
        epsilon = []
        i_opt=[]
        e_opt=[]

        sorted_by_icost = sorted(resource_slots, key=lambda slot: slot["i_cost"])
        sorted_by_ecost = sorted(resource_slots, key=lambda slot: slot["e_cost"])

        for i in range(self.vm_pair_count*2):
            iota.append(sorted_by_icost[i])
            epsilon.append(sorted_by_ecost[i])

        i=j=k=0
        while(k< self.vm_pair_count):
            if(iota[i]["selected"]):
                i+=1
                continue
            if(epsilon[j]["selected"]):
                j+=1
                continue
            if(iota[i]["pm_id"] != epsilon[j]["pm_id"]):
                i_opt.append(iota[i])
                e_opt.append(epsilon[j])
                iota[i]["selected"] = True
                epsilon[j]["selected"] = True
                i += 1
                j += 1
            else:
                if(iota[i]["i_cost"] + epsilon[j+1]["e_cost"] < epsilon[j]["e_cost"] + iota[i+1]["i_cost"]):
                    i_opt.append(iota[i])
                    e_opt.append(epsilon[j+1])
                    iota[i]["selected"] = True
                    epsilon[j+1]["selected"] = True
                    i += 1
                    j += 2
                else:
                    i_opt.append(iota[i+1])
                    e_opt.append(epsilon[j])
                    iota[i+1]["selected"] = True
                    epsilon[j]["selected"] = True
                    i += 2
                    j += 1
            k += 1
        sorted_pairs = sorted(self.vm_pairs, key=lambda vm_pair: vm_pair.traffic_rate, reverse=True)
        for i in range(self.vm_pair_count):
            sorted_pairs[i].first_vm_location = i_opt[i]["pm_id"]
            sorted_pairs[i].second_vm_location = e_opt[i]["pm_id"]
        
        #print out total cost of configuration
        total_cost = 0
        for i in range(self.vm_pair_count):
            total_cost += self.calc_pair_cost(sorted_pairs[i])
        print(f"Total cost of configuration for PAL placement: {total_cost}")




    def randomize_traffic(self):
        # Using NumPy for efficient random traffic generation
        for pair in self.vm_pairs:
            prob = random.random()
            if prob < 0.25:
                rand_rate = random.randint(0, 299)  # light
            elif prob < 0.95:
                rand_rate = random.randint(300, 700)  # medium
            else:
                rand_rate = random.randint(701, 1000)  # heavy
            pair.traffic_rate = rand_rate

    def reset_pms(self):
        """Reset every PM’s available capacity to full (homogeneous)."""
        for pm in range(self.first_pm, self.last_pm + 1):
            self.tree[pm].capacity_left = self.pm_capacity

    def cs2_migration(self):
        self.calculate_initial_cost()
        self.vmp_mcf_file()
        self.read_mcf_pairs_output()

    def init_ac(self):
        self.d = self.vm_pair_count * 2 * self.pm_count
        #self.policy = {vm: {pm: 1 / self.pm_count for pm in range(self.first_pm, self.last_pm + 1)} for vm in self.vm_pairs}
        self.policy = {i: {pm: 1 / self.pm_count for pm in range(self.first_pm, self.last_pm + 1)} for i in range(self.vm_pair_count * 2)}
        #print(f"Policy keys: {list(self.policy.keys())}")


        self.T = np.eye(self.d)
        self.q_table = {}
        self.B = np.eye(self.d)
        #self.phi = np.zeros(self.d)
        self.theta = np.zeros(self.d)
        self.z = np.zeros((self.d,1))
        self.C = 0
        self.time=1
        #set all pm capacities to empty
        for i in range(self.first_pm, self.last_pm+1):
            self.tree[i].capacity_left = self.pm_capacity

    def select_action(self, curr_vm):
        pm_choices = list(self.policy[curr_vm].keys())
        #print pm_choices
        #print(f"PM choices for VM {curr_vm}: {pm_choices}")
        pm_probs = list(self.policy[curr_vm].values())

        # Check PM capacities and set probabilities to 0 for PMs at capacity
        for i, pm in enumerate(pm_choices):
            if self.tree[pm].capacity_left <= 0:  
                pm_probs[i] = 0

        # Normalize the probabilities to sum to 1
        total_prob = sum(pm_probs)
        if total_prob > 0:
            pm_probs = [p / total_prob for p in pm_probs]
        else:
            raise ValueError("No valid PMs available for VM migration.")

        # Select an action based on the adjusted probabilities
        selected_action = np.random.choice(pm_choices, p=pm_probs)
        #decrement selected PM's capacity 
        self.tree[selected_action].capacity_left -= 1
        #print selected_action
        #print(f"Selected PM for VM {curr_vm}: {selected_action}")
        return selected_action

        
    def simulate_action(self,actions):
        total_cost = 0

        for i in actions.keys():
            
            #migration cost
            if(i%2==0):
                total_cost+= self.distance(actions[i], self.vnfs[0], True) * self.vm_pairs[i//2].traffic_rate
                #total_cost*= self.vm_pairs[i//2].traffic_rate
                total_cost+= self.distance(self.vm_pairs[i//2].first_vm_location, actions[i], True) * self.migration_coefficient
            else:
                total_cost+= self.distance(actions[i], self.vnfs[self.vnf_count-1], True) * self.vm_pairs[i//2].traffic_rate
                #total_cost*= self.vm_pairs[i//2].traffic_rate
                total_cost+= self.distance(self.vm_pairs[i//2].second_vm_location, actions[i], True) * self.migration_coefficient

        #save original location, set new location, get next action, get its phi, reset location to original, finally also return the nect phi
        original_locations = []
        next_actions = {}
        current_state = self.get_state()
        for i in range(self.vm_pair_count):
            original_locations.append(self.vm_pairs[i].first_vm_location)
            original_locations.append(self.vm_pairs[i].second_vm_location)
            
            self.vm_pairs[i].first_vm_location = actions[i*2]
            self.vm_pairs[i].second_vm_location = actions[i*2+1]

            

            #next_actions[self.vm_pairs_sorted_index[i]*2]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2])   
            #next_actions[self.vm_pairs_sorted_index[i]*2+1]= self.select_action(current_state[self.vm_pairs_sorted_index[i]*2+1]) 
            next_actions[i * 2] = self.select_action(i * 2)  # Ingress VM
            next_actions[i * 2 + 1] = self.select_action(i * 2 + 1)  # Egress VM 

        phi = self.get_phi(next_actions)
        for i in range(self.vm_pair_count):
            self.vm_pairs[i].first_vm_location = original_locations[i*2]
            self.vm_pairs[i].second_vm_location = original_locations[i*2+1]


        return total_cost, phi
    
    def get_phi(self, actions):
        phi = np.zeros((self.vm_pair_count * 2, self.pm_count))
        #print(actions)
        for i, action in enumerate(actions):
            #vm_pair_index = i // 2
            #print action
            #print("actionz")
            #print(actions[action])
            phi[i,actions[action]-self.first_pm] = 1
        return phi.flatten().reshape(-1, 1)

            

    def ac_migration(self):
        actor = Actor(input_dim=2, hidden_dim=64, output_dim=self.pm_count)
        critic = Critic(input_dim=2, hidden_dim=64)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)

        for episode in range(self.episodes):
            self.randomize_traffic()
            for pm in range(self.first_pm, self.last_pm+1):
                self.tree[pm].capacity_left = self.pm_capacity
            
            episode_cost = 0

            sorted_pairs = self.get_sorted_pairs()
            vm_indices_sorted = [np.where(self.vm_pairs == p)[0][0] for p in sorted_pairs]

            for vm_i in range(self.vm_pair_count*2):
                is_ingress = vm_i % 2 == 0
                pair_i = vm_i // 2
                curr_location = sorted_pairs[pair_i].first_vm_location if is_ingress else sorted_pairs[pair_i].second_vm_location
                state = torch.tensor([vm_i, int(is_ingress)], dtype=torch.float32)
                
                #This uses OLD locations, not locations for this episode
                #Get available PMs for actions
                available_pms = [pm for pm in range(self.first_pm, self.last_pm + 1)
                             if self.tree[pm].capacity_left > 0 and (
                                 pm != sorted_pairs[pair_i].second_vm_location if is_ingress else pm != sorted_pairs[pair_i].first_vm_location)]
                
                logits= actor(state)
                pm_logits = logits[[pm-self.first_pm for pm in available_pms]]
                pm_probs = torch.softmax(pm_logits, dim=0)
                actioni = torch.multinomial(pm_probs, 1).item()
                selected_pm = available_pms[actioni]

                #compute migration reward
                old_loc = curr_location
                new_loc = selected_pm
                if is_ingress:
                    reward = -self.distance(old_loc, new_loc, True) * self.migration_coefficient + self.distance(old_loc, self.vnfs[0], True) * sorted_pairs[pair_i].traffic_rate
                    sorted_pairs[pair_i].first_vm_location = selected_pm
                else:
                    reward = -self.distance(old_loc, new_loc, True) * self.migration_coefficient + self.distance(old_loc, self.vnfs[-1], True) * self.vm_pairs[pair_i].traffic_rate
                    sorted_pairs[pair_i].second_vm_location = selected_pm
                episode_cost += -reward

                #value estimates for critic
                value = critic(state)
                next_state = torch.tensor([vm_i+1, int((vm_i)%2==0)], dtype=torch.float32)
                next_value = critic(next_state) if vm_i+1 < self.vm_pair_count*2 else torch.tensor(0.0)
                td_target = reward + self.discount_factor * next_value
                td_error = td_target - value

                #update critic
                critic_loss = td_error.pow(2)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                #now update actor
                log_prob = torch.log(pm_probs[actioni])
                actor_loss = -log_prob * td_error.detach()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
            self.episode_costs.append(-episode_cost)
            print(f"Episode {episode} - Cost: {-episode_cost}")
            self.plot_episodes_cost()

    def plot_episodes_cost(self):
        #save the plot to the project root directory
        plt.plot(self.episode_costs)
        plt.xlabel('Episodes')
        plt.ylabel('Cost')
        plt.title('Cost over time')
        plt.savefig('cost_over_time.png')
        

    def  policy_calculator(self, phi, theta):
        pass

    def get_state(self):
        #State 
        # State includes VM pair locations
        #self.sorted = self.get_sorted_pairs()
        #self.vm_pairs_sorted_index=[] #index i of this list is the vm pair num of the ith element in sorted
        state = []
        for i in range(self.vm_pair_count):
            #state.append(sorted[i].first_vm_location)
            #state.append(sorted[i].second_vm_location)
            state.append(self.vm_pairs[i].first_vm_location)
            state.append(self.vm_pairs[i].second_vm_location)

            #self.vm_pairs_sorted_index.append(np.where(self.vm_pairs == self.sorted[i])[0][0])

        return state
    
    def get_valid_actions(self, curr_pair):
        actions = []
        #sorted_pairs = self.vm_pairs
        #for i, vmp in enumerate (sorted_pairs):
            #current_pm1, current_pm2 = vmp.first_vm_location, vmp.second_vm_location
            #if i is less than curr_pair, then it has already been assigned a pm
            #if (i<curr_pair):
            #    actions.append((i, current_pm1, current_pm2))
            #    continue
            #if i is curr_pair, then it is its turn to be assigned a pm
            #if (i==curr_pair):
        for pm1 in range(self.first_pm, self.last_pm+1):
            for pm2 in range(self.first_pm, self.last_pm+1):
                if (pm1!=pm2 and self.tree[pm1].capacity_left>0 and self.tree[pm2].capacity_left>0):
                    actions.append((curr_pair, pm1, pm2))
                            
            #elif(i>curr_pair):
            #    actions.append((i,-1,-1))#not assigned a pm, not its turn yet
    

    def get_reward(self, action):
        # Reward is the negative of the difference in communication cost 
        # between the vm pairs old location and new location
        # (old location comm. cost - new location comm. cost)
        # plus the migration cost
        curr_pair, pm1, pm2 = action
        old_comm_cost = self.calc_pair_cost(self.vm_pairs[curr_pair])
        #old_comm_cost = self.vm_pairs[curr_pair].get_communication_cost(self)
        new_comm_cost = self.get_pair_cost(pm1, pm2)
        migration_cost = self.get_pair_cost(self.vm_pairs[curr_pair].first_vm_location, pm1)
        migration_cost += self.get_pair_cost(self.vm_pairs[curr_pair].second_vm_location, pm2)
        migration_cost *= self.migration_coefficient

        return -(old_comm_cost - (new_comm_cost + migration_cost))
    
    def calc_pair_cost(self, vm_pair):
        first_vnf = self.vnfs[0]
        last_vnf = self.vnfs[self.vnf_count - 1]
        #ingress
        cost = self.distance(vm_pair.first_vm_location, first_vnf, True)

        # Intra-VNF chaining: VNF_i to VNF_{i+1}
        for i in range(self.vnf_count - 1):
            vnf_src = self.vnfs[i]
            vnf_dst = self.vnfs[i + 1]
            cost += self.distance(vnf_src, vnf_dst, True)

        #egress 
        cost+= self.distance(last_vnf, vm_pair.second_vm_location, True)
        cost *= vm_pair.traffic_rate
        return cost

    def calc_total_cost(self):
        total_cost = 0
        for i in range(self.vm_pair_count):
            #total_cost += self.vm_pairs[i].get_communication_cost(self)
            self.calc_pair_cost(self.vm_pairs[i])
        return total_cost

    def vmp_mcf_file(self):
        """
        Generates the MCF input file for VM migration and replication.
        """
        arccount = 2 * self.vm_pair_count * self.pm_count  # Edges between VMs and PMs
        arccount += self.vm_pair_count * 2  # Edges from supply node to VM pairs
        arccount += self.pm_count  # Edges between PMs and the demand node

        nodecount = (self.vm_pair_count * 2) + self.pm_count + 2

        firstline = f"p min {nodecount} {arccount}\n"
        secline = f"c min-cost flow problem with {nodecount} nodes and {arccount} arcs \n"
        thirdline = f"n 1 {self.vm_pair_count * 2}\n"
        fourthln = f"c supply of {self.vm_pair_count * 2} at node 1 \n"
        fifthln = f"n {nodecount} {-1 * self.vm_pair_count * 2}\n"
        sixthln = f"c demand of {-1 * self.vm_pair_count * 2} at node {nodecount}\n"
        sevln = "c arc list follows \n"
        eithln = "c arc has <tail> <head> <capacity l.b.> <capacity u.b> <cost> \n"

        firstlns = firstline + secline + thirdline + fourthln + fifthln + sixthln + sevln + eithln

        supplyarcs = []
        countnode = 2

        for i in range(self.vm_pair_count * 2):
            supplyarcs.append(f"a 1 {countnode} 0 1 0 \n")
            countnode += 1

        firstvm = countnode
        vmarcs = ["c arcs from VMs to PMs \n"]

        for i in range(self.vm_pair_count * 2):
            countnode = firstvm
            for j in range(self.pm_count):
                last_val = 0
                vmnum = i // 2

                if i % 2 == 0:  # Ingress VM
                    last_val += (self.migration_coefficient * self.distance(self.tree[self.vm_pairs[vmnum].first_vm_location], self.tree[self.first_pm + j], False))
                    last_val += (self.vm_pairs[vmnum].traffic_rate * self.distance(self.tree[self.first_pm + j], self.tree[self.vnfs[0]], False))
                else:  # Egress VM
                    last_val += (self.migration_coefficient * self.distance(self.tree[self.vm_pairs[vmnum].second_vm_location], self.tree[self.first_pm + j], False))
                    last_val += (self.vm_pairs[vmnum].traffic_rate * self.distance(self.tree[self.first_pm + j], self.tree[self.vnfs[-1]], False))

                vmarcs.append(f"a {i + 2} {countnode} 0 1 {last_val}\n")
                countnode += 1

        pmarcs = ["c arcs from PMs to destination \n"]
        for i in range(self.pm_count):
            pmarcs.append(f"a {i + firstvm} {countnode} 0 {self.pm_capacity} 0 \n")

        output = firstlns + ''.join(supplyarcs) + ''.join(vmarcs) + ''.join(pmarcs)

        try:
            with open("mcf_replication.inp", "w") as file:
                file.write(output)

            print("mcf_replication.inp has been written to in the project root file directory")
        except Exception as e:
            print(f"Failed to write MCF file: {e}")

        if self.pm_count * self.pm_capacity < self.vm_pair_count * 2:
            print("Replication of every VM not possible.")

        #next, exec cs2 with passing in the generated file, and saving the output to a file
        #we want to run cs2 < mcf_replication.inp > output.txt
        os.system("cs2 < mcf_replication.inp > output.txt")

    def calculate_initial_cost(self):
        """
        Calculate and print the total communication cost before migration.
        """
        initial_total_cost = 0

        for vm_pair in self.vm_pairs:
            # Cost of communication for ingress
            ingress_cost = (
                vm_pair.traffic_rate
                * self.distance(
                    self.tree[vm_pair.first_vm_location],
                    self.tree[self.vnfs[0]],
                    False,
                )
            )
            # Cost of communication for egress
            egress_cost = (
                vm_pair.traffic_rate
                * self.distance(
                    self.tree[self.vnfs[-1]],
                    self.tree[vm_pair.second_vm_location],
                    False,
                )
            )
            ordered_cost = 0
            for j in range(len(self.vnfs) - 1):
                ordered_cost += vm_pair.traffic_rate * self.distance(
                    self.tree[self.vnfs[j]], self.tree[self.vnfs[j + 1]], False
                )
            
            initial_total_cost += ingress_cost + egress_cost + ordered_cost

        print(f"The total communication cost before migration is: {initial_total_cost}")
        return initial_total_cost

    def read_mcf_pairs_output(self):
        output_file = "output.txt"

        if not os.path.exists(output_file):
            print(f"File not found: {output_file}")
            return

        placed = 0
        total_cost = 0
        migr_cost = 0

        try:
            with open(output_file, 'r') as file:
                for line in file:
                    line = line.strip()

                    # Ignore comment lines
                    if line.startswith('c') or line.startswith('s'):
                        continue

                    # Parse the line into tokens
                    tokens = line.split()
                    if len(tokens) < 4:
                        continue

                    first_num = int(tokens[1])
                    if first_num == 1:
                        continue  # Ignore supply arc lines

                    vmpair_num = (first_num - 2) // 2  # Calculate VM pair ID
                    second_num = int(tokens[2])
                    third_num = int(tokens[3])

                    # Skip sink node flow
                    if second_num == (2 + (self.vm_pair_count * 2) + self.pm_count):
                        continue

                    pm_num = second_num - ((self.vm_pair_count * 2) + 2)

                    if third_num > 0:
                        # Update placement for VM pairs
                        if first_num % 2 == 0:
                            self.vm_pairs[vmpair_num].mcfMigrVm1Pm = pm_num + self.first_pm
                        else:
                            self.vm_pairs[vmpair_num].mcfMigrVm2Pm = pm_num + self.first_pm
                        placed += 1

            # Calculate migration and total costs
            for i in range(len(self.vm_pairs)):
                migr_cost += self.migration_coefficient * self.distance(
                    self.tree[self.vm_pairs[i].first_vm_location], self.tree[self.vm_pairs[i].mcfMigrVm1Pm], False
                )
                migr_cost += self.migration_coefficient * self.distance(
                    self.tree[self.vm_pairs[i].second_vm_location], self.tree[self.vm_pairs[i].mcfMigrVm2Pm], False
                )

                total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                    self.tree[self.vm_pairs[i].mcfMigrVm1Pm], self.tree[self.vnfs[0]], False
                )
                total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                    self.tree[self.vm_pairs[i].mcfMigrVm2Pm], self.tree[self.vnfs[-1]], False
                )

                for j in range(len(self.vnfs) - 1):
                    total_cost += self.vm_pairs[i].traffic_rate * self.distance(
                        self.tree[self.vnfs[j]], self.tree[self.vnfs[j + 1]], False
                    )

            # Output results
            print(f"Number of VMs placed: {placed}")
            print(f"The MCF total migration cost is: {migr_cost}")
            print(f"The MCF total cost is: {migr_cost + total_cost}")



        except Exception as e:
            print(f"An error occurred: {e}")

