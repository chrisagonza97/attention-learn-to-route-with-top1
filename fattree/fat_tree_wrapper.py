"""Fat Tree N-Stroll Wrapper — generates cost matrices for n-stroll instances."""

import torch
import numpy as np

try:
    from .fat_tree import FatTree
    FAT_TREE_AVAILABLE = True
    print("[fat_tree_wrapper] Successfully imported FatTree class")
except ImportError as e:
    FAT_TREE_AVAILABLE = False
    print(f"[fat_tree_wrapper] Could not import FatTree ({e}), using embedded topology generator")


class FatTreeNStrollWrapper:
    """
    Wrapper around the FatTree class to generate n-stroll instances.

    Extracts topology information from FatTree and creates cost matrices
    suitable for the n-stroll problem.
    """
    
    def __init__(self, k: int = 4):
        """
        Initialize wrapper with FatTree.
        
        Args:
            k: Fat tree parameter (must be even)
        """
        assert k % 2 == 0, "k must be even"
        self.k = k
        
        if FAT_TREE_AVAILABLE:
            # Use FatTree class
            self.fat_tree = FatTree(
                k=k,
                vm_pair_count=1,
                vnf_capacity=1,
                vnf_count=1,
                pm_capacity=10
            )
            self._init_from_fat_tree()
        else:
            # Fallback to embedded topology
            self._init_embedded(k)
    
    def _init_from_fat_tree(self):
        k = self.k
        ft = self.fat_tree

        self.num_core = (k * k) // 4
        self.num_agg = (k * k) // 2
        self.num_edge = (k * k) // 2
        self.num_switches = self.num_core + self.num_agg + self.num_edge
        self.num_pm = (k * k * k) // 4
        
        # Index ranges (matching FatTree)
        self.first_pm = ft.first_pm
        self.last_pm = ft.last_pm
    
    def _init_embedded(self, k: int):
        """Fallback: embedded topology generator (no external dependency)."""
        self.num_core = (k * k) // 4
        self.num_agg = (k * k) // 2
        self.num_edge = (k * k) // 2
        self.num_switches = self.num_core + self.num_agg + self.num_edge
        self.num_pm = (k * k * k) // 4
        
        self.first_pm = self.num_switches
        self.last_pm = self.num_switches + self.num_pm - 1
        
        self.switch_neighbors = {i: [] for i in range(self.num_switches)}
        self.pm_to_edge = {}

        agg_start = self.num_core
        edge_start = self.num_core + self.num_agg
        
        # Core to Aggregate
        for pod in range(k):
            for agg_pos in range(k // 2):
                agg_idx = agg_start + pod * (k // 2) + agg_pos
                for core_pos in range(k // 2):
                    core_idx = agg_pos * (k // 2) + core_pos
                    self.switch_neighbors[core_idx].append(agg_idx)
                    self.switch_neighbors[agg_idx].append(core_idx)
        
        # Aggregate to Edge
        for pod in range(k):
            for agg_pos in range(k // 2):
                agg_idx = agg_start + pod * (k // 2) + agg_pos
                for edge_pos in range(k // 2):
                    edge_idx = edge_start + pod * (k // 2) + edge_pos
                    if edge_idx not in self.switch_neighbors[agg_idx]:
                        self.switch_neighbors[agg_idx].append(edge_idx)
                        self.switch_neighbors[edge_idx].append(agg_idx)
        
        # PM to Edge
        for pm_offset in range(self.num_pm):
            pm_idx = self.first_pm + pm_offset
            pod = pm_offset // ((k * k) // 4)
            edge_in_pod = (pm_offset % ((k * k) // 4)) // (k // 2)
            edge_idx = edge_start + pod * (k // 2) + edge_in_pod
            self.pm_to_edge[pm_idx] = edge_idx
    
    def generate_instance(self, min_visits_ratio: float = 0.3) -> dict:
        """
        Generate an n-stroll instance.
        
        If FatTree available: uses distance() to determine edges
        - distance == 1 → edge exists (cost = 1.0)
        - distance != 1 → no edge (cost = inf)
        
        If embedded fallback: uses pre-built adjacency lists
        
        Args:
            min_visits_ratio: Fraction of switches that must be visited
        
        Returns:
            Dictionary with cost_matrix, start_idx, end_idx, min_visits
        """
        # Select random start and end PMs
        start_pm = np.random.randint(self.first_pm, self.last_pm + 1)
        end_pm = np.random.randint(self.first_pm, self.last_pm + 1)
        while end_pm == start_pm:
            end_pm = np.random.randint(self.first_pm, self.last_pm + 1)
        
        # Remap graph (matches original fat-tree convention: switches first, PMs last):
        # Nodes 0 to num_switches-1: switches (in original order)
        # Node num_switches:     start PM
        # Node num_switches + 1: end PM
        n_nodes = self.num_switches + 2
        start_idx = self.num_switches      # e.g. 20 for k=4
        end_idx   = self.num_switches + 1  # e.g. 21 for k=4

        # Mapping: remapped index -> original fat tree index
        remap_to_orig = list(range(self.num_switches)) + [start_pm, end_pm]

        # Build cost matrix
        cost_matrix = torch.full((n_nodes, n_nodes), float('inf'))

        if FAT_TREE_AVAILABLE:
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        cost_matrix[i, j] = 0.0
                    else:
                        orig_i = remap_to_orig[i]
                        orig_j = remap_to_orig[j]
                        dist = self.fat_tree.distance(orig_i, orig_j, True)
                        if dist == 1:
                            cost_matrix[i, j] = 1.0
        else:
            # Embedded fallback: use pre-built adjacency
            for i in range(n_nodes):
                cost_matrix[i, i] = 0.0

            # Start PM connects to its edge switch (switches are at their original index)
            start_edge_remap = self.pm_to_edge[start_pm]
            cost_matrix[start_idx, start_edge_remap] = 1.0
            cost_matrix[start_edge_remap, start_idx] = 1.0

            # End PM connects to its edge switch
            end_edge_remap = self.pm_to_edge[end_pm]
            cost_matrix[end_idx, end_edge_remap] = 1.0
            cost_matrix[end_edge_remap, end_idx] = 1.0

            # Switch-to-switch connections
            for switch_orig in range(self.num_switches):
                for neighbor_orig in self.switch_neighbors[switch_orig]:
                    cost_matrix[switch_orig, neighbor_orig] = 1.0

        # Calculate min_visits
        min_visits = max(3, int(self.num_switches * min_visits_ratio) + 2)
        min_visits = min(min_visits, n_nodes)

        return {
            'cost_matrix': cost_matrix,
            'start_idx': torch.tensor(start_idx),
            'end_idx': torch.tensor(end_idx),
            'min_visits': torch.tensor(min_visits)
        }


# =============================================================================
# PUBLIC API - Call these from problem_top1.py
# =============================================================================

# Cache wrapper instances to avoid rebuilding topology each time
_wrapper_cache = {}

def generate_fat_tree_instance(k: int = 4, min_visits_ratio: float = 0.3) -> dict:
    """
    Generate an n-stroll instance on a fat tree topology.
    
    Simple adjacency: direct connection = cost 1, no connection = inf.
    
    Args:
        k: Fat tree parameter (must be even: 4, 6, 8, ...)
        min_visits_ratio: Fraction of switches that must be visited
    
    Returns:
        Dictionary with cost_matrix, start_idx, end_idx, min_visits
    """
    global _wrapper_cache
    
    if k not in _wrapper_cache:
        _wrapper_cache[k] = FatTreeNStrollWrapper(k)
    
    return _wrapper_cache[k].generate_instance(min_visits_ratio)


def get_fat_tree_graph_size(k: int) -> int:
    """
    Get the n-stroll graph size for a fat tree with parameter k.
    
    Returns: num_switches + 2 (for start and end PMs)
    """
    num_switches = (k * k) // 4 + (k * k) // 2 + (k * k) // 2
    return num_switches + 2


def is_fat_tree_available() -> bool:
    """Check if the external FatTree class was successfully imported."""
    return FAT_TREE_AVAILABLE


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing Fat Tree N-Stroll Wrapper")
    print("="*60)
    
    print(f"\nFatTree class available: {FAT_TREE_AVAILABLE}")
    
    for k in [4, 6]:
        print(f"\n--- Fat Tree k={k} ---")
        graph_size = get_fat_tree_graph_size(k)
        print(f"Graph size: {graph_size}")
        
        instance = generate_fat_tree_instance(k=k, min_visits_ratio=0.3)
        print(f"Cost matrix shape: {instance['cost_matrix'].shape}")
        print(f"Start idx: {instance['start_idx'].item()}")
        print(f"End idx: {instance['end_idx'].item()}")
        print(f"Min visits: {instance['min_visits'].item()}")
        
        # Count edges
        cm = instance['cost_matrix']
        finite_edges = ((~torch.isinf(cm)) & (cm > 0)).sum().item()
        print(f"Finite edges: {finite_edges}")