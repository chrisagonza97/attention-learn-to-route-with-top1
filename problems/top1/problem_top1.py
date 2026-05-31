from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
from problems.top1.state_top1 import StateTop
from utils.beam_search import beam_search


class TOP(object):
    """
    n-stroll (k-stroll) problem:
    - Single walk from start node s to end node t
    - Must visit at least n distinct nodes
    - Objective: minimize total path cost
    """

    NAME = 'top'

    @staticmethod
    def get_costs(dataset, pi):
        """
        Calculate the cost of the path.
        
        Note: pi contains the sequence of ACTIONS (nodes selected), starting from the first
        action taken FROM start_idx. So the actual path is [start_idx] + pi.
        
        Args:
            dataset: Dictionary with 'cost_matrix', 'start_idx', 'end_idx', 'min_visits'
            pi: Tensor of shape (batch_size, path_length) containing node indices (actions)
        
        Returns:
            costs: Tensor of shape (batch_size,) with path costs
            None: placeholder for compatibility
        """
        batch_size = pi.size(0)
        device = pi.device
        
        # Large penalty for invalid paths (but finite to allow gradient computation)
        LARGE_PENALTY = 1000.0
        
        if pi.size(-1) == 0:
            return torch.full((batch_size,), LARGE_PENALTY, dtype=torch.float, device=device), None
        
        cost_matrix = dataset['cost_matrix']
        start_idx = dataset['start_idx']
        end_idx = dataset['end_idx']
        min_visits = dataset['min_visits']
        
        # Prepend start_idx to pi to get the full path
        # pi contains actions taken FROM start, so full path is [start] + [actions]
        start_idx_col = start_idx.unsqueeze(1)  # (batch_size, 1)
        full_path = torch.cat([start_idx_col, pi], dim=1)  # (batch_size, path_length + 1)
        
        # Calculate path cost
        total_cost = torch.zeros(batch_size, dtype=torch.float, device=device)
        
        if full_path.size(-1) > 1:
            # Get consecutive node pairs
            from_nodes = full_path[:, :-1]  # (batch_size, path_length)
            to_nodes = full_path[:, 1:]      # (batch_size, path_length)
            
            # Gather edge costs - vectorized
            batch_idx = torch.arange(batch_size, device=device)[:, None].expand_as(from_nodes)
            edge_costs = cost_matrix[batch_idx, from_nodes, to_nodes]
            
            # Replace inf with large penalty
            edge_costs = torch.where(torch.isinf(edge_costs), 
                                     torch.full_like(edge_costs, LARGE_PENALTY),
                                     edge_costs)
            
            total_cost = edge_costs.sum(dim=1)
        
        # Check end node - the last action should be the end node
        ends_incorrect = (pi[:, -1] != end_idx)
        total_cost = total_cost + ends_incorrect.float() * LARGE_PENALTY
        
        # Check minimum visits constraint on the full path
        n_unique = torch.zeros(batch_size, dtype=torch.long, device=device)
        for b in range(batch_size):
            n_unique[b] = torch.unique(full_path[b]).numel()
        
        visits_deficit = torch.clamp(min_visits - n_unique, min=0).float()
        total_cost = total_cost + visits_deficit * LARGE_PENALTY
        
        return total_cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TopDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTop.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TOP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


# =============================================================================
# FAT TREE INTEGRATION
# =============================================================================
# Import from the fattree folder

try:
    from fattree.fat_tree_wrapper import (
        generate_fat_tree_instance,
        get_fat_tree_graph_size,
        is_fat_tree_available
    )
    if is_fat_tree_available():
        print("[problem_top1] Using FatTree class for fat_tree instances")
    else:
        print("[problem_top1] Using embedded topology for fat_tree instances")
except ImportError:
    # Fallback: embedded minimal implementation
    print("[problem_top1] fat_tree_wrapper not found, using embedded topology")
    
    def generate_fat_tree_instance(k: int = 4, min_visits_ratio: float = 0.3):
        """Minimal embedded fat tree generator."""
        num_core = (k * k) // 4
        num_agg = (k * k) // 2
        num_edge = (k * k) // 2
        num_switches = num_core + num_agg + num_edge
        num_pm = (k * k * k) // 4
        first_pm = num_switches
        
        n_nodes = num_switches + 2
        
        # Select random PMs
        start_pm = np.random.randint(first_pm, first_pm + num_pm)
        end_pm = np.random.randint(first_pm, first_pm + num_pm)
        while end_pm == start_pm:
            end_pm = np.random.randint(first_pm, first_pm + num_pm)
        
        # Build minimal cost matrix
        cost_matrix = torch.full((n_nodes, n_nodes), float('inf'))
        for i in range(n_nodes):
            cost_matrix[i, i] = 0
        
        # PM to edge connections
        edge_start = num_core + num_agg
        
        def pm_to_edge(pm_idx):
            pm_offset = pm_idx - first_pm
            pod = pm_offset // ((k * k) // 4)
            edge_in_pod = (pm_offset % ((k * k) // 4)) // (k // 2)
            return edge_start + pod * (k // 2) + edge_in_pod
        
        start_edge = pm_to_edge(start_pm) + 1
        end_edge = pm_to_edge(end_pm) + 1
        
        cost_matrix[0, start_edge] = 1.0
        cost_matrix[start_edge, 0] = 1.0
        cost_matrix[n_nodes-1, end_edge] = 1.0
        cost_matrix[end_edge, n_nodes-1] = 1.0
        
        # Switch connections (simplified)
        agg_start = num_core
        for pod in range(k):
            for agg_pos in range(k // 2):
                agg_idx = agg_start + pod * (k // 2) + agg_pos + 1
                for core_pos in range(k // 2):
                    core_idx = agg_pos * (k // 2) + core_pos + 1
                    cost_matrix[core_idx, agg_idx] = 1.0
                    cost_matrix[agg_idx, core_idx] = 1.0
                for edge_pos in range(k // 2):
                    edge_idx = edge_start + pod * (k // 2) + edge_pos + 1
                    cost_matrix[agg_idx, edge_idx] = 1.0
                    cost_matrix[edge_idx, agg_idx] = 1.0
        
        min_visits = max(3, int(num_switches * min_visits_ratio) + 2)
        
        return {
            'cost_matrix': cost_matrix,
            'start_idx': torch.tensor(0),
            'end_idx': torch.tensor(n_nodes - 1),
            'min_visits': torch.tensor(min_visits)
        }
    
    def get_fat_tree_graph_size(k: int) -> int:
        num_switches = (k * k) // 4 + (k * k) // 2 + (k * k) // 2
        return num_switches + 2
    
    def is_fat_tree_available() -> bool:
        return False


# =============================================================================
# GENERAL INSTANCE GENERATOR
# =============================================================================

def generate_instance(size, min_visits_ratio=0.3, graph_type='random', sparsity=0.3, fat_tree_k=None, min_visits=None):
    """
    Generate an n-stroll problem instance with adjacency matrix.
    
    Args:
        size: Total number of nodes (ignored for fat_tree if fat_tree_k is set)
        min_visits_ratio: Ratio of nodes that must be visited (default 0.3)
        graph_type: Type of graph ('random', 'euclidean', 'complete', 'sparse', 'fat_tree')
        sparsity: For sparse graphs, probability that an edge does NOT exist
        fat_tree_k: For fat_tree type, the k parameter (4, 6, 8...)
        min_visits: Exact minimum visits (overrides min_visits_ratio if set)
    
    Returns:
        Dictionary with problem instance data
    """
    
    # Handle fat tree specially
    if graph_type == 'fat_tree':
        # If fat_tree_k not provided, derive from size
        if fat_tree_k is None:
            # size = num_switches + 2 = 5k²/4 + 2
            # k² = (size - 2) * 4 / 5
            k_squared = (size - 2) * 4 / 5
            fat_tree_k = int(np.sqrt(k_squared))
            fat_tree_k = max(4, (fat_tree_k // 2) * 2)  # Round to even
        instance = generate_fat_tree_instance(fat_tree_k, min_visits_ratio)
        # Override min_visits if specified
        if min_visits is not None:
            instance['min_visits'] = torch.tensor(min_visits)
        return instance
    
    # Generate cost matrix based on graph type
    if graph_type == 'euclidean':
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        cost_matrix = torch.cdist(loc, loc, p=2)
        
    elif graph_type == 'random':
        cost_matrix = torch.rand(size, size)
        cost_matrix = (cost_matrix + cost_matrix.t()) / 2
        cost_matrix.fill_diagonal_(0)
        
    elif graph_type == 'complete':
        cost_matrix = torch.rand(size, size) * 10
        cost_matrix = (cost_matrix + cost_matrix.t()) / 2
        cost_matrix.fill_diagonal_(0)
        
    elif graph_type == 'sparse':
        cost_matrix = torch.rand(size, size) * 10
        mask = torch.rand(size, size) > sparsity
        mask = mask & mask.t()
        cost_matrix = cost_matrix * mask.float()
        cost_matrix[~mask] = float('inf')
        cost_matrix.fill_diagonal_(0)
        
        # Ensure connectivity
        for i in range(size):
            if cost_matrix[i].min() == float('inf') and i < size - 1:
                next_node = (i + 1) % size
                edge_cost = torch.rand(1).item() * 10
                cost_matrix[i, next_node] = edge_cost
                cost_matrix[next_node, i] = edge_cost
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Randomly select start and end nodes (must be different)
    start_idx = torch.randint(0, size, (1,)).item()
    end_idx = torch.randint(0, size, (1,)).item()
    while end_idx == start_idx:
        end_idx = torch.randint(0, size, (1,)).item()
    
    # Calculate minimum number of nodes to visit
    if min_visits is None:
        min_visits_val = max(2, int(size * min_visits_ratio))
    else:
        min_visits_val = min_visits
    
    return {
        'cost_matrix': cost_matrix,
        'start_idx': torch.tensor(start_idx),
        'end_idx': torch.tensor(end_idx),
        'min_visits': torch.tensor(min_visits_val)
    }


# =============================================================================
# DATASET CLASS
# =============================================================================

def get_fat_tree_size(k: int) -> int:
    """
    Compute the n-stroll graph size for a fat tree with parameter k.
    
    Returns: num_switches + 2 (start PM + end PM)
    
    Use this to set graph_size when using fat_tree:
        graph_size = get_fat_tree_size(k=4)  # Returns 22
    """
    num_switches = (k * k) // 4 + (k * k) // 2 + (k * k) // 2
    return num_switches + 2


class TopDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, 
                 distribution=None, graph_type=None, min_visits_ratio=0.3, sparsity=0.3,
                 fat_tree_k=None, min_visits=None):
        """
        Dataset for n-stroll problem with adjacency matrices.
        
        Args:
            filename: Path to .pkl file to load (None for on-the-fly generation)
            size: Number of nodes (ignored for fat_tree)
            num_samples: Number of instances to generate
            offset: Offset for loading from file
            distribution: Graph type (alias for graph_type for compatibility)
            graph_type: 'random', 'euclidean', 'complete', 'sparse', 'fat_tree'
            min_visits_ratio: Fraction of nodes that must be visited (ignored if min_visits set)
            sparsity: For sparse graphs
            fat_tree_k: Fat tree parameter (default: derived from size)
            min_visits: Exact minimum visits (overrides min_visits_ratio)
        """
        super(TopDataset, self).__init__()
        
        # Map distribution to graph_type for compatibility
        if distribution is not None and graph_type is None:
            graph_type = distribution
        elif graph_type is None:
            graph_type = 'random'

        self.graph_type = graph_type
        self.fat_tree_k = fat_tree_k
        self.min_visits = min_visits

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'cost_matrix': torch.FloatTensor(cost_matrix),
                        'start_idx': torch.tensor(start_idx),
                        'end_idx': torch.tensor(end_idx),
                        'min_visits': torch.tensor(min_visits if self.min_visits is None else self.min_visits)
                    }
                    for cost_matrix, start_idx, end_idx, min_visits in (data[offset:offset+num_samples])
                ]
        else:
            self.data = [
                generate_instance(size, min_visits_ratio, graph_type, sparsity, fat_tree_k, min_visits)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_fat_tree_info(k: int):
    """Print information about a fat tree topology."""
    num_core = (k * k) // 4
    num_agg = (k * k) // 2
    num_edge = (k * k) // 2
    num_switches = num_core + num_agg + num_edge
    num_pm = (k * k * k) // 4
    graph_size = num_switches + 2
    
    print(f"\nFat Tree k={k} Topology:")
    print(f"  Core switches:      {num_core}")
    print(f"  Aggregate switches: {num_agg}")
    print(f"  Edge switches:      {num_edge}")
    print(f"  Total switches:     {num_switches}")
    print(f"  Physical machines:  {num_pm}")
    print(f"  N-stroll graph size: {graph_size} (use --graph_size {graph_size})")
    print(f"  Default min_visits (30%): {max(3, int(num_switches * 0.3) + 2)}")


if __name__ == "__main__":
    # Print fat tree info for common k values
    for k in [4, 6, 8]:
        print_fat_tree_info(k)
    
    # Test generation
    print("\n" + "="*60)
    print("Testing instance generation...")
    print("="*60)
    
    # Test random
    inst = generate_instance(20, graph_type='random')
    print(f"\nRandom graph: {inst['cost_matrix'].shape}, start={inst['start_idx'].item()}, end={inst['end_idx'].item()}")
    
    # Test fat tree
    inst = generate_instance(22, graph_type='fat_tree', fat_tree_k=4)
    print(f"Fat tree k=4: {inst['cost_matrix'].shape}, start={inst['start_idx'].item()}, end={inst['end_idx'].item()}")
    
    # Test that size is ignored for fat_tree when k is provided
    inst = generate_instance(100, graph_type='fat_tree', fat_tree_k=4)  # size=100 should be ignored
    print(f"Fat tree k=4 (size ignored): {inst['cost_matrix'].shape}")