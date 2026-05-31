import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_op_data(dataset_size, op_size, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


def get_fat_tree_graph_size(k):
    """Compute graph size for fat tree with parameter k."""
    num_switches = (k * k) // 4 + (k * k) // 2 + (k * k) // 2
    return num_switches + 2


def generate_top_data(dataset_size, graph_size, graph_type='random', sparsity=0.3, min_visits_ratio=0.3, fat_tree_k=None, min_visits=None):
    """
    Generate n-stroll (k-stroll) problem data with adjacency matrices.
    
    Args:
        dataset_size: Number of instances to generate
        graph_size: Number of nodes in each graph (ignored for fat_tree if fat_tree_k set)
        graph_type: Type of graph ('random', 'euclidean', 'complete', 'sparse', 'fat_tree')
        sparsity: For sparse graphs, probability that an edge does NOT exist (default 0.3)
        min_visits_ratio: Ratio of nodes that must be visited (default 0.3 = 30%)
        fat_tree_k: For fat_tree, the k parameter (4, 6, 8...)
        min_visits: Exact minimum visits (overrides min_visits_ratio if set)
    
    Returns:
        List of tuples: (cost_matrix, start_idx, end_idx, min_visits)
    """
    data = []
    
    # Handle fat_tree specially
    if graph_type == 'fat_tree':
        if fat_tree_k is None:
            # Derive k from size
            k_squared = (graph_size - 2) * 4 / 5
            fat_tree_k = int(np.sqrt(k_squared))
            fat_tree_k = max(4, (fat_tree_k // 2) * 2)
        
        # Recompute actual graph size from k
        graph_size = get_fat_tree_graph_size(fat_tree_k)
        num_switches = graph_size - 2
        
        # Calculate min_visits
        if min_visits is None:
            min_visits_val = max(3, int(num_switches * min_visits_ratio) + 2)
        else:
            min_visits_val = min_visits
        
        # Try to import the wrapper that uses the FatTree class
        try:
            from fattree.fat_tree_wrapper import generate_fat_tree_instance
            print(f"Generating {dataset_size} fat tree instances (k={fat_tree_k}, graph_size={graph_size}, min_visits={min_visits_val})...")
            for i in range(dataset_size):
                instance = generate_fat_tree_instance(fat_tree_k, min_visits_ratio)
                # Override min_visits if specified
                data.append((
                    instance['cost_matrix'].numpy().tolist(),
                    int(instance['start_idx'].item()),
                    int(instance['end_idx'].item()),
                    min_visits_val
                ))
                if (i + 1) % 1000 == 0:
                    print(f"  Generated {i + 1}/{dataset_size}")
            return data
        except ImportError as e:
            print(f"Warning: Could not import fat_tree_wrapper ({e}), using embedded generator")
            # Use embedded generator below
            return generate_fat_tree_embedded(dataset_size, fat_tree_k, min_visits_val)
    
    # Calculate min_visits for non-fat-tree graphs
    if min_visits is None:
        min_visits_val = max(2, int(graph_size * min_visits_ratio))
    else:
        min_visits_val = min_visits
    
    print(f"Generating {dataset_size} {graph_type} instances (graph_size={graph_size}, min_visits={min_visits_val})...")
    
    # Non-fat-tree graphs
    for _ in range(dataset_size):
        # Generate cost matrix based on graph type
        if graph_type == 'euclidean':
            # Generate Euclidean graph (for compatibility/testing)
            loc = np.random.uniform(size=(graph_size, 2))
            # Calculate pairwise distances
            cost_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=-1)
            
        elif graph_type == 'random':
            # Random edge weights in [0, 1]
            cost_matrix = np.random.uniform(size=(graph_size, graph_size))
            # Make symmetric (undirected graph)
            cost_matrix = (cost_matrix + cost_matrix.T) / 2
            # Zero diagonal (no self-loops)
            np.fill_diagonal(cost_matrix, 0)
            
        elif graph_type == 'complete':
            # Complete graph with random weights in [0, 10]
            cost_matrix = np.random.uniform(size=(graph_size, graph_size)) * 10
            cost_matrix = (cost_matrix + cost_matrix.T) / 2
            np.fill_diagonal(cost_matrix, 0)
            
        elif graph_type == 'sparse':
            # Sparse random graph
            cost_matrix = np.random.uniform(size=(graph_size, graph_size)) * 10
            # Create sparsity mask (some edges don't exist)
            mask = np.random.uniform(size=(graph_size, graph_size)) > sparsity
            # Make symmetric
            mask = mask & mask.T
            # Apply mask (non-existent edges have infinite cost)
            cost_matrix = cost_matrix * mask
            cost_matrix[~mask] = np.inf
            np.fill_diagonal(cost_matrix, 0)
            
            # Ensure graph is connected (at least create a spanning tree)
            # Simple approach: ensure each node connects to at least one other node
            for i in range(graph_size):
                if np.min(cost_matrix[i]) == np.inf and i < graph_size - 1:
                    # Connect to next node with finite cost
                    next_node = (i + 1) % graph_size
                    edge_cost = np.random.uniform() * 10
                    cost_matrix[i, next_node] = edge_cost
                    cost_matrix[next_node, i] = edge_cost
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # Randomly select start and end nodes (must be different)
        start_idx = np.random.randint(0, graph_size)
        end_idx = np.random.randint(0, graph_size)
        while end_idx == start_idx:
            end_idx = np.random.randint(0, graph_size)
        
        # Calculate minimum number of nodes to visit (at least 2: start and end)
        min_visits = max(2, int(graph_size * min_visits_ratio))
        
        data.append((
            cost_matrix.tolist(),
            start_idx,
            end_idx,
            min_visits_val
        ))
    
    return data


def generate_fat_tree_embedded(dataset_size, k, min_visits):
    """
    Embedded fat tree generator (no external dependencies).
    Used as fallback if fat_tree_wrapper is not available.
    
    Args:
        dataset_size: Number of instances
        k: Fat tree parameter
        min_visits: Exact minimum visits required
    """
    data = []
    
    num_core = (k * k) // 4
    num_agg = (k * k) // 2
    num_edge = (k * k) // 2
    num_switches = num_core + num_agg + num_edge
    num_pm = (k * k * k) // 4
    first_pm = num_switches
    n_nodes = num_switches + 2
    
    agg_start = num_core
    edge_start = num_core + num_agg
    
    # Build switch adjacency once
    switch_neighbors = {i: [] for i in range(num_switches)}
    
    # Core to Aggregate
    for pod in range(k):
        for agg_pos in range(k // 2):
            agg_idx = agg_start + pod * (k // 2) + agg_pos
            for core_pos in range(k // 2):
                core_idx = agg_pos * (k // 2) + core_pos
                switch_neighbors[core_idx].append(agg_idx)
                switch_neighbors[agg_idx].append(core_idx)
    
    # Aggregate to Edge
    for pod in range(k):
        for agg_pos in range(k // 2):
            agg_idx = agg_start + pod * (k // 2) + agg_pos
            for edge_pos in range(k // 2):
                edge_idx = edge_start + pod * (k // 2) + edge_pos
                if edge_idx not in switch_neighbors[agg_idx]:
                    switch_neighbors[agg_idx].append(edge_idx)
                    switch_neighbors[edge_idx].append(agg_idx)
    
    # PM to edge mapping
    pm_to_edge = {}
    for pm_offset in range(num_pm):
        pm_idx = first_pm + pm_offset
        pod = pm_offset // ((k * k) // 4)
        edge_in_pod = (pm_offset % ((k * k) // 4)) // (k // 2)
        pm_to_edge[pm_idx] = edge_start + pod * (k // 2) + edge_in_pod
    
    print(f"Generating {dataset_size} fat tree instances (k={k}, graph_size={n_nodes}, min_visits={min_visits}) [embedded]...")
    
    for i in range(dataset_size):
        # Random start and end PMs
        start_pm = np.random.randint(first_pm, first_pm + num_pm)
        end_pm = np.random.randint(first_pm, first_pm + num_pm)
        while end_pm == start_pm:
            end_pm = np.random.randint(first_pm, first_pm + num_pm)
        
        # Build cost matrix
        cost_matrix = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(cost_matrix, 0)
        
        # Start PM connects to its edge
        start_edge_remap = pm_to_edge[start_pm] + 1
        cost_matrix[0, start_edge_remap] = 1.0
        cost_matrix[start_edge_remap, 0] = 1.0
        
        # End PM connects to its edge
        end_edge_remap = pm_to_edge[end_pm] + 1
        cost_matrix[n_nodes - 1, end_edge_remap] = 1.0
        cost_matrix[end_edge_remap, n_nodes - 1] = 1.0
        
        # Switch-to-switch
        for switch_orig in range(num_switches):
            switch_remap = switch_orig + 1
            for neighbor_orig in switch_neighbors[switch_orig]:
                neighbor_remap = neighbor_orig + 1
                cost_matrix[switch_remap, neighbor_remap] = 1.0
        
        min_visits_val = min(min_visits, n_nodes)
        
        data.append((
            cost_matrix.tolist(),
            0,  # start_idx is always 0
            n_nodes - 1,  # end_idx is always last
            min_visits_val
        ))
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{dataset_size}")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem: 'tsp', 'vrp', 'pctsp', 'op', 'top', or 'all'")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    
    # TOP-specific arguments
    parser.add_argument('--graph_type', type=str, default='random',
                        help="Graph type for TOP: 'random', 'euclidean', 'complete', 'sparse', 'fat_tree'")
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help="Sparsity for sparse graphs (default: 0.3)")
    parser.add_argument('--min_visits_ratio', type=float, default=0.3,
                        help="Minimum visit ratio (default: 0.3), ignored if --min_visits is set")
    parser.add_argument('--min_visits', type=int, default=None,
                        help="Exact minimum visits required (overrides --min_visits_ratio)")
    parser.add_argument('--fat_tree_k', type=int, default=None,
                        help="Fat tree k parameter (4, 6, 8...). Overrides graph_sizes for fat_tree.")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'vrp': [None],
        'pctsp': [None],
        'op': ['const', 'unif', 'dist'],
        'top': ['random', 'euclidean', 'complete', 'sparse', 'fat_tree']
    }
    
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem.get(opts.problem, [None])
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            
            # Handle fat_tree_k: override graph_sizes
            if opts.fat_tree_k is not None and problem == 'top':
                graph_sizes_to_use = [get_fat_tree_graph_size(opts.fat_tree_k)]
                distribution = 'fat_tree'  # Force fat_tree distribution
            else:
                graph_sizes_to_use = opts.graph_sizes
            
            for graph_size in graph_sizes_to_use:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(opts.dataset_size, graph_size)
                elif problem == 'pctsp':
                    dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                elif problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                elif problem == "top":
                    graph_type_to_use = distribution if distribution else opts.graph_type
                    dataset = generate_top_data(
                        opts.dataset_size, 
                        graph_size, 
                        graph_type=graph_type_to_use,
                        sparsity=opts.sparsity,
                        min_visits_ratio=opts.min_visits_ratio,
                        fat_tree_k=opts.fat_tree_k,
                        min_visits=opts.min_visits
                    )
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(f"Sample instance: {dataset[0][:2]}...")  # Print first part of first instance
                print(f"Saving {len(dataset)} instances to {filename}")
                save_dataset(dataset, filename)