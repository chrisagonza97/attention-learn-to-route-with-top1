"""
DATA GENERATION FOR TOP1 PROBLEM
================================

Add this function to generate_data.py and update the main block accordingly.
"""

import numpy as np
import argparse
import os
import pickle


def check_extension(filename):
    """Ensure filename has .pkl extension."""
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    """Save dataset to pickle file."""
    filepath = check_extension(filename)
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {filepath}")


def generate_top1_data(dataset_size, graph_size, graph_type='random', sparsity=0.3, min_visits_ratio=0.3):
    """
    Generate n-stroll (k-stroll) problem data.
    
    Args:
        dataset_size: Number of instances to generate
        graph_size: Number of nodes in each graph  
        graph_type: Type of graph ('random', 'euclidean', 'complete', 'sparse')
        sparsity: For sparse graphs, probability that an edge does NOT exist
        min_visits_ratio: Ratio of nodes that must be visited
    
    Returns:
        List of tuples: (cost_matrix, start_idx, end_idx, min_visits)
    """
    data = []
    
    for _ in range(dataset_size):
        if graph_type == 'euclidean':
            loc = np.random.uniform(size=(graph_size, 2))
            cost_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=-1)
            
        elif graph_type == 'random':
            cost_matrix = np.random.uniform(size=(graph_size, graph_size))
            cost_matrix = (cost_matrix + cost_matrix.T) / 2
            np.fill_diagonal(cost_matrix, 0)
            
        elif graph_type == 'complete':
            cost_matrix = np.random.uniform(size=(graph_size, graph_size)) * 10
            cost_matrix = (cost_matrix + cost_matrix.T) / 2
            np.fill_diagonal(cost_matrix, 0)
            
        elif graph_type == 'sparse':
            cost_matrix = np.random.uniform(size=(graph_size, graph_size)) * 10
            mask = np.random.uniform(size=(graph_size, graph_size)) > sparsity
            mask = mask & mask.T
            cost_matrix = cost_matrix * mask
            cost_matrix[~mask] = np.inf
            np.fill_diagonal(cost_matrix, 0)
            
            # Ensure connectivity
            for i in range(graph_size):
                if np.all(np.isinf(cost_matrix[i])) or (np.min(cost_matrix[i][cost_matrix[i] != 0]) == np.inf):
                    next_node = (i + 1) % graph_size
                    edge_cost = np.random.uniform() * 10
                    cost_matrix[i, next_node] = edge_cost
                    cost_matrix[next_node, i] = edge_cost
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # Select start and end nodes
        start_idx = np.random.randint(0, graph_size)
        end_idx = np.random.randint(0, graph_size)
        while end_idx == start_idx:
            end_idx = np.random.randint(0, graph_size)
        
        min_visits = max(2, int(graph_size * min_visits_ratio))
        
        data.append((
            cost_matrix.tolist(),
            int(start_idx),
            int(end_idx),
            int(min_visits)
        ))
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='top1', help="Problem type: 'top1'")
    parser.add_argument('--data_distribution', type=str, default='random',
                        help="Graph type: 'random', 'euclidean', 'complete', 'sparse'")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances")
    parser.add_argument("-f", action='store_true', help="Overwrite existing files")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--sparsity', type=float, default=0.3, help="Sparsity for sparse graphs")
    parser.add_argument('--min_visits_ratio', type=float, default=0.3, help="Minimum visit ratio")

    opts = parser.parse_args()
    
    np.random.seed(opts.seed)
    
    for graph_size in opts.graph_sizes:
        datadir = os.path.join(opts.data_dir, opts.problem)
        os.makedirs(datadir, exist_ok=True)
        
        if opts.filename is None:
            filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                opts.problem,
                "_{}".format(opts.data_distribution) if opts.data_distribution else "",
                graph_size, opts.name, opts.seed))
        else:
            filename = check_extension(opts.filename)
        
        if not opts.f and os.path.isfile(filename):
            print(f"File {filename} already exists! Use -f to overwrite.")
            continue
        
        print(f"Generating {opts.dataset_size} instances of size {graph_size}...")
        dataset = generate_top1_data(
            opts.dataset_size,
            graph_size,
            graph_type=opts.data_distribution,
            sparsity=opts.sparsity,
            min_visits_ratio=opts.min_visits_ratio
        )
        
        print(f"Sample instance: {dataset[0]}")
        save_dataset(dataset, filename)