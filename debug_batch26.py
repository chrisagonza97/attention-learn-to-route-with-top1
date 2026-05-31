"""
Debug script to find exactly what's causing NaN.
Run from attention-learn-to-route directory.
"""

import torch
import pickle
import os

def debug_validation():
    from problems.top1.state_top1 import StateTop
    from problems.top1.problem_top1 import TopDataset
    
    print("="*60)
    print("Debugging TOP Validation")
    print("="*60)
    
    # Load the validation dataset
    val_path = 'data/top1/top1_random20_validation_seed4321.pkl'
    
    if not os.path.exists(val_path):
        print(f"File not found: {val_path}")
        return
    
    # Check dataset size first
    with open(val_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Dataset size: {len(raw_data)} instances")
    
    dataset = TopDataset(filename=val_path, num_samples=len(raw_data))
    
    print(f"Loaded {len(dataset)} instances")
    print()
    
    # Test with small batch
    batch_size = min(32, len(dataset))
    batch_data = [dataset[i] for i in range(batch_size)]
    
    # Stack into tensors
    cost_matrices = torch.stack([d['cost_matrix'] for d in batch_data])
    start_idx = torch.stack([d['start_idx'] for d in batch_data])
    end_idx = torch.stack([d['end_idx'] for d in batch_data])
    min_visits = torch.stack([d['min_visits'] for d in batch_data])
    
    n_nodes = cost_matrices.size(1)
    
    print(f"Batch size: {batch_size}")
    print(f"N nodes: {n_nodes}")
    print(f"Min visits (first): {min_visits[0].item()}")
    print()
    
    # Check for problematic instances
    print("Checking all instances for issues...")
    problematic = []
    for i in range(len(dataset)):
        d = dataset[i]
        cm = d['cost_matrix']
        s = d['start_idx'].item()
        e = d['end_idx'].item()
        mv = d['min_visits'].item()
        
        # Check if start can reach enough nodes
        reachable_from_start = (~torch.isinf(cm[s])).sum().item() - 1  # -1 for self
        
        # Check overall connectivity - can we visit enough distinct nodes?
        # Simple check: are there enough nodes reachable from start?
        if reachable_from_start < mv - 1:  # -1 because start counts as visited
            problematic.append((i, f"Start {s} can only reach {reachable_from_start} nodes, need {mv-1} more"))
    
    if problematic:
        print(f"Found {len(problematic)} problematic instances:")
        for idx, reason in problematic[:10]:
            print(f"  Instance {idx}: {reason}")
    else:
        print("No obviously problematic instances found.")
    
    print()
    print("Running state machine simulation on first batch...")
    
    input_dict = {
        'cost_matrix': cost_matrices,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'min_visits': min_visits
    }
    
    state = StateTop.initialize(input_dict)
    
    max_steps = 100
    for step in range(max_steps):
        mask = state.get_mask()
        
        # Debug: print mask shape
        if step == 0:
            print(f"Mask shape: {mask.shape}")
        
        # Check for all-masked situations
        all_masked = mask.all(dim=-1)
        if all_masked.dim() > 1:
            all_masked = all_masked.squeeze(-1)
        
        if all_masked.any():
            problematic_indices = torch.where(all_masked)[0]
            print(f"\n*** Step {step}: ALL NODES MASKED for instances: {problematic_indices.tolist()}")
            
            for idx in problematic_indices[:3]:  # Show first 3
                idx = idx.item()
                print(f"\n  Instance {idx}:")
                print(f"    Current node: {state.prev_a[idx].item()}")
                print(f"    End node: {end_idx[idx].item()}")
                print(f"    n_visited: {state.n_visited[idx].item()}")
                print(f"    min_visits: {min_visits[idx].item()}")
                
                # Get the mask for this instance
                if mask.dim() == 3:
                    inst_mask = mask[idx, 0]
                else:
                    inst_mask = mask[idx]
                print(f"    Mask (True=blocked): {inst_mask.tolist()}")
                
                # Check reachability
                curr = state.prev_a[idx].item()
                batch_idx = state.ids[idx].item()
                edges = cost_matrices[batch_idx, curr, :]
                reachable = ~torch.isinf(edges)
                reachable[curr] = False  # exclude self
                print(f"    Reachable nodes: {torch.where(reachable)[0].tolist()}")
                print(f"    Visited: {torch.where(state.visited_[idx, 0] > 0)[0].tolist()}")
            
            print("\n*** BUG FOUND: All nodes masked but not finished!")
            return
        
        if state.all_finished():
            print(f"\nSUCCESS: Finished at step {step}")
            final_costs = state.lengths.squeeze(-1)
            print(f"Final costs: min={final_costs.min():.4f}, max={final_costs.max():.4f}, mean={final_costs.mean():.4f}")
            return
        
        # Select action - prefer end node when possible
        if mask.dim() == 3:
            available = ~mask.squeeze(1)
        else:
            available = ~mask
            
        selected = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            avail = torch.where(available[b])[0]
            if len(avail) > 0:
                # Prefer end node if available and have enough visits
                if state.n_visited[b].item() >= min_visits[b].item() and available[b, end_idx[b]]:
                    selected[b] = end_idx[b]
                else:
                    selected[b] = avail[0]
            else:
                print(f"No available nodes for instance {b} at step {step}!")
                selected[b] = end_idx[b]  # Fallback
        
        state = state.update(selected)
        
        if step % 10 == 0:
            at_end = (state.prev_a.squeeze(-1) == end_idx).sum().item()
            print(f"  Step {step}: n_visited=[{state.n_visited.min().item()}, {state.n_visited.max().item()}], at_end={at_end}/{batch_size}")
    
    print(f"\nDid not finish after {max_steps} steps")
    print("Checking final state...")
    at_end = (state.prev_a.squeeze(-1) == end_idx)
    enough = state.n_visited.squeeze(-1) >= min_visits
    print(f"At end: {at_end.sum().item()}/{batch_size}")
    print(f"Enough visits: {enough.sum().item()}/{batch_size}")


if __name__ == "__main__":
    debug_validation()