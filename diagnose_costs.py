"""
Diagnostic script to understand why costs are ~1002 instead of ~2-5.
Run from attention-learn-to-route directory.
"""

import torch
import pickle

def diagnose_costs():
    from problems.top1.problem_top1 import TOP, TopDataset
    from problems.top1.state_top1 import StateTop
    
    print("="*60)
    print("Diagnosing TOP Costs")
    print("="*60)
    
    # Generate a small batch
    dataset = TopDataset(size=20, num_samples=10, distribution='random')
    
    # Collate
    batch = {
        'cost_matrix': torch.stack([d['cost_matrix'] for d in [dataset[i] for i in range(10)]]),
        'start_idx': torch.stack([d['start_idx'] for d in [dataset[i] for i in range(10)]]),
        'end_idx': torch.stack([d['end_idx'] for d in [dataset[i] for i in range(10)]]),
        'min_visits': torch.stack([d['min_visits'] for d in [dataset[i] for i in range(10)]])
    }
    
    batch_size = 10
    
    print(f"Batch size: {batch_size}")
    print(f"Start indices: {batch['start_idx'].tolist()}")
    print(f"End indices: {batch['end_idx'].tolist()}")
    print(f"Min visits: {batch['min_visits'].tolist()}")
    print()
    
    # Simulate a path through the state machine
    state = StateTop.initialize(batch)
    
    actions = []
    max_steps = 30
    
    for step in range(max_steps):
        mask = state.get_mask()
        
        if state.all_finished():
            print(f"Finished at step {step}")
            break
        
        # Select first available (or end node if available and enough visits)
        available = ~mask.squeeze(1)
        selected = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            avail = torch.where(available[b])[0]
            if len(avail) > 0:
                end_node = batch['end_idx'][b].item()
                if state.n_visited[b].item() >= batch['min_visits'][b].item() and available[b, end_node]:
                    selected[b] = end_node
                else:
                    selected[b] = avail[0]
        
        actions.append(selected.clone())
        state = state.update(selected)
    
    # Construct the path (pi)
    # Start with start_idx, then all actions
    pi_list = [batch['start_idx']]
    pi_list.extend(actions)
    pi = torch.stack(pi_list, dim=1)
    
    print(f"\nConstructed paths (pi):")
    print(f"  Shape: {pi.shape}")
    for b in range(min(3, batch_size)):
        print(f"  Instance {b}: {pi[b].tolist()}")
        print(f"    Start: {batch['start_idx'][b].item()}, End: {batch['end_idx'][b].item()}")
        print(f"    Path starts at: {pi[b, 0].item()}, Path ends at: {pi[b, -1].item()}")
        print(f"    Unique nodes: {torch.unique(pi[b]).numel()}, Min required: {batch['min_visits'][b].item()}")
    
    # Calculate costs
    print(f"\nCalculating costs...")
    costs, _ = TOP.get_costs(batch, pi)
    
    print(f"\nCosts: {costs.tolist()}")
    print(f"Mean cost: {costs.mean().item():.4f}")
    
    # Check what penalties are being applied
    print(f"\n--- Penalty breakdown ---")
    for b in range(min(3, batch_size)):
        path = pi[b]
        start_correct = (path[0] == batch['start_idx'][b]).item()
        end_correct = (path[-1] == batch['end_idx'][b]).item()
        n_unique = torch.unique(path).numel()
        min_visits = batch['min_visits'][b].item()
        visits_ok = n_unique >= min_visits
        
        # Calculate raw path cost
        raw_cost = 0
        for i in range(len(path) - 1):
            edge = batch['cost_matrix'][b, path[i], path[i+1]].item()
            raw_cost += edge
        
        print(f"\nInstance {b}:")
        print(f"  Start correct: {start_correct}")
        print(f"  End correct: {end_correct} (path ends at {path[-1].item()}, should be {batch['end_idx'][b].item()})")
        print(f"  Visits OK: {visits_ok} ({n_unique} >= {min_visits})")
        print(f"  Raw path cost: {raw_cost:.4f}")
        print(f"  Reported cost: {costs[b].item():.4f}")
        
        expected_penalty = 0
        if not start_correct:
            expected_penalty += 1000
        if not end_correct:
            expected_penalty += 1000
        if not visits_ok:
            expected_penalty += (min_visits - n_unique) * 1000
        
        print(f"  Expected penalty: {expected_penalty}")
        print(f"  Expected total: {raw_cost + expected_penalty:.4f}")


if __name__ == "__main__":
    diagnose_costs()