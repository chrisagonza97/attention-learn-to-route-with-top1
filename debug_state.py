"""
Debug script to test TOP state machine.
Run this from your attention-learn-to-route directory.
"""

import torch
import sys

# Test the state machine directly
def test_state_machine():
    from problems.top1.state_top1 import StateTop
    
    print("="*60)
    print("Testing TOP State Machine")
    print("="*60)
    
    batch_size = 2
    n_nodes = 5  # Small for debugging
    
    # Create simple test input
    # Complete graph with random costs
    torch.manual_seed(42)  # For reproducibility
    cost_matrix = torch.rand(batch_size, n_nodes, n_nodes)
    cost_matrix = (cost_matrix + cost_matrix.transpose(-1, -2)) / 2
    cost_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    
    start_idx = torch.tensor([0, 0])
    end_idx = torch.tensor([4, 4])  # End at node 4
    min_visits = torch.tensor([3, 3])  # Must visit 3 nodes
    
    input_dict = {
        'cost_matrix': cost_matrix,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'min_visits': min_visits
    }
    
    print(f"Setup: {n_nodes} nodes, start=0, end=4, min_visits=3")
    print()
    
    # Initialize state
    state = StateTop.initialize(input_dict)
    
    print(f"Initial state:")
    print(f"  prev_a (current node): {state.prev_a.squeeze().tolist()}")
    print(f"  n_visited: {state.n_visited.squeeze().tolist()}")
    print(f"  all_finished: {state.all_finished()}")
    print()
    
    # Simulate a few steps
    max_steps = 10
    for step in range(max_steps):
        print(f"--- Step {step} ---")
        
        # Get mask
        mask = state.get_mask()
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask[0]: {mask[0, 0].tolist()}")
        
        # Check if all masked
        all_masked = mask.all(dim=-1).squeeze(-1)
        print(f"  All masked: {all_masked.tolist()}")
        
        if state.all_finished():
            print(f"  FINISHED!")
            break
        
        # Select action - PRIORITIZE END NODE if available
        available = ~mask.squeeze(1)  # (batch_size, n_nodes)
        selected = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            avail_idx = torch.where(available[b])[0]
            end_node = end_idx[b].item()
            
            if len(avail_idx) > 0:
                # Check if end node is available - if so, pick it!
                if available[b, end_node]:
                    selected[b] = end_node
                    print(f"  Batch {b}: selecting END NODE {end_node} from available {avail_idx.tolist()}")
                else:
                    # Otherwise pick first available (simulating exploration)
                    selected[b] = avail_idx[0]
                    print(f"  Batch {b}: selecting node {selected[b].item()} from available {avail_idx.tolist()}")
            else:
                print(f"  Batch {b}: NO AVAILABLE NODES!")
                selected[b] = end_node  # Force end
        
        # Update state
        state = state.update(selected)
        
        print(f"  After update:")
        print(f"    prev_a: {state.prev_a.squeeze().tolist()}")
        print(f"    n_visited: {state.n_visited.squeeze().tolist()}")
        print(f"    lengths: {[f'{x:.4f}' for x in state.lengths.squeeze().tolist()]}")
        print(f"    all_finished: {state.all_finished()}")
        print()
        
        if state.all_finished():
            print("="*60)
            print("SUCCESS - State machine terminated correctly!")
            print(f"Final path cost: {state.lengths.squeeze().tolist()}")
            print("="*60)
            break
    else:
        print(f"WARNING: Did not finish after {max_steps} steps!")
        print("This indicates an infinite loop problem.")


if __name__ == "__main__":
    test_state_machine()