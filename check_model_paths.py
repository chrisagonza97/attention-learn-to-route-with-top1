"""
Check what paths the model actually generates.
"""

import torch
from torch.utils.data import DataLoader

def check_model_paths():
    from problems.top1.problem_top1 import TOP, TopDataset
    from nets.attention_model import AttentionModel
    
    print("="*60)
    print("Checking Model-Generated Paths")
    print("="*60)
    
    # Create model
    problem = TOP
    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        problem=problem,
        n_encode_layers=3
    )
    model.eval()
    model.set_decode_type('greedy')
    
    # Create small dataset
    dataset = TopDataset(size=20, num_samples=4, distribution='random')
    
    # Collate manually
    batch = {
        'cost_matrix': torch.stack([dataset[i]['cost_matrix'] for i in range(4)]),
        'start_idx': torch.stack([dataset[i]['start_idx'] for i in range(4)]),
        'end_idx': torch.stack([dataset[i]['end_idx'] for i in range(4)]),
        'min_visits': torch.stack([dataset[i]['min_visits'] for i in range(4)])
    }
    
    print(f"Start indices: {batch['start_idx'].tolist()}")
    print(f"End indices: {batch['end_idx'].tolist()}")
    print(f"Min visits: {batch['min_visits'].tolist()}")
    print()
    
    # Run model
    with torch.no_grad():
        cost, log_likelihood = model(batch)
    
    print(f"Costs from model: {cost.tolist()}")
    print()
    
    # Now manually run to get the paths
    with torch.no_grad():
        embeddings = model._init_embed(batch)
        _log_p, pi = model._inner(batch, embeddings)
    
    print(f"Path shape (pi): {pi.shape}")
    print(f"Paths:")
    for b in range(4):
        print(f"  Instance {b}: {pi[b].tolist()}")
        print(f"    Start should be: {batch['start_idx'][b].item()}")
        print(f"    End should be: {batch['end_idx'][b].item()}")
        print(f"    Path[0]: {pi[b, 0].item()}, Path[-1]: {pi[b, -1].item()}")
        
        # Check if start is in path
        if batch['start_idx'][b].item() in pi[b].tolist():
            start_pos = pi[b].tolist().index(batch['start_idx'][b].item())
            print(f"    Start node at position: {start_pos}")
        else:
            print(f"    WARNING: Start node NOT in path!")


if __name__ == "__main__":
    check_model_paths()