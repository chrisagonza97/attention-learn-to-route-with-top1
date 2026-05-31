from problems.top1.problem_top1 import generate_instance
from problems.top1.state_top1 import StateTop
import torch

inst = generate_instance(22, graph_type='fat_tree', fat_tree_k=4)
for k, v in inst.items():
    inst[k] = v.unsqueeze(0) if isinstance(v, torch.Tensor) else v

state = StateTop.initialize(inst)
print('Start:', state.prev_a.item(), 'End:', inst['end_idx'].item())

for step in range(80):
    if state.all_finished():
        print('FINISHED at step', step)
        break
    mask = state.get_mask()
    selected = torch.where(~mask.squeeze())[0][0].unsqueeze(0)
    state = state.update(selected)

print('Final node:', state.prev_a.item(), 'n_visited:', state.n_visited.item(), 'cost:', state.lengths.item())