# HANDOFF — TOP-1 / N-Stroll Problem on Fat-Tree Networks

## Goal

Add the **N-Stroll (K-Stroll) problem** to the attention-learn-to-route codebase and train it on fat-tree data center topologies. The other problems in the repo (TSP, VRP, OP, PCTSP) are untouched and out of scope.

**The N-Stroll problem:**
- Given a graph with a cost matrix, start node `s`, end node `t`, and minimum visit count `min_visits`
- Find the shortest path from `s` to `t` that visits at least `min_visits` distinct nodes
- Backtracking is allowed; revisits don't count toward `min_visits`
- Structurally the inverse of the Orienteering Problem (minimize cost + min-visit constraint vs. maximize prize + budget constraint)

**Application:** VM migration in fat-tree data center networks — traffic must traverse a minimum number of switches for VNF/service-function chaining.

**Planned extension:** Change `min_visits` from a count to a **specific set of required nodes** (e.g. visit nodes {5, 12, 17}). This is the Steiner path variant — NP-hard in general — and is where the neural network has a genuine advantage over BFS. See "Next Steps" for what this requires.

---

## Files in Scope

### Created from scratch
| File | Purpose |
|------|---------|
| `problems/top1/problem_top1.py` | `TOP` class, dataset, instance generator (random/euclidean/sparse/fat_tree graph types) |
| `problems/top1/state_top1.py` | `StateTop` NamedTuple — step logic, masking, visit counting |
| `problems/top1/__init__.py` | Empty init |
| `fattree/fat_tree_wrapper.py` | Converts FatTree topology to TOP cost-matrix format |
| `fattree/vm_pair.py` | Stub (was missing, blocked FatTree import) |
| `fattree/sized_vm_pair.py` | Stub (was missing, blocked FatTree import) |
| `fattree/ac_migrate_pytorch.py` | Stub (was missing, blocked FatTree import) |
| `inspect_model.py` | Visualization tool — shows BFS optimal vs model path on fat-tree topology |
| `diagnose_costs2.py` | Diagnostic script — edge costs, BFS costs, topology stats |

### Modified
| File | Change |
|------|--------|
| `fattree/fat_tree.py` | Changed bare imports to relative imports; guarded `pulp` with try/except |
| `nets/attention_model.py` | Added `is_top` flag; custom `_init_embed()` that builds node features from cost matrix |
| `problems/__init__.py` | Added `from problems.top1.problem_top1 import TOP` |
| `options.py` | Added `--fat_tree_k` and `--min_visits` CLI args |
| `generate_data.py` | Added TOP data generation support |
| `utils/functions.py` | Registered `'top': TOP` in problem registry |

Debug/diagnostic scripts at repo root: `debug.py`, `debug_state.py`, `debug_batch26.py`, `diagnose_costs.py`, `diagnose_costs2.py`, `check_model_paths.py`

---

## Problem Structure — Key Facts

- **k=4 fat-tree:** 22 nodes total — 4 core + 8 agg + 8 edge switches (nodes 0–19), start PM (node 20), end PM (node 21)
- **All edge costs = 1.0** (hop count)
- **Optimal path length:** 2–6 hops depending on start/end pair (2 if same edge switch, 6 if cross-pod)
- **Distinct problem instances:** 16 PMs × 15 = **240** distinct (start, end) pairs — very small space
- **`min_visits=3` is trivially satisfied** by any valid path — shortest paths already visit ≥3 nodes. The constraint only bites at higher values (e.g. default ratio 0.3 → min_visits=8)
- **Untrained model score:** ~47 (oscillates 2 nodes for 44 steps, then BFS safety forces it to end)
- **`name` vs folder:** Problem registers as `'top'` (CLI flag, outputs dir), code lives in `problems/top1/`

---

## Current Progress

### Completed this session

1. **Pushed to GitHub** — repo is at https://github.com/chrisagonza97/attention-learn-to-route-with-top1 (not a fork, just a push of the local clone; no "forked from" badge but full history is there)

2. **Cleaned up LLM-style language** across all new files — removed "YOUR/your" references, conversational phrases ("we don't need VMs", "We only COUNT distinct nodes") from `fat_tree_wrapper.py`, `problem_top1.py`, `state_top1.py`, `generate_data.py`, `state_top1.py`

3. **Updated README.md** — added "AI coding assistant context" section at the bottom pointing to this file

4. **Completed 6-epoch training run** (`top_ft_k4_20260530T163640`, `epoch_size=10240`, `min_visits=3`)

### Training results (6-epoch run)

| Epoch | Val avg cost | Baseline updated? |
|-------|-------------|-------------------|
| 0 | 42.3 | Yes (47 → 36) |
| 1 | 35.7 | No |
| 2 | 41.8 | Yes (29.7 vs 36.1) |
| 3 | 30.7 | No |
| 4 | 39.9 | No |
| 5 | 31.4 | No (p=0.26) |

Baseline froze at epoch 2 (mean 29.94) and never updated. Optimal is 4–6 hops; model is stuck at ~30.

### Training diagnosis — why it's not converging

**Key signal:** training batch cost (sampled decoding) ~14, validation cost (greedy decoding) ~30–42. Sampled paths are nearly half the cost of greedy paths, which is backwards — greedy should be at least as good as average sampled.

This means the model has learned a **high-entropy policy**: it occasionally samples a short path by random luck (dragging the sampled average down), but the greedy path — always picking the most probable next node — doesn't reliably navigate toward the end. The model can identify where the end node is but doesn't know how to route toward it through the graph.

**Root cause: node features are too weak.** The only features that vary between instances are `is_start` and `is_end`. The other two features (normalized node index, avg outgoing cost) are identical across all 240 instances since the topology never changes. With no distance or relational information, the model can't learn a confident routing policy — it's navigating blind.

This is not a bug. The code is correct. It's a feature design problem.

---

## What Worked

- Adapting the OP state-machine pattern (`NamedTuple`, `initialize`, `update`, `get_mask`, `all_finished`) to TOP worked cleanly
- Custom `_init_embed()` using 4 features compiled and ran correctly
- The rollout baseline and REINFORCE loop required no changes
- `inspect_model.py` visualization confirmed untrained model behavior (oscillation → BFS-forced termination)
- Model did learn something — went from ~47 (pure random oscillation) to ~30 range, showing the pipeline is working

## What Didn't Work / Cautions

- **Node features too weak to learn routing:** only `is_start` and `is_end` vary across instances. Model can't navigate the graph without distance information.
- **Training plateau / oscillating validation:** baseline froze after epoch 2. Grad norm always clipped at 1.0 throughout all 6 epochs — learning rate too high for fine-tuning phase.
- **Sampled vs greedy gap:** sampled ~14, greedy ~30–42 is a clear sign of a high-entropy policy with no confident routing direction.
- **`min_visits=3` adds no constraint:** any valid path satisfies it. This doesn't cause the convergence failure but means the constraint mechanism is untested.
- **`get_mask()` is the training bottleneck:** Python `for b in range(batch_size)` loop, not vectorized. Scales O(total_instances × steps). With `epoch_size=51200` this makes each epoch 30–50 min. Vectorizing would give 10–50× speedup.

---

## Next Steps

### Highest priority — fix convergence

**1. Add BFS distance-to-end as a 5th node feature** (highest impact, do this first)

In `nets/attention_model.py`, inside `_init_embed()` where `self.is_top` is handled, add:

```python
# Feature 5: BFS distance from each node to end node, normalized
# Computed per-instance since end changes. All costs are 1.0 so BFS = shortest hop count.
dist_to_end = torch.full((batch_size, n_nodes), float(n_nodes), device=device)
for b in range(batch_size):
    end = end_idx[b].item()
    cm = cost_matrix[b]
    visited_bfs = {end}
    queue = [end]
    dist_to_end[b, end] = 0.0
    d = 1
    while queue:
        next_q = []
        for node in queue:
            for nb in range(n_nodes):
                if nb not in visited_bfs and not torch.isinf(cm[node, nb]):
                    dist_to_end[b, nb] = d
                    visited_bfs.add(nb)
                    next_q.append(nb)
        queue = next_q
        d += 1
dist_to_end = dist_to_end / n_nodes  # normalize

features = torch.stack([node_indices, avg_costs, start_indicator, end_indicator, dist_to_end], dim=-1)
```

Also update `node_dim = 5` in `__init__` (currently set to 4 for TOP).

This gives the model a routing gradient — it can see "node X is 2 hops from the end, node Y is 5 hops" and learn to prefer shorter paths. Without this, the model is navigating blind.

**2. Adjust learning dynamics**

```powershell
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_ft_dist_feature --epoch_size 10240 --n_epochs 30 `
  --batch_size 128 --eval_batch_size 128 --val_size 1024 `
  --lr_decay 0.95 --bl_alpha 0.10 `
  2>&1 | Tee-Object -FilePath training_log.txt
```

- `--lr_decay 0.95`: lets the model fine-tune after initial fast drop instead of staying at full LR
- `--bl_alpha 0.10`: looser baseline update threshold — epoch 5 nearly updated at p=0.26, stricter than needed given noisy 240-instance signal

### Performance
3. **Vectorize `get_mask()`** in `state_top1.py` — replace `for b in range(batch_size)` with tensor operations. Required before running `epoch_size=51200+` at reasonable speed.

### Problem extension — specific required nodes
4. **Change `min_visits` from count to required set** — where BFS stops being a competitor:
   - `state_top1.py`: replace `min_visits: Tensor (batch_size,)` with `required_nodes: Tensor (batch_size, n_required)`; change `all_finished()` and `get_mask()` from count comparison to set membership check
   - `attention_model.py`: add `is_required` binary node feature — marks nodes that must be visited
   - `fat_tree_wrapper.py`: generate a fixed-size random subset of switches as required nodes per instance
   - `problem_top1.py`: update `get_costs()` to check set membership instead of counting unique nodes
   - CLI: replace `--min_visits` with `--n_required`

### Longer term
5. **Non-uniform costs** — replace unit costs with traffic-load-based weights from the FatTree class
6. **BFS baseline comparison** — implement BFS-with-required-nodes as ground truth for small instances to measure optimality gap

---

## How to Run

**Python env:** `C:\Users\Chris\miniconda3\envs\attention_tsp\python.exe`
**Working dir:** `d:\TOPS-prob\attention-learn-to-route\`

```powershell
# Sanity check
python -c "from fattree.fat_tree_wrapper import is_fat_tree_available; print(is_fat_tree_available())"
# Should print: True

# Minimal training to get a checkpoint for inspection (~30s)
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout --run_name top_inspect --epoch_size 128 --n_epochs 1 --batch_size 64 --eval_batch_size 64 --val_size 128

# Recommended next training run (after adding distance-to-end feature)
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_ft_dist_feature --epoch_size 10240 --n_epochs 30 `
  --batch_size 128 --eval_batch_size 128 --val_size 1024 `
  --lr_decay 0.95 --bl_alpha 0.10 `
  2>&1 | Tee-Object -FilePath training_log.txt

# Full training — only practical after vectorizing get_mask()
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout --run_name top_ft_k4 --epoch_size 51200 --n_epochs 50 --batch_size 128 --eval_batch_size 128 --val_size 1024 2>&1 | Tee-Object -FilePath training_log.txt

# Visualize — static two-panel (BFS optimal vs model)
python inspect_model.py

# Visualize — step-through animation of model walk
python inspect_model.py --animate

# Visualize specific checkpoint with specific seed
python inspect_model.py --checkpoint outputs\top_22\<run_dir> --seed 7 --animate
```
