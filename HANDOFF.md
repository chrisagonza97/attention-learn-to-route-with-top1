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
| `problems/top1/top1_readme.md` | Documentation |
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
| `fattree/fat_tree_wrapper.py` | **Node remapping updated** — switches at 0–19, start PM at 20, end PM at 21 (matches fat-tree convention) |
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

### Fixes applied this session

1. **Duplicate `_bfs_next_hop`** in `state_top1.py` — removed the shadowed first definition
2. **Debug `print` statements** in `get_mask()` — removed all stdout spam
3. **Node remapping** in `fat_tree_wrapper.py` — start PM moved from index 0 to index 20 to match fat-tree convention (switches 0–19, PMs 20–21). All old checkpoints and logs were wiped (they used the old layout and are invalid).

### Training runs

**Run `top_ft_k4` — currently in progress (started this session)**
- Command: `epoch_size=10240, n_epochs=6, batch_size=128, val_size=1024, min_visits=3`
- Expected wall time: ~45 min
- Check `training_log.txt` for results

**Previous run `top_ft_mv10` (prior session, now deleted — used old node layout)**
- 10 epochs, epoch_size=5120: val cost dropped 46→42, baseline froze after epoch 2 at 28.38
- Not reproducible (old remapping) — treat as directional evidence only

**Weak baseline model** exists at `outputs/top_22/top_inspect_<timestamp>/` — 1 epoch of 128 instances, useful only as a "random walk" reference for `inspect_model.py` comparison.

---

## What Worked

- Adapting the OP state-machine pattern (`NamedTuple`, `initialize`, `update`, `get_mask`, `all_finished`) to TOP worked cleanly
- Custom `_init_embed()` using 4 features (normalized node index, avg outgoing cost, start indicator, end indicator) compiled and ran correctly
- The rollout baseline and REINFORCE loop required no changes
- `inspect_model.py` visualization confirmed untrained model behavior (oscillation → BFS-forced termination)

## What Didn't Work / Cautions

- **Training plateau:** Previous run froze at epoch 2 — likely needs more epochs/data, possibly better node features
- **`min_visits=3` adds no constraint:** any valid path satisfies it. For testing the actual constraint mechanism, use `--min_visits 8` or let the default ratio (0.3 → 8) apply
- **get_mask() is the training bottleneck:** Python `for b in range(batch_size)` loop, not vectorized. Scales O(total_instances × steps). With epoch_size=51200 this makes each epoch 30–50 min. **Vectorizing this loop would give 10–50× speedup** and should be done before running large experiments.

---

## Next Steps

### Immediate
1. **Check `training_log.txt`** — look for `Update baseline` and validation cost dropping epoch over epoch
2. **Run `inspect_model.py`** after training to compare against weak baseline:
   ```powershell
   # Trained model (auto-picks latest)
   python inspect_model.py --animate

   # Weak baseline for comparison
   python inspect_model.py --checkpoint outputs\top_22\top_inspect_<timestamp>
   ```
   Use the same `--seed N` for both to compare identical instances.

### Performance
3. **Vectorize `get_mask()`** in `state_top1.py` — replace `for b in range(batch_size)` with tensor operations. Required before running epoch_size=51200+ experiments at reasonable speed.

### Problem extension — specific required nodes
4. **Change `min_visits` from count to required set** — this is where BFS stops being a competitor:
   - `state_top1.py`: replace `min_visits: Tensor (batch_size,)` with `required_nodes: Tensor (batch_size, n_required)` (fixed count); change `all_finished()` and `get_mask()` from count comparison to set membership check
   - `attention_model.py`: add 5th node feature `is_required` (binary) — marks nodes that must be visited
   - `fat_tree_wrapper.py`: generate a fixed-size random subset of switches as required nodes instead of a count
   - `problem_top1.py`: update `get_costs()` to check set membership instead of counting unique nodes
   - CLI: replace `--min_visits` with `--n_required` (number of required nodes to randomly pick per instance)

### Longer term
5. **Non-uniform costs** — replace unit costs with traffic-load-based weights from the FatTree class
6. **Richer node features** — node degree, min/max outgoing cost, BFS distance to start/end
7. **BFS baseline comparison** — implement BFS-with-required-nodes (exponential state space) as ground truth for small instances to measure optimality gap

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

# Real training (~45 min, 6 epochs)
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout --run_name top_ft_k4 --epoch_size 10240 --n_epochs 6 --batch_size 128 --eval_batch_size 128 --val_size 1024 2>&1 | Tee-Object -FilePath training_log.txt

# Full training — only practical after vectorizing get_mask()
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout --run_name top_ft_k4 --epoch_size 51200 --n_epochs 50 --batch_size 128 --eval_batch_size 128 --val_size 1024 2>&1 | Tee-Object -FilePath training_log.txt

# Visualize — static two-panel (BFS optimal vs model)
python inspect_model.py

# Visualize — step-through animation of model walk
python inspect_model.py --animate

# Visualize specific checkpoint with specific seed
python inspect_model.py --checkpoint outputs\top_22\<run_dir> --seed 7 --animate
```
