# N-Stroll Problem — Fat-Tree VM Migration Routing

This document covers the N-Stroll extension added to the [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) codebase. The original repo trains attention-based models for TSP, VRP, OP, and PCTSP — those are untouched. Everything described here is new code.

---

## Background and motivation

In a data center, virtual machine (VM) migration from server A to server B isn't just point-to-point routing. Traffic may need to traverse a minimum number of intermediate switches — for example, to pass through virtual network functions (VNFs) like firewalls, load balancers, or traffic monitors in a service-function chain.

The routing problem becomes: **find the shortest walk from A to B that visits at least K distinct nodes**, where K is the service-function chaining requirement.

This is structurally the inverse of the Orienteering Problem (OP): OP maximizes prizes within a budget; N-Stroll minimizes path cost subject to a minimum-visit constraint.

---

## Problem definition

**Given:**
- A graph with a cost matrix (edge weights between nodes)
- Start node `s` (source server / PM)
- End node `t` (destination server / PM)
- `min_visits`: minimum number of distinct nodes the walk must include

**Find:** a walk from `s` to `t` that visits at least `min_visits` distinct nodes, minimizing total edge cost.

**Key rules:**
- Backtracking is allowed — the walk can revisit nodes
- Only *distinct* nodes count toward `min_visits`; revisits don't count twice
- The walk terminates the moment it reaches `t` with `min_visits` satisfied

---

## Fat-tree topology

The training environment is a **k=4 fat-tree** data center network — a standard 3-tier topology used in production data centers.

```
Tier 3 (Core):        C0  C1  C2  C3          ← 4 switches,  nodes 0–3
Tier 2 (Aggregation): A0 A1 A2 A3 A4 A5 A6 A7  ← 8 switches,  nodes 4–11
Tier 1 (Edge):        E0 E1 E2 E3 E4 E5 E6 E7  ← 8 switches,  nodes 12–19
Physical machines:    Start PM (node 20)         ← source server
                      End PM   (node 21)         ← destination server
```

Total: **22 nodes**. All edge costs are **1.0** (hop count). The fat-tree class itself (`fattree/fat_tree.py`) provides the topology; `fattree/fat_tree_wrapper.py` converts it to a cost matrix.

**Key topology facts for k=4:**
- 16 physical machines (PMs), giving 240 distinct (start, end) pairs
- Shortest path between any two PMs: 2 hops (same edge switch) to 6 hops (cross-pod)
- The fat-tree `distance()` function directly computes shortest hop distance between any two nodes

### Node indexing convention

Switches are nodes 0–19 (matching original fat-tree IDs). The two PMs for a given instance are always remapped to nodes 20 (start) and 21 (end). This means the cost matrix is always 22×22, and `start_idx=20`, `end_idx=21` for every instance — instance variation comes entirely from which edge switches nodes 20 and 21 connect to.

---

## Why a neural network instead of BFS?

For the current setup (unit costs, "any K nodes" constraint), BFS finds the optimal path instantly. The neural network pays off when the problem gets harder:

| Scenario | BFS / Dijkstra | Learned model |
|----------|---------------|---------------|
| Unit costs, any K nodes | Optimal, instant | Overkill |
| Non-uniform costs | Dijkstra handles it | Can generalize across cost configurations |
| **Specific required nodes** | Exponential state space | Generalizes across required-node sets |
| Dynamic/changing costs | Must recompute each time | Policy adapts at inference time |

The planned next step is switching from "visit any K nodes" to "visit this specific set of nodes" (e.g. nodes {5, 12, 17} must appear on the path). This is a Steiner path variant — NP-hard in general — and is where the neural approach has a genuine advantage.

---

## Code structure

### New files

| File | Purpose |
|------|---------|
| `problems/top1/problem_top1.py` | `TOP` class: `get_costs()`, `make_dataset()`, `make_state()`. Instance generator for fat-tree and other graph types. |
| `problems/top1/state_top1.py` | `StateTop` NamedTuple: decoder state machine. Tracks current node, visited set, step count, path cost. Implements `initialize()`, `update()`, `get_mask()`, `all_finished()`. |
| `fattree/fat_tree_wrapper.py` | Wraps the `FatTree` class to produce cost matrices. Caches the wrapper instance to avoid rebuilding the topology on every call. |
| `fattree/vm_pair.py` | Stub — required for `FatTree` import to succeed. |
| `fattree/sized_vm_pair.py` | Stub — required for `FatTree` import to succeed. |
| `fattree/ac_migrate_pytorch.py` | Stub — required for `FatTree` import to succeed. |
| `inspect_model.py` (repo root) | Visualization tool — loads a checkpoint, generates an instance, shows BFS optimal vs model path. |

### Modified files

| File | What changed |
|------|-------------|
| `fattree/fat_tree.py` | Bare imports changed to relative imports; `pulp` import guarded with try/except |
| `nets/attention_model.py` | Added `is_top` flag; custom `_init_embed()` builds node features from cost matrix instead of coordinates |
| `problems/__init__.py` | Added `from problems.top1.problem_top1 import TOP` |
| `options.py` | Added `--fat_tree_k` and `--min_visits` CLI arguments |
| `generate_data.py` | Added TOP data generation support |
| `utils/functions.py` | Registered `'top': TOP` in problem registry |

---

## Node features

The attention model's encoder takes a feature vector per node. For TOP, `_init_embed()` in `attention_model.py` builds 4 features from the cost matrix:

1. **Normalized node index** — position in 0–1 range
2. **Normalized average outgoing cost** — mean edge weight to reachable neighbors, normalized
3. **Start indicator** — 1.0 if this node is the start PM, 0.0 otherwise
4. **End indicator** — 1.0 if this node is the end PM, 0.0 otherwise

These replace the (x, y) coordinates used for TSP/VRP.

---

## Masking strategy

At each decoding step, `get_mask()` returns a binary mask of infeasible next nodes. The logic:

- **Always masked:** self-loops, nodes with no direct edge (inf cost)
- **Before `min_visits` reached:** end node is masked; only unvisited reachable nodes are allowed. If no unvisited nodes are reachable, backtracking to visited nodes is permitted (but end node stays masked)
- **After `min_visits` reached:** all reachable nodes are allowed, including the end node
- **Finished (at end node with enough visits):** everything masked except end node (prevents NaN)
- **Max steps safety (step ≥ 2×n_nodes):** BFS forces the next hop toward the end node, overriding all other logic. This guarantees termination even for an untrained/stuck model.

---

## Installation

`requirements.txt` in the repo root is a full conda environment export. To recreate it:

```bash
conda create --name attention_tsp --file requirements.txt
conda activate attention_tsp
```

If that fails due to platform differences, install the core packages manually:

```bash
conda create -n attention_tsp python=3.10
conda activate attention_tsp
pip install torch torchvision tqdm scipy numpy matplotlib networkx tensorboard-logger PyQt5
```

Verify the environment:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from fattree.fat_tree_wrapper import is_fat_tree_available; print('FatTree:', is_fat_tree_available())"
# Both should print True
```

---

## Training

All commands assume the working directory is the repo root and the `attention_tsp` conda environment is active.

### Quick sanity check (~30 seconds)

Confirms the pipeline runs end-to-end without errors:

```powershell
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_test --epoch_size 128 --n_epochs 1 `
  --batch_size 64 --eval_batch_size 64 --val_size 128
```

Expected output includes `Successfully imported FatTree class` and no NaN costs. Initial `avg_cost` will be ~47 (untrained model oscillates randomly).

### Real training (~45 min, demonstrable learning)

```powershell
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_ft_k4 --epoch_size 10240 --n_epochs 6 `
  --batch_size 128 --eval_batch_size 128 --val_size 1024 `
  2>&1 | Tee-Object -FilePath training_log.txt
```

### Reading the training log

| Line | What it means |
|------|--------------|
| `Successfully imported FatTree class` | Real fat-tree topology in use |
| `avg_cost: ~47` (epoch 0) | Normal — untrained model wanders for 44 steps |
| `Validation overall avg_cost: XX` | End-of-epoch metric. Target: fall toward 4–6 (optimal range) |
| `Update baseline` | Model beat the rollout baseline — learning is happening |
| `grad_norm: 493, clipped: 1.0` | High early on, expected. Should fall as training progresses |
| `difference 0.0` | Model tied the baseline — no improvement this epoch |

### Checkpoints and TensorBoard

- Checkpoints: `outputs/top_22/<run_name>/epoch-N.pt`
- TensorBoard logs: `logs/top_22/<run_name>/`

```powershell
tensorboard --logdir logs
# Open http://localhost:6006
```

Key metrics to watch in TensorBoard: `val_avg_reward` (validation cost per epoch), `avg_cost` (training batch cost), `grad_norm`.

### Resuming a stopped run

```powershell
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_ft_k4_v2 `
  --resume outputs\top_22\top_ft_k4_<timestamp>\epoch-5.pt
```

---

## Inspecting a trained model

`inspect_model.py` loads a checkpoint, generates a random fat-tree instance, runs the model greedily, and visualizes the result against the BFS-optimal path.

### Static two-panel view

```powershell
python inspect_model.py                            # latest checkpoint, random instance
python inspect_model.py --checkpoint outputs\top_22\<run_name>
python inspect_model.py --seed 7                   # specific random instance
python inspect_model.py --n 3                      # show 3 instances in sequence
python inspect_model.py --save result.png          # save to file
```

### Animated step-through

```powershell
python inspect_model.py --animate
```

Shows the model's walk step by step. Left panel: BFS optimal (static reference). Right panel: model walk building incrementally, current position marked with a star. Use Prev / Play / Next buttons.

### Reading the visualization

- **Green edges** — traversed once (forward progress)
- **Red edges, xN label** — traversed N times (oscillation / backtracking)
- **S** (red, node 20) — start PM
- **T** (purple, node 21) — end PM
- **C0–C3** (blue) — core switches
- **A0–A7** (orange) — aggregation switches
- **E0–E7** (green) — edge switches

**Untrained model:** oscillates between 2 neighbors for 44 steps, then BFS safety forces it to the end. Cost ~47, gap ~700% above optimal.  
**Trained model:** should show a direct 4–6 hop path, no oscillation.

### Comparing two checkpoints

Use the same `--seed` so you're looking at the same instance:

```powershell
python inspect_model.py --checkpoint outputs\top_22\top_test_<timestamp> --seed 5
python inspect_model.py --checkpoint outputs\top_22\top_ft_k4_<timestamp> --seed 5
```

---

## Known limitations

### `get_mask()` is slow (not vectorized)

The masking code has a Python `for b in range(batch_size)` loop that runs at every decoding step. This is the training bottleneck — it scales linearly with total instances processed, making large epoch sizes slow (~30–50 min/epoch at epoch_size=51200). Vectorizing this loop with tensor operations would give a 10–50× speedup and is the most impactful engineering task before scaling up experiments.

### `min_visits=3` adds no constraint

Any valid path on a k=4 fat-tree visits at least 3 nodes (PM → at least 1 switch → PM), so `--min_visits 3` is always trivially satisfied. The constraint only becomes meaningful at higher values. Use `--min_visits 8` or omit `--min_visits` to use the default (0.3 × 20 switches = 8) to actually exercise the masking logic.

### Only 240 distinct problem instances

With 16 PMs and k=4, there are only 240 unique (start, end) pairs. The topology is fixed. This means the model is essentially memorizing 240 solutions rather than generalizing. This is fine as a proof-of-concept but limits what can be concluded about generalization.

---

## Training results and open problem

A 6-epoch run (`epoch_size=10240`, `min_visits=3`) showed the model learning something — validation cost fell from ~47 (untrained) to the ~30 range — but did not converge. Optimal is 4–6 hops; the model plateaued well above that.

| Epoch | Val avg cost | Baseline updated? |
|-------|-------------|-------------------|
| 0 | 42.3 | Yes |
| 1 | 35.7 | No |
| 2 | 41.8 | Yes |
| 3 | 30.7 | No |
| 4 | 39.9 | No |
| 5 | 31.4 | No |

The baseline froze after epoch 2 and validation oscillated rather than trending down. The root cause is identifiable from a key signal: **training batch cost (sampled decoding) settled at ~14, while validation cost (greedy decoding) stayed at ~30–42**. Sampled paths are nearly half the cost of greedy paths, which is backwards — greedy should be at least as good as average sampled.

This means the model learned a high-entropy policy that occasionally samples a short path by chance but has no confident greedy direction. The model can identify the end node but can't navigate toward it through the graph.

**Root cause: node features don't encode routing information.** The four current features are: normalized node index, avg outgoing cost, `is_start`, `is_end`. The first two are identical across all 240 instances since the topology never changes. Only `is_start` and `is_end` vary — they mark the source and destination, but give the model no signal about which direction to go through the intermediate switches.

This is not a bug in the code. The pipeline is correct. It is a feature design limitation.

---

## Planned next steps

### 1. Add distance-to-end node feature (highest priority for convergence)

Add BFS hop distance from each node to the end node as a 5th feature in `_init_embed()` in `nets/attention_model.py`. Since all edge costs are 1.0, BFS distance is well-defined and cheap to compute. This gives the model a routing gradient — it can see which nodes are closer to the destination and learn to prefer them.

Also update `node_dim = 5` (currently 4) in `AttentionModel.__init__` for the TOP branch, and retrain with learning rate decay and a looser baseline threshold:

```powershell
python run.py --problem top --fat_tree_k 4 --min_visits 3 --baseline rollout `
  --run_name top_ft_dist_feature --epoch_size 10240 --n_epochs 30 `
  --batch_size 128 --eval_batch_size 128 --val_size 1024 `
  --lr_decay 0.95 --bl_alpha 0.10
```

### 2. Specific required nodes

Change the constraint from "visit any K nodes" to "visit this specific set of nodes". For example: nodes {5, 12, 17} must appear on the path from node 20 to node 21.

Files to change:
- **`state_top1.py`**: Replace `min_visits: Tensor (batch_size,)` with `required_nodes: Tensor (batch_size, n_required)`. Change `all_finished()` and `get_mask()` from count comparison to set membership check.
- **`attention_model.py`**: Add an `is_required` binary node feature (1.0 for required nodes, 0.0 otherwise).
- **`fat_tree_wrapper.py`**: Generate a random subset of switches as required nodes per instance.
- **`problem_top1.py`**: Update `get_costs()` to check set membership instead of counting unique nodes.

### 3. Non-uniform edge costs

Replace unit costs with traffic-load-based weights from the FatTree class, making the problem more realistic and harder for classical algorithms.

---

## Diagnostic scripts (repo root)

| Script | Purpose |
|--------|---------|
| `diagnose_costs2.py` | Shows edge cost distribution, BFS shortest paths, and constrained-optimal costs for a few instances |
| `debug.py`, `debug_state.py`, `debug_batch26.py` | Development debugging scripts, can be ignored |
| `check_model_paths.py` | Verifies checkpoint paths load correctly |

---

## AI coding assistant context

[HANDOFF.md](HANDOFF.md) contains a denser version of this document written for use with AI coding tools (Claude, Copilot, etc.). It covers current training state, what has and hasn't worked, and the specific file changes needed for the planned next steps. Load it as context when starting a new session.
