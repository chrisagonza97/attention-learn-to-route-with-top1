"""
Diagnose:
1. What are actual edge costs in the fat-tree cost matrix?
2. What is BFS shortest path cost (no constraint)?
3. What is shortest path visiting >= min_visits nodes?
4. Why is avg_cost ~47 for an untrained model?
"""
import torch
from collections import deque
from fattree.fat_tree_wrapper import generate_fat_tree_instance


def bfs_shortest_path_cost(cost_matrix, start, end, n_nodes):
    """Returns (cost, path) for shortest hop path start->end ignoring min_visits."""
    if start == end:
        return 0, [start]
    dist = {start: 0}
    prev = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in range(n_nodes):
            if nbr not in dist and not torch.isinf(cost_matrix[node, nbr]) and cost_matrix[node, nbr] > 0:
                dist[nbr] = dist[node] + cost_matrix[node, nbr].item()
                prev[nbr] = node
                if nbr == end:
                    # Reconstruct
                    path = []
                    cur = end
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return dist[end], list(reversed(path))
                queue.append(nbr)
    return float('inf'), []


def bfs_min_visits_path(cost_matrix, start, end, n_nodes, min_visits):
    """
    BFS over state (node, frozenset_visited) to find shortest path
    from start to end visiting >= min_visits distinct nodes.
    Only feasible for small graphs.
    """
    # State: (current_node, visited_frozenset)
    # Too expensive for full state space on 22 nodes if min_visits is large.
    # Use a bounded search: track best cost per (node, n_distinct_visited).
    INF = float('inf')
    # dist[(node, n_visited_count)] = best cost seen
    # This is an approximation — doesn't track which nodes visited, only count
    # But for min_visits=3 on 22 nodes it's exact enough

    init_visited = frozenset([start])
    # (cost, node, visited_set)
    import heapq
    heap = [(0.0, start, init_visited)]
    best = {}  # (node, frozenset) -> cost
    best[(start, init_visited)] = 0.0

    while heap:
        cost, node, visited = heapq.heappop(heap)

        key = (node, visited)
        if best.get(key, INF) < cost:
            continue

        if node == end and len(visited) >= min_visits:
            # Reconstruct not tracked, just return cost
            return cost

        for nbr in range(n_nodes):
            if not torch.isinf(cost_matrix[node, nbr]) and cost_matrix[node, nbr] > 0:
                new_cost = cost + cost_matrix[node, nbr].item()
                new_visited = visited | frozenset([nbr])
                new_key = (nbr, new_visited)
                if new_cost < best.get(new_key, INF):
                    best[new_key] = new_cost
                    heapq.heappush(heap, (new_cost, nbr, new_visited))

    return INF


def analyze_instance(inst, label=""):
    cm = inst['cost_matrix']
    start = inst['start_idx'].item()
    end = inst['end_idx'].item()
    min_v = inst['min_visits'].item()
    n = cm.size(0)

    # Edge stats
    finite = cm[(~torch.isinf(cm)) & (cm > 0)]
    print(f"\n{'='*50}")
    if label:
        print(f"Instance: {label}")
    print(f"  Nodes: {n}, start={start}, end={end}, min_visits={min_v}")
    print(f"  Edge costs — min:{finite.min().item():.2f}  max:{finite.max().item():.2f}  "
          f"mean:{finite.mean().item():.2f}  count:{len(finite)}")

    # BFS shortest (no constraint)
    cost_unconstrained, path = bfs_shortest_path_cost(cm, start, end, n)
    print(f"  BFS shortest (no constraint): cost={cost_unconstrained:.1f}, hops={len(path)-1}, path={path}")
    print(f"    Distinct nodes on shortest path: {len(set(path))}")

    # Shortest with min_visits constraint (only if graph small enough)
    if n <= 22:
        cost_constrained = bfs_min_visits_path(cm, start, end, n, min_v)
        print(f"  BFS shortest (>={min_v} visits):   cost={cost_constrained:.1f}")
        if cost_constrained < float('inf'):
            overhead = cost_constrained - cost_unconstrained
            print(f"    Detour overhead vs unconstrained: +{overhead:.1f} hops")
    else:
        print(f"  (Skipping constrained BFS — graph too large for exhaustive search)")

    # Degree stats
    out_degrees = [(~torch.isinf(cm[i]) & (cm[i] > 0)).sum().item() for i in range(n)]
    print(f"  Node out-degrees — min:{min(out_degrees)}  max:{max(out_degrees)}  "
          f"avg:{sum(out_degrees)/len(out_degrees):.1f}")
    print(f"  (Untrained model avg_cost ~47 = wandering ~47 steps × cost 1.0 per step)")


if __name__ == "__main__":
    torch.manual_seed(42)
    print("Generating 5 fat-tree k=4 instances...")
    for i in range(5):
        inst = generate_fat_tree_instance(k=4, min_visits_ratio=0.3)
        analyze_instance(inst, label=f"#{i+1}")
