import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTop(NamedTuple):
    """
    State for the n-stroll (k-stroll) problem.
    
    The agent must traverse from start_idx to end_idx, visiting at least
    min_visits distinct nodes, while minimizing total path cost.
    
    Revisited nodes do not count twice toward min_visits.
    """
    # Fixed input
    cost_matrix: torch.Tensor  # (batch_size, n_nodes, n_nodes)
    start_idx: torch.Tensor    # (batch_size,)
    end_idx: torch.Tensor      # (batch_size,)
    min_visits: torch.Tensor   # (batch_size,)

    # For beam search: index back to original data
    ids: torch.Tensor  # (batch_size, 1) or (batch_size * beam_size, 1)

    # State
    first_a: torch.Tensor      # First action (start node)
    prev_a: torch.Tensor       # Previous action (current node)
    visited_: torch.Tensor     # Tracks which nodes have been visited (for counting distinct)
    lengths: torch.Tensor      # Current path cost
    n_visited: torch.Tensor    # Count of distinct nodes visited
    i: torch.Tensor            # Step counter

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.cost_matrix.size(-1))

    @property
    def n_nodes(self):
        return self.cost_matrix.size(-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            n_visited=self.n_visited[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        """
        Initialize state for n-stroll problem.
        Agent starts at start_idx with that node marked as visited.
        """
        cost_matrix = input['cost_matrix']
        start_idx = input['start_idx']
        end_idx = input['end_idx']
        min_visits = input['min_visits']

        batch_size, n_nodes, _ = cost_matrix.size()
        device = cost_matrix.device
        
        # Initialize visited mask - shape (batch_size, 1, n_nodes)
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(batch_size, 1, n_nodes, dtype=torch.uint8, device=device)
        else:
            visited_ = torch.zeros(batch_size, 1, (n_nodes + 63) // 64, dtype=torch.int64, device=device)
        
        # Mark start node as visited
        if visited_dtype == torch.uint8:
            start_idx_expanded = start_idx[:, None, None]  # (batch_size, 1, 1)
            visited_ = visited_.scatter(-1, start_idx_expanded, 1)
        else:
            visited_ = mask_long_scatter(visited_, start_idx[:, None])
        
        return StateTop(
            cost_matrix=cost_matrix,
            start_idx=start_idx,
            end_idx=end_idx,
            min_visits=min_visits,
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            first_a=start_idx[:, None],
            prev_a=start_idx[:, None],
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=device),
            n_visited=torch.ones(batch_size, 1, dtype=torch.long, device=device),
            i=torch.zeros(1, dtype=torch.int64, device=device)
        )

    def get_final_cost(self):
        """Return the final path cost."""
        assert self.all_finished()
        return self.lengths

    def update(self, selected):
        """
        Update state after selecting a node to visit.
        """
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        selected = selected[:, None]  # (batch_size, 1)
        
        # Get indices for gathering from cost matrix
        batch_indices = self.ids.squeeze(-1)
        from_nodes = self.prev_a.squeeze(-1)
        to_nodes = selected.squeeze(-1)
        
        # Look up edge cost - handle potential inf values
        edge_cost = self.cost_matrix[batch_indices, from_nodes, to_nodes]
        edge_cost = torch.where(torch.isinf(edge_cost), 
                                torch.full_like(edge_cost, 1000.0), 
                                edge_cost)
        
        # Update path cost
        lengths = self.lengths + edge_cost[:, None]

        # Check if node was already visited
        if self.visited_.dtype == torch.uint8:
            selected_expanded = selected[:, :, None]  # (batch_size, 1, 1)
            was_visited = self.visited_.gather(-1, selected_expanded).squeeze(-1)  # (batch_size, 1)
            # Update visited mask
            visited_ = self.visited_.scatter(-1, selected_expanded, 1)
        else:
            was_visited = mask_long2bool(self.visited_, n=self.n_nodes).gather(-1, selected[:, :, None]).squeeze(-1)
            visited_ = mask_long_scatter(self.visited_, selected)

        # Increment visit count only if this is a NEW node
        n_visited = self.n_visited + (1 - was_visited.long())

        return self._replace(
            prev_a=selected,
            visited_=visited_,
            lengths=lengths,
            n_visited=n_visited,
            i=self.i + 1
        )

    def all_finished(self):
        """
        Check if all instances have finished.
        Finished = at end node AND visited >= min_visits nodes.
        """
        batch_indices = self.ids.squeeze(-1)
        
        at_end = (self.prev_a.squeeze(-1) == self.end_idx[batch_indices])
        
        if not at_end.all():
            return False
        
        min_visits_met = (self.n_visited.squeeze(-1) >= self.min_visits[batch_indices])
        return min_visits_met.all()

    def get_current_node(self):
        """Returns the current node."""
        return self.prev_a
    
    def get_mask(self):
        """
        Get mask of infeasible actions (True = infeasible).
        
        MASKING STRATEGY:
        - Always mask: self-loops, unreachable nodes (inf cost)
        - Before min_visits: PREFER unvisited, but ALLOW visited if no unvisited reachable
        - Before min_visits: Always mask end node
        - After min_visits: Allow everything reachable (including end)
        - Finished: Mask everything except end node
        
        This allows BACKTRACKING when needed while still encouraging new visits.
        """
        batch_size = self.ids.size(0)
        n_nodes = self.n_nodes
        device = self.cost_matrix.device
        batch_indices = self.ids.squeeze(-1)
        
        current_nodes = self.prev_a.squeeze(-1)  # (batch_size,)
        end_nodes = self.end_idx[batch_indices]  # (batch_size,)
        n_visited_flat = self.n_visited.squeeze(-1)  # (batch_size,)
        min_visits_flat = self.min_visits[batch_indices]  # (batch_size,)
        
        # Check conditions
        enough_visits = n_visited_flat >= min_visits_flat
        at_end = (current_nodes == end_nodes)
        finished = at_end & enough_visits
        
        # === Step 1: Get reachable nodes (finite cost edges) ===
        edges_from_current = self.cost_matrix[batch_indices, current_nodes, :]  # (batch_size, n_nodes)
        reachable = ~torch.isinf(edges_from_current)  # (batch_size, n_nodes)
        
        # === Step 2: Get visited mask ===
        if self.visited_.dtype == torch.uint8:
            visited_mask = self.visited_.squeeze(1).bool()  # (batch_size, n_nodes)
        else:
            visited_mask = mask_long2bool(self.visited_, n=n_nodes).squeeze(1)
        
        # === Step 3: Build mask ===
        # Start with: mask unreachable nodes
        mask = ~reachable  # True where NOT reachable
        
        # Always mask self-loops
        mask[torch.arange(batch_size, device=device), current_nodes] = True
        
        # === Step 4: Handle each instance based on state ===
        for b in range(batch_size):
            if finished[b]:
                # FINISHED: mask everything except end node (to prevent NaN)
                mask[b] = True
                mask[b, end_nodes[b]] = False
                
            elif enough_visits[b]:
                # ENOUGH VISITS: can go anywhere reachable (including end)
                # mask already set to ~reachable, which is correct
                reachable_b = reachable[b].clone()
                reachable_b[current_nodes[b]] = False  # No self-loop
                mask[b] = ~reachable_b

            else:
                # NOT ENOUGH VISITS: need to visit more nodes
                # Check if any unvisited nodes are reachable
                unvisited_reachable = reachable[b] & ~visited_mask[b]
                unvisited_reachable[current_nodes[b]] = False  # Exclude self
                unvisited_reachable[end_nodes[b]] = False       # Can't go to end yet
                
                if unvisited_reachable.any():
                    # There ARE unvisited reachable nodes → only allow those
                    mask[b] = ~unvisited_reachable
                else:
                    # NO unvisited reachable nodes → must BACKTRACK
                    # Allow any reachable node except self and end
                    mask[b] = ~reachable[b]
                    mask[b, current_nodes[b]] = True  # No self-loop
                    mask[b, end_nodes[b]] = True       # Can't end yet
        
        # === Step 5: Safety - ensure at least one node available ===
        all_masked = mask.all(dim=-1)
        stuck = all_masked & ~finished
        
        if stuck.any():
            for b in range(batch_size):
                if stuck[b]:
                    # Emergency: allow any reachable node
                    reachable_b = reachable[b].clone()
                    reachable_b[current_nodes[b]] = False
                    
                    if enough_visits[b] and reachable_b[end_nodes[b]]:
                        mask[b, end_nodes[b]] = False
                    elif reachable_b.any():
                        first_reachable = reachable_b.nonzero(as_tuple=True)[0][0]
                        mask[b, first_reachable] = False
                    else:
                        # Completely stuck - allow end even if unreachable
                        mask[b, end_nodes[b]] = False
        
        # === Step 6: Max steps safety - use BFS to force path to end ===
        max_steps = 2 * n_nodes
        if self.i.item() >= max_steps:
            for b in range(batch_size):
                if not finished[b]:
                    current_node = current_nodes[b].item()
                    end_node = end_nodes[b].item()
                    cost_matrix_b = self.cost_matrix[batch_indices[b]]
                    
                    mask[b] = True  # Mask everything
                    
                    # Use BFS to find next hop toward end
                    next_hop = self._bfs_next_hop(cost_matrix_b, current_node, end_node, n_nodes)
                    
                    if next_hop is not None:
                        mask[b, next_hop] = False
                    else:
                        # No path - allow any reachable as fallback
                        reachable_b = reachable[b].clone()
                        reachable_b[current_node] = False
                        if reachable_b.any():
                            first_reachable = reachable_b.nonzero(as_tuple=True)[0][0]
                            mask[b, first_reachable] = False
                        else:
                            mask[b, end_node] = False
        
        return mask.unsqueeze(1)  # (batch_size, 1, n_nodes)
    
    def _bfs_next_hop(self, cost_matrix, start, end, n_nodes):
        """
        BFS to find the next hop from start toward end.
        Returns the first node on shortest path to end, or None if no path.
        """
        if start == end:
            return None
        
        # Check if directly reachable
        if not torch.isinf(cost_matrix[start, end]):
            return end
        
        # BFS
        from collections import deque
        visited = {start}
        queue = deque()
        
        # Add all neighbors of start
        for neighbor in range(n_nodes):
            if neighbor != start and not torch.isinf(cost_matrix[start, neighbor]):
                if neighbor == end:
                    return neighbor
                queue.append((neighbor, neighbor))  # (current, first_hop)
                visited.add(neighbor)
        
        while queue:
            current, first_hop = queue.popleft()
            for neighbor in range(n_nodes):
                if neighbor not in visited and not torch.isinf(cost_matrix[current, neighbor]):
                    if neighbor == end:
                        return first_hop
                    queue.append((neighbor, first_hop))
                    visited.add(neighbor)
        
        return None

    def construct_solutions(self, actions):
        """Construct the solution path from actions."""
        return actions