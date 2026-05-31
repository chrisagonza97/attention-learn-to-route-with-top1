"""
Visualize a fat-tree instance with optimal BFS path and model's decoded path.

Usage:
    python inspect_model.py                          # latest checkpoint, 1 random instance
    python inspect_model.py --checkpoint outputs/top_22/top_inspect
    python inspect_model.py --seed 7                 # different instance
    python inspect_model.py --n 3                    # show 3 instances in sequence
    python inspect_model.py --animate                # step-through animation of model walk
    python inspect_model.py --save out.png           # save to file instead of showing window
"""
import argparse
import os
from collections import deque, Counter

import torch
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from utils.functions import load_model
from fattree.fat_tree_wrapper import generate_fat_tree_instance


# ---------------------------------------------------------------------------
# Fat-tree layout helpers
# ---------------------------------------------------------------------------

def node_layer(idx, k=4):
    nc = (k*k)//4; na = (k*k)//2; ne = (k*k)//2; ns = nc+na+ne
    if idx < nc:            return 'core',  f'C{idx}',           '#3498DB'
    elif idx < nc+na:       return 'agg',   f'A{idx-nc}',        '#E67E22'
    elif idx < ns:          return 'edge',  f'E{idx-nc-na}',     '#27AE60'
    elif idx == ns:         return 'start', 'S',                  '#C0392B'
    else:                   return 'end',   'T',                  '#8E44AD'


def get_positions(cost_matrix, k=4):
    """Fixed hierarchical positions: core top, agg/edge middle, PMs bottom."""
    nc = (k*k)//4; na = (k*k)//2; ne = (k*k)//2; ns = nc+na+ne
    pos = {}
    for i in range(nc):
        pos[i] = ((i + 0.5) * ne / nc, 3.0)
    for i in range(na):
        pos[nc + i] = (i + 0.5, 2.0)
    for i in range(ne):
        pos[nc + na + i] = (i + 0.5, 1.0)
    # PMs: sit below their edge switch
    for pm in [ns, ns + 1]:
        for j in range(nc + na, ns):
            if not torch.isinf(cost_matrix[pm, j]) and cost_matrix[pm, j] > 0:
                ex, _ = pos[j]
                offset = -0.18 if pm == ns else 0.18
                pos[pm] = (ex + offset, 0.0)
                break
        else:
            pos[pm] = (0.0 if pm == ns else ne, 0.0)
    return pos


def bfs_shortest_path(cm, start, end, n):
    prev = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        for nbr in range(n):
            if nbr not in prev and not torch.isinf(cm[node, nbr]) and cm[node, nbr] > 0:
                prev[nbr] = node
                if nbr == end:
                    path = []
                    cur = end
                    while cur is not None:
                        path.append(cur); cur = prev[cur]
                    return list(reversed(path))
                q.append(nbr)
    return [start]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, cost_matrix, path, pos, title, k=4, current_step=None):
    """Draw one fat-tree panel with a highlighted path."""
    nc = (k*k)//4; na = (k*k)//2; ne = (k*k)//2; ns = nc+na+ne
    n = ns + 2
    ax.clear()

    # Count traversals per undirected edge
    traversals = Counter()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        traversals[(min(u, v), max(u, v))] += 1

    # Background edges
    for i in range(n):
        for j in range(i + 1, n):
            if not torch.isinf(cost_matrix[i, j]) and cost_matrix[i, j] > 0:
                x1, y1 = pos[i]; x2, y2 = pos[j]
                ax.plot([x1, x2], [y1, y2], color='#DEDEDE', lw=0.9, zorder=1)

    # Path edges — green = traversed once, red = oscillation, width ∝ count
    for (u, v), count in traversals.items():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        color = '#2ECC71' if count == 1 else '#E74C3C'
        lw = min(2.5 + count * 1.5, 14)
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2, alpha=0.85)
        if count > 1:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f'x{count}', ha='center', va='bottom', fontsize=7,
                    color='black', fontweight='bold', zorder=5)

    # Nodes
    visited = set(path)
    for idx in range(n):
        x, y = pos[idx]
        _, label, color = node_layer(idx, k)
        in_path = idx in visited
        size = 320 if idx >= ns else (220 if in_path else 100)
        ec = 'black' if in_path else 'none'
        ax.scatter(x, y, s=size, c=color, zorder=3, edgecolors=ec, linewidths=1.5)
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                color='white', fontweight='bold', zorder=4)

    # Current position marker (for animation)
    if current_step is not None and 0 <= current_step < len(path):
        cx, cy = pos[path[current_step]]
        ax.scatter(cx, cy, s=600, c='yellow', zorder=6,
                   edgecolors='black', linewidths=2, marker='*')

    # Layer labels
    for y_val, lbl in [(3.0, 'Core'), (2.0, 'Agg'), (1.0, 'Edge'), (0.0, 'PMs')]:
        ax.text(-0.8, y_val, lbl, ha='right', va='center',
                fontsize=8, color='#999', style='italic')

    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlim(-1.2, ne + 0.5)
    ax.set_ylim(-0.7, 3.6)
    ax.axis('off')


def add_legend(fig):
    patches = [
        mpatches.Patch(color='#3498DB', label='Core switch'),
        mpatches.Patch(color='#E67E22', label='Aggregation switch'),
        mpatches.Patch(color='#27AE60', label='Edge switch'),
        mpatches.Patch(color='#C0392B', label='Start PM (S)'),
        mpatches.Patch(color='#8E44AD', label='End PM (T)'),
        mpatches.Patch(color='#2ECC71', label='Edge traversed once'),
        mpatches.Patch(color='#E74C3C', label='Edge oscillation (xN)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))


# ---------------------------------------------------------------------------
# Static two-panel view
# ---------------------------------------------------------------------------

def show_static(inst, bfs_p, model_p, model_cost, bfs_cost, k=4, save=None):
    cm  = inst['cost_matrix']
    end = inst['end_idx'].item()
    mv    = inst['min_visits'].item()
    pos   = get_positions(cm, k)

    fig, (ax_opt, ax_model) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#F8F9FA')
    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    opt_title  = f'BFS Optimal  |  cost={bfs_cost:.0f}  hops={len(bfs_p)-1}'
    model_title = (f'Model (epoch-{{}})  |  cost={model_cost:.0f}  steps={len(model_p)-1}  '
                   f'distinct={len(set(model_p))}/{mv}  '
                   f'[{"OK" if model_p[-1]==end and len(set(model_p))>=mv else "VIOLATED"}]')

    draw_panel(ax_opt,   cm, bfs_p,   pos, opt_title,   k)
    draw_panel(ax_model, cm, model_p, pos, model_title, k)
    add_legend(fig)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f'Saved to {save}')
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Animated step-through view
# ---------------------------------------------------------------------------

def show_animated(inst, bfs_p, model_p, model_cost, bfs_cost, k=4):
    cm  = inst['cost_matrix']
    end = inst['end_idx'].item()
    mv  = inst['min_visits'].item()
    pos = get_positions(cm, k)
    n_steps = len(model_p)

    state = {'step': 0, 'playing': False}

    fig, (ax_opt, ax_model) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#F8F9FA')
    plt.subplots_adjust(bottom=0.22, wspace=0.05)

    opt_title = f'BFS Optimal  |  cost={bfs_cost:.0f}  hops={len(bfs_p)-1}'
    draw_panel(ax_opt, cm, bfs_p, pos, opt_title, k)

    def update_model(step):
        visible_path = model_p[:step + 1]
        cumcost = step  # all edges cost 1.0
        distinct = len(set(visible_path))
        constraint = 'OK' if visible_path[-1] == end and distinct >= mv else '...'
        title = (f'Model walk  |  step {step}/{n_steps-1}  '
                 f'cost={cumcost}  distinct={distinct}/{mv}  [{constraint}]')
        draw_panel(ax_model, cm, visible_path, pos, title, k, current_step=step)
        fig.canvas.draw_idle()

    update_model(0)
    add_legend(fig)

    # Buttons (row above legend)
    ax_prev = plt.axes([0.35, 0.12, 0.08, 0.05])
    ax_play = plt.axes([0.44, 0.12, 0.12, 0.05])
    ax_next = plt.axes([0.57, 0.12, 0.08, 0.05])
    btn_prev = Button(ax_prev, '< Prev')
    btn_play = Button(ax_play, 'Play')
    btn_next = Button(ax_next, 'Next >')

    anim_container = [None]

    def step_prev(_):
        state['playing'] = False
        btn_play.label.set_text('Play')
        if anim_container[0]:
            anim_container[0].event_source.stop()
        state['step'] = max(0, state['step'] - 1)
        update_model(state['step'])

    def step_next(_):
        state['playing'] = False
        btn_play.label.set_text('Play')
        if anim_container[0]:
            anim_container[0].event_source.stop()
        state['step'] = min(n_steps - 1, state['step'] + 1)
        update_model(state['step'])

    def toggle_play(_):
        if state['playing']:
            state['playing'] = False
            btn_play.label.set_text('Play')
            if anim_container[0]:
                anim_container[0].event_source.stop()
        else:
            state['playing'] = True
            btn_play.label.set_text('Pause')
            def animate(_frame):
                if not state['playing']:
                    return
                if state['step'] < n_steps - 1:
                    state['step'] += 1
                    update_model(state['step'])
                else:
                    state['playing'] = False
                    btn_play.label.set_text('Play')
            anim_container[0] = FuncAnimation(
                fig, animate, interval=300, cache_frame_data=False)
            fig.canvas.draw_idle()

    btn_prev.on_clicked(step_prev)
    btn_next.on_clicked(step_next)
    btn_play.on_clicked(toggle_play)

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_checkpoint():
    base = 'outputs/top_22'
    if not os.path.isdir(base):
        return None
    runs = sorted(os.listdir(base))
    for run in reversed(runs):
        run_dir = os.path.join(base, run)
        if os.path.isdir(run_dir) and any(f.endswith('.pt') for f in os.listdir(run_dir)):
            return run_dir
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int, default=1, help='Number of instances')
    parser.add_argument('--animate', action='store_true', help='Step-through animation')
    parser.add_argument('--save', type=str, default=None, help='Save to file (static only)')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint()
    if not checkpoint:
        print('No checkpoint found. Run training first.')
        exit(1)

    print(f'Loading model from: {checkpoint}')
    model, model_args = load_model(checkpoint)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    model.set_decode_type('greedy')
    print(f'Device: {device}')

    k = model_args.get('fat_tree_k', 4)
    torch.manual_seed(args.seed)

    for i in range(args.n):
        inst = generate_fat_tree_instance(k=k, min_visits_ratio=0.3)
        if model_args.get('min_visits') is not None:
            inst['min_visits'] = torch.tensor(model_args['min_visits'])

        cm    = inst['cost_matrix']
        start = inst['start_idx'].item()
        end   = inst['end_idx'].item()
        n     = cm.size(0)

        # BFS optimal path
        bfs_p    = bfs_shortest_path(cm, start, end, n)
        bfs_cost = float(len(bfs_p) - 1)

        # Model path
        batch = {kk: v.unsqueeze(0).to(device) for kk, v in inst.items()}
        with torch.no_grad():
            cost_t, _, pi = model(batch, return_pi=True)
        model_p    = [start] + pi[0].cpu().tolist()
        model_cost = cost_t[0].item()

        print(f'\nInstance #{i+1}: start={start} end={end} | '
              f'BFS={bfs_cost:.0f} hops | Model={model_cost:.0f} cost '
              f'({len(model_p)-1} steps, {len(set(model_p))} distinct nodes)')

        save_path = args.save if args.n == 1 else (
            args.save.replace('.', f'_{i+1}.') if args.save else None)

        if args.animate:
            show_animated(inst, bfs_p, model_p, model_cost, bfs_cost, k)
        else:
            show_static(inst, bfs_p, model_p, model_cost, bfs_cost, k, save=save_path)
