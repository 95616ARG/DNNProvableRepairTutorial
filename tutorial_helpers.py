# %matplotlib inline
import ipywidgets
import warnings; warnings.filterwarnings("ignore")

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

import sytorch as st
import torch
import numpy as np

torch.set_grad_enabled(False)

import matplotlib as mpl
from matplotlib import pyplot as plt


def get_dnn():
    with st.no_grad():
        dnn = st.nn.Sequential(
            st.nn.Linear(1, 3),
            st.nn.ReLU(),
            st.nn.Linear(3, 1)
        )
        dnn[0].weight[:] = torch.as_tensor([[-1., 1., 0.5]]).T
        dnn[0].bias[:]   = torch.as_tensor([0., -2., 0.])
        dnn[2].weight[:] = torch.as_tensor([[0.5, -0.5, 1.]])
        dnn[2].bias[:]   = torch.as_tensor([-0.5])

        return dnn
    
xmin, xmax =  -2., 4.
ymin, ymax = -0.7, 1.

def find_linear_regions(dnn, xlim, num=10000, step=1e-2):
    # X = torch.linspace(*xlim, num)[...,None]
    X = torch.arange(*xlim, step=step)[...,None]
    AP = dnn.activation_pattern(X)[1]
    bounderies = list(X[:-1][(AP[1:] != AP[:-1]).any(-1)].flatten())
    
    if len(bounderies) == 0:
        return (xlim, )
    
    X = X.flatten().tolist()
    if bounderies[0] != X[0]:
        bounderies.insert(0, X[0])
        
    if bounderies[-1] == X[-2]:
        bounderies[-1] = X[-1]
    else:
        bounderies.append(X[-1])
        
    return list(zip(bounderies[:-1], bounderies[1:]))

def plot(dnn, xlim=(-2., 4.), ylim=(), ax=None, **kwargs):
    X = torch.linspace(*xlim, 100)[...,None]
    # X = torch.arange(*xlim, step)[...,None]
    Y = dnn(X)
    if ax is None: ax = plt.gca()
    ax.plot(X, Y, **kwargs)

def plot_regions(dnn, xlim=(-2., 4.), ylim=(), ax=None, **kwargs):
    regions = find_linear_regions(dnn, xlim)
    for xlim in regions:
        plot(dnn, xlim=xlim, ax=ax)

def plot_polytope(dnn, x1, x2, **kwargs):
    if x1 > x2: x1, x2 = x2, x1
    regions = find_linear_regions(dnn, (x1, x2))
    for x1, x2 in regions:
        points = torch.Tensor([x1, x2])[...,None]
        plot_segment(dnn, points, **kwargs)

def plot_points(N, points, lb, ub, ax=None, alpha=1., label='endpoints', **kwargs):
    if ax is None: ax=plt.gca()
    with torch.no_grad(), st.no_symbolic():
        outputs = N(points)
        sat_mask = (lb - 1e-4 <= outputs) * (outputs <= ub + 1e-4)
        unsat_mask = ~sat_mask
        # print(outputs[sat_mask])
        # print(outputs[unsat_mask])
    ax.scatter(points[  sat_mask], outputs[  sat_mask], color='b', label=f"sat. {label}", **kwargs)
    ax.scatter(points[unsat_mask], outputs[unsat_mask], color='r', label=f"unsat. {label}", **kwargs)

def plot_segment(N, points, lb, ub, ax=None, alpha=1., **kwargs):
    if ax is None: ax=plt.gca()

    with torch.no_grad(), st.no_symbolic():

        x1, x2 = points.reshape(-1).tolist()
        X = torch.arange(x1, x2, 1e-2)[...,None]
        Y = N(X)
        sat_mask = (lb - 1e-4 <= Y) * (Y <= ub + 1e-4)
        unsat_mask = ~sat_mask

        if not sat_mask.any():
            ax.plot(points, N(points), 'r', linewidth=3)
            return

        z1, z2 = X[sat_mask][[0, -1]]

        ax.plot(points, N(points), 'b', linewidth=3)

        if x1 <= z1:
            pts = torch.Tensor([x1, z1])[...,None]
            ax.plot(pts, N(pts), 'r', linewidth=3)

        if z2 <= x2:
            pts = torch.Tensor([z2, x2])[...,None]
            ax.plot(pts, N(pts), 'r', linewidth=3)

def draw_neural_net(ax, N, N0=None, ap=None, left=0.1, right=0.9, bottom=0., top=1.):
    if N0 is None:
        N0 = N

    layer_sizes = [N[0].weight.shape[1]] + [l.bias.shape[0] for l in N if hasattr(l, 'bias')]
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

            x, y = circle.get_center()
            if ap is not None and n % 2 == 1:
                node_ap = ap[n//2]
                ax.text(
                        x+0.03, y-0.07, 'On' if node_ap[m] else 'Off', 
                        ha='left',
                        zorder=10,
                        bbox=dict(
                            facecolor='white', 
                            edgecolor='red' if node_ap[m] else 'blue', 
                            alpha=1., 
                            pad=2.0),)

            if n > 0:
                b0 = N0[n*2-2].bias[m]
                b1 = N[n*2-2].bias[m]
                if b0 == b1:
                    ax.text(x, y+.1, rf"{b1:.1f}",
                            c='k',
                            ha='center',
                            fontfamily='monospace',
                            fontweight=1000)
                else:
                    ax.text(x, y+.1, rf"{b1:.1f}",
                            bbox=dict(facecolor='lime', edgecolor='lime', alpha=.5, pad=2.0),
                            c='k',
                            ha='center',
                            fontfamily='monospace',
                            fontweight=1000)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

                x, y = line.get_xydata().mean(0)
                w0 = N0[n*2].weight[o,m]
                w1 = N[n*2].weight[o,m]
                if w0 == w1:
                    ax.text(
                            x, y+.03,
                            rf"{w1:.1f}",
                            c='k',
                            ha='center',
                            fontfamily='monospace',
                            fontweight=1000)
                else:
                    ax.text(
                            x, y+.03,
                            rf"{w1:.1f}",
                            bbox=dict(facecolor='lime', edgecolor='lime', alpha=.5, pad=2.0),
                            c='k',
                            ha='center',
                            fontfamily='monospace',
                            fontweight=1000)


def interactive_pointwise_repair(
        pointwise_repair,
        x1=-1.5, 
        lb=-0.1, ub=0.1, 
        ref=-1., 
        r0=True, r1=True, r2=True,
        ap_mode='ref',
        run=True):

    r0 = r0 == 'On'
    r1 = r1 == 'On'
    r2 = r2 == 'On'
    fig, ((ax0n, ax0), (ax1n, ax1)) = plt.subplots(2, 2, figsize=(9, 8))

    N0 = get_dnn()
    N = get_dnn()
    points = torch.tensor([x1])[:,None]
    ref_points = torch.tensor([ref])[:,None]

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    plot_regions(N, ax=ax0)
    plot_points(N, points, lb=lb, ub=ub, ax=ax0, label='points')
    
    if ap_mode != 'Manual':
        ref_x = ref
        ref_y = N(ref_points[0]).item()
        ax0.scatter(ref_x, ref_y, marker='o', facecolors='none', edgecolors='k', label="ref point")
        ref_ap = N.activation_pattern(ref_points[0:1])[1]
        ax0.annotate(
                f"Activation pattern\nfrom ref. point\n{tuple('On' if b else 'Off' for b in ref_ap[0])}", 
                xy=(ref_x, ref_y), 
                xytext=(ref_x+.8, ref_y+.4),
                ha='center',
                arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.4"))
        
    ax0.hlines([lb, ub], xmin=xmin-1., xmax=xmax+1., alpha=.6, color='k', linestyles='dashed', label="bounds")
    # ax0.legend(loc='upper right', bbox_to_anchor=(3.1, 1.))
    ax0.legend(loc='upper right', bbox_to_anchor=(1.8, 1.))
    
    
    ax0t = ax0.twinx()
    ax0t.set_ylim(ymin, ymax)
    ax0t.set_yticks([lb, ub])
    
    ax0n.axis('off')
    draw_neural_net(ax0n, N, N0, ap=ref_ap if ap_mode != 'Manual' else N.activation_pattern(points)[1])

    if not run:
        return

    with suppress_stdout_stderr():
        if ap_mode == 'Manual':
            ap = [[], np.asanyarray([[r0, r1, r2]]), []]
        else:
            ap = N.activation_pattern(ref_points)

        N = pointwise_repair(N=N, 
                            x1=x1,
                            lb=lb, ub=ub,
                            ap=ap)
    
        # solver = st.GurobiSolver().verbose_(False)
        # N.to(solver).requires_symbolic_weight_and_bias().repair()
        # if ap_mode == 'Manual':
        #     ap = [[], np.asanyarray([[r0, r1, r2], [r0, r1, r2]]), []]
        # else:
        #     ap = N.activation_pattern(ref_points)
        # sy = N(points, pattern=ap)
        # param_deltas = N.parameter_deltas()
        # succeed =  solver.solve(
        #     lb <= sy, sy <= ub,
        #     minimize = param_deltas.norm_ub('linf+l1_normalized')
        # )
    if N is None:
        print("Infeasible!")
        return 

    # plot_regions(N)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    plot_regions(N, ax=ax1)
    plot_points(N, points, lb=lb, ub=ub, ax=ax1, label='points')
    ax1.hlines([lb, ub], xmin=xmin-1., xmax=xmax+1., alpha=.6, color='k', linestyles='dashed', label="bounds")

    ax1t = ax1.twinx()
    ax1t.set_ylim(ymin, ymax)
    ax1t.set_yticks([lb, ub])
    
    plt.subplots_adjust(wspace=.35)
    fig.suptitle(rf"$\mathcal{{N}}({x1:.1f}) \in [{lb:.1f},{ub:.1f}]$",
                 size=20, weight=1000)

    ax1n.axis('off')
    draw_neural_net(ax1n, N, N0, ap=ap[1])


def interactive_polytope_repair(
        polytope_repair,
        x1=-1.5, x2=-0.5, 
        lb=-0.1, ub=0.1, 
        ref=-1., 
        r0=True, r1=True, r2=True,
        ap_mode='ref',
        run=True):

    r0 = r0 == 'On'
    r1 = r1 == 'On'
    r2 = r2 == 'On'
    fig, ((ax0n, ax0), (ax1n, ax1)) = plt.subplots(2, 2, figsize=(9, 8))

    N0 = get_dnn()
    N = get_dnn()
    points = torch.tensor([x1, x2])[:,None]
    ref_points = torch.tensor([ref, ref])[:,None]

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    plot_regions(N, ax=ax0)
    plot_points(N, points, lb=lb, ub=ub, ax=ax0)
    plot_polytope(N, x1, x2, lb=lb, ub=ub, ax=ax0)
    ax0.plot(points-10, N(points), 'r', linewidth=3, label='unsat. segment')
    ax0.plot(points-10, N(points), 'b', linewidth=3, label='sat. segment')

    
    if ap_mode != 'Manual':
        ref_x = ref
        ref_y = N(ref_points[0]).item()
        ax0.scatter(ref_x, ref_y, marker='o', facecolors='none', edgecolors='k', label="ref point")
        ref_ap = N.activation_pattern(ref_points[0:1])[1]
        ax0.annotate(
                f"Activation pattern\nfrom ref. point\n{tuple('On' if b else 'Off' for b in ref_ap[0])}", 
                xy=(ref_x, ref_y), 
                xytext=(ref_x+.8, ref_y+.4),
                ha='center',
                arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.4"))
        
    ax0.hlines([lb, ub], xmin=xmin-1., xmax=xmax+1., alpha=.6, color='k', linestyles='dashed', label="bounds")
    # ax0.legend(loc='upper right', bbox_to_anchor=(3.1, 1.))
    ax0.legend(loc='upper right', bbox_to_anchor=(1.8, 1.))
    
    
    ax0t = ax0.twinx()
    ax0t.set_ylim(ymin, ymax)
    ax0t.set_yticks([lb, ub])
    
    ax0n.axis('off')
    draw_neural_net(ax0n, N, N0, ap=ref_ap if ap_mode != 'Manual' else None)

    if not run:
        return

    with suppress_stdout_stderr():
        if ap_mode == 'Manual':
            ap = [[], np.asanyarray([[r0, r1, r2], [r0, r1, r2]]), []]
        else:
            ap = N.activation_pattern(ref_points)

        N = polytope_repair(N=N, 
                            x1=x1, x2=x2, 
                            lb=lb, ub=ub,
                            ap=ap)
    
        # solver = st.GurobiSolver().verbose_(False)
        # N.to(solver).requires_symbolic_weight_and_bias().repair()
        # if ap_mode == 'Manual':
        #     ap = [[], np.asanyarray([[r0, r1, r2], [r0, r1, r2]]), []]
        # else:
        #     ap = N.activation_pattern(ref_points)
        # sy = N(points, pattern=ap)
        # param_deltas = N.parameter_deltas()
        # succeed =  solver.solve(
        #     lb <= sy, sy <= ub,
        #     minimize = param_deltas.norm_ub('linf+l1_normalized')
        # )
    if N is None:
        print("Infeasible!")
        return 

    # plot_regions(N)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    plot_regions(N, ax=ax1)
    plot_points(N, points, lb=lb, ub=ub, ax=ax1)
    plot_segment(N, points, lb=lb, ub=ub, ax=ax1)
    ax1.hlines([lb, ub], xmin=xmin-1., xmax=xmax+1., alpha=.6, color='k', linestyles='dashed', label="bounds")

    ax1t = ax1.twinx()
    ax1t.set_ylim(ymin, ymax)
    ax1t.set_yticks([lb, ub])
    
    plt.subplots_adjust(wspace=.35)
    fig.suptitle(rf"$\forall x\in[{x1:.1f},{x2:.1f}].\mathcal{{N}}(x) \in [{lb:.1f},{ub:.1f}]$",
                 size=20, weight=1000)

    ax1n.axis('off')
    draw_neural_net(ax1n, N, N0, ap=ap[1])

x1 = ipywidgets.FloatSlider(value=-1.5, min=xmin, max=xmax, description='x1')
x2 = ipywidgets.FloatSlider(value=-0.5, min=xmin, max=xmax, description='x2')
ref= ipywidgets.FloatSlider(value= -1., min=xmin, max=xmax, description='Ref. Point')
ap_mode = ipywidgets.ToggleButtons(
        options=['Manual', 'Reference Point'],
        description='Use activation pattern from',
        disabled=False,
        button_style='info',
    )
r0 = ipywidgets.ToggleButtons(
        options=['On', 'Off'],
        description='Manual activation pattern:',
        disabled=False,
        button_style='',
    )
r1 = ipywidgets.ToggleButtons(
        options=['On', 'Off'],
        disabled=False,
        button_style='',
    )
r2 = ipywidgets.ToggleButtons(
        options=['On', 'Off'],
        disabled=False,
        button_style='',
    )
lb = ipywidgets.FloatSlider(value=-0.1, min=ymin, max=ymax, description='lb')
ub = ipywidgets.FloatSlider(value= 0.1, min=ymin, max=ymax, description='ub')

def interact_polytope_repair_with(polytope_repair):
    r0.value='On'
    r1.value='Off'
    r2.value='Off'
    ui = ipywidgets.HBox([
        ipywidgets.VBox([x1, x2, ub, lb]), 
        ipywidgets.VBox([ap_mode, ref]),
        ipywidgets.VBox([r0, r1, r2]),
    ], )
    def f(**kwargs):
        return interactive_polytope_repair(polytope_repair, **kwargs)
    out = ipywidgets.interactive_output(
        f, {
        'x1': x1, 'x2': x2, 'ref': ref, 
        'lb': lb, 'ub': ub, 
        'r0': r0, 'r1': r1, 'r2': r2,
        'ap_mode': ap_mode,
    })
    out.layout = ipywidgets.Layout(height='800px')
    display(ui, out)


def interact_pointwise_repair_with(pointwise_repair):
    r0.value='On'
    r1.value='Off'
    r2.value='Off'
    ui = ipywidgets.HBox([
        ipywidgets.VBox([x1, ub, lb]), 
        ipywidgets.VBox([ap_mode, ref]),
        ipywidgets.VBox([r0, r1, r2]),
    ], )
    def f(**kwargs):
        return interactive_pointwise_repair(pointwise_repair, **kwargs)
    out = ipywidgets.interactive_output(
        f, {
        'x1': x1, 'ref': ref, 
        'lb': lb, 'ub': ub, 
        'r0': r0, 'r1': r1, 'r2': r2,
        'ap_mode': ap_mode,
    })
    out.layout = ipywidgets.Layout(height='800px')
    display(ui, out)

