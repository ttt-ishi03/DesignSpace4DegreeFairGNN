import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import os.path as osp

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42

DEGREE_RANGE = {'citeseer': 30, 'chameleon': 50, 'arxiv-year': 30, 'emnlp': 30, 'squirrel': 60}
YLIM = {'citeseer': 1400, 'chameleon': 400, 'arxiv-year': 30000, 'emnlp': 1500, 'squirrel': 1400}

def save_test_results_fig(
        label, 
        pred, 
        recall,
        degree, 
        seed, 
        output_path,
        n_class,
        ylim=None, 
        dataset=None, 
        figsize=(4, 9/4),
        ratio=20,
    ):
    if ylim is None:
        ylim = int(YLIM[dataset]*0.25)
    label_class_distr_by_degree = calc_class_distribution(degree, label, n_class, ratio=ratio)
    preds_class_distr_by_degree = calc_class_distribution(degree, pred, n_class, ratio=ratio)
    show_multi_bar(
        np.arange(3),
        label_class_distr_by_degree,
        title="",
        ylim=ylim,
        filename=osp.join(output_path, f'test_label_class_distr_{seed}_ratio={ratio}.pdf'),
        xticks=["low", "mid", "high"],
        figsize=figsize,
    )
    show_multi_bar(
        np.arange(3),
        preds_class_distr_by_degree,
        title="",
        ylim=ylim,
        filename=osp.join(output_path, f'test_preds_class_distr_{seed}_ratio={ratio}.pdf'),
        xticks=["low", "mid", "high"],
        figsize=figsize,
    )
    show_multi_bar(
        np.arange(3),
        [[r0, r1, r2] for r0, r1, r2 in zip(recall['low'], recall['mid'], recall['high'])],
        title="",
        ylim=100,
        filename=osp.join(output_path, f'test_recall_distr_{seed}_ratio={ratio}.pdf'),
        xticks=["low", "mid", "high"],
        figsize=figsize,
        y_label='recall',
    )


def save_degree_dist(data, dataset, n_class, drange=None, ylim=None, save_dir='./', figsize=(4, 9/4), interval=5, ratio=20):
    if ylim is None:
        ylim = YLIM[dataset]
    if drange is None:
        drange = DEGREE_RANGE[dataset]
    degree = np.array(data.degree, dtype='uint16').flatten()
    label = np.array(data.labels, dtype='uint16').flatten()
    p = np.percentile(degree, [ratio, 100-ratio])
    degree_distr, degree_distr_class = calc_degree_distribution(degree, label, degree_range=drange)
    degree_distr_train, _ = calc_degree_distribution(degree, label, degree_range=drange, mask=data.idx_train)
    degree_distr_val, _ = calc_degree_distribution(degree, label, degree_range=drange, mask=data.idx_val)
    degree_distr_test, _ = calc_degree_distribution(degree, label, degree_range=drange, mask=data.idx_test)
    class_distr_by_degree = calc_class_distribution(degree, label, n_class, ratio=ratio)
    show_single_bar(
        np.arange(drange+1),
        degree_distr,
        title=f'{dataset} degree distribution',
        ylim=ylim,
        filename=os.path.join(save_dir, f'{dataset}_p=[{int(p[0])},{int(p[1])}]_degree_distribution.pdf'),
        figsize=figsize,
        interval=interval,
    )
    show_multi_bar(
        np.arange(drange+1),
        degree_distr_class,
        title=f'{dataset} degree distribution by class',
        ylim=ylim,
        filename=os.path.join(save_dir, f'{dataset}_p=[{int(p[0])},{int(p[1])}]_degree_distribution_by_class.pdf'),
        figsize=figsize,
        interval=interval,
    )
    show_multi_bar(
        np.arange(3),
        class_distr_by_degree,
        title="",
        ylim=ylim,
        filename=os.path.join(save_dir, f'{dataset}_p=[{int(p[0])},{int(p[1])}]_class_distribution_by_degree_ratio={ratio}.pdf'),
        xticks=["low", "mid", "high"],
        figsize=figsize,
        interval=interval,
    )
    show_multi_bar(
        np.arange(drange+1),
        [degree_distr_train, degree_distr_val, degree_distr_test],
        title=f'{dataset} degree distribution by train/val/test',
        ylim=ylim,
        labels=['train', 'val', 'test'],
        filename=os.path.join(save_dir, f'{dataset}_p=[{int(p[0])},{int(p[1])}]_degree_distribution_by_train_val_test.pdf'),
        figsize=figsize,
        interval=interval,
    )

def show_multi_plot(X, Y, title, labels=None, ylim=1.01, x_label = "degree", y_label = "accuracy", save_fig=True, filename="./hogehoge.pdf"):
    if (isinstance(Y, list) and not (isinstance(Y[0], list) or (isinstance(Y[0], np.ndarray) and Y[0].ndim == 1))) or (isinstance(Y, np.ndarray) and Y.ndim == 1):
        Y = [Y]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if labels is not None:
        for label, y in zip(labels, Y):
            ax.plot(X, y, label=label)
            # ax.plot(X, y, linewidth=3, label=label)
    else:
        for y in Y:
            ax.plot(X, y, linewidth=3)
    ax.set_title(title, y=-0.17)
    if labels is not None:
        ax.legend()
    
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.subplots_adjust(top=0.98, bottom=0.15)
    fig.show()
    if save_fig:
        fig.savefig(filename)

def inc_degree_list(degree_list: list, d):
    if d < len(degree_list) - 1:
        degree_list[d] += 1
    else:
        degree_list[-1] += 1


def calc_degree_distribution(degrees, labels, degree_range=15, mask = None):
    if type(labels) == np.ndarray:
        casted_labels = labels.astype(np.uint8)
    else:
        casted_labels = labels.cpu().numpy().astype(np.uint8)
    offset = min(casted_labels)
    casted_labels = casted_labels - offset
    
    n_class = len(np.unique(labels))

    degree_distr = np.zeros(degree_range+1, dtype='uint32')  # degree distribution
    degree_distr_class = np.zeros((n_class, degree_range+1), dtype='uint32')  # degree distribution by class
    
    if mask is None:
        for i, d in enumerate(degrees):
            inc_degree_list(degree_distr, d)
            inc_degree_list(degree_distr_class[casted_labels[i]], d)
    else:
        if len(mask) != len(casted_labels):
            casted_labels = casted_labels[mask]
        for d, label in zip(degrees[mask], casted_labels):
            inc_degree_list(degree_distr, d)
            inc_degree_list(degree_distr_class[label], d)
    
    return degree_distr.tolist(), degree_distr_class.tolist()

def calc_class_distribution(degrees, labels, n_class, mask = None, ratio=20):
    if type(labels) == np.ndarray:
        casted_labels = labels.astype(np.uint8)
    else:
        casted_labels = labels.cpu().numpy().astype(np.uint8)
    offset = min(casted_labels)
    casted_labels = casted_labels - offset
    
    # n_class = len(np.unique(casted_labels))
    # n_class = 5
    degree_distr_class = np.zeros((n_class, 3), dtype='uint32')  # degree distribution by class
    p = np.percentile(degrees, [ratio, 100-ratio])

    if mask is None:
        for i, d in enumerate(degrees):
            if d <= p[0]:
                idx = 0
            elif d > p[1]:
                idx = 2
            else:
                idx = 1
            degree_distr_class[casted_labels[i]][idx] += 1
    else:
        if len(mask) != len(casted_labels):
            casted_labels = casted_labels[mask]
        for d, label in zip(degrees[mask], casted_labels):
            if d < p[0]:
                idx = 0
            elif d >= p[1]:
                idx = 2
            else:
                idx = 1
            degree_distr_class[casted_labels[label]][idx] += 1
            
    return degree_distr_class.tolist()

def show_single_bar(
        X, 
        Y, 
        title, 
        ylim=None, 
        x_label = "degree", 
        y_label = "number of nodes", 
        save_fig=True, 
        filename="./hogehoge.pdf", 
        figsize = (16, 9),
        interval = 5,
        is_last_bar_nodes_GEqN = True,
    ):
    fig = plt.figure()
    
    n_pixel = 1920*1080
    dpi = int(np.sqrt(n_pixel / (figsize[0] * figsize[1])))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(X, Y)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    # ax.set_title(title, y=-0.22)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    if is_last_bar_nodes_GEqN:
        normal_xticks = [x for x in X[:-1] if x % interval == 0]
        ax.set_xticks(normal_xticks + [max(X)])
        ax.set_xticklabels(normal_xticks + [f'$\\geq{max(X)}$'])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(top=0.95, bottom=0.25, left=0.23, right=0.98)
    fig.show()
    if save_fig:
        fig.savefig(filename, dpi=dpi)

def show_multi_bar(
        X, 
        Y, 
        title, 
        ylim=None, 
        labels=None, 
        x_label = "degree", 
        y_label = "number of nodes", 
        save_fig=True, 
        filename="./hogehoge.pdf", 
        xticks = None, 
        figsize = (16, 9),
        interval = 5,
        is_last_bar_nodes_GEqN = True,
    ):
    n = len(Y)
    
    group_gap = 0.4
    bar_gap = 0.02
    width = (1 - group_gap)/n
    shift = width + bar_gap

    n_pixel = 1920*1080
    dpi = int(np.sqrt(n_pixel / (figsize[0] * figsize[1])))
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        if labels is not None:
            ax.bar(X+shift*i, Y[i], width, label = labels[i])
        else:
            ax.bar(X+shift*i, Y[i], width)
    if labels is not None:
        ax.legend()
    
    # ax.set_title(title, y=-0.32)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    if xticks is not None:
        ax.set_xticks(X)
        ax.set_xticklabels(xticks)
    elif is_last_bar_nodes_GEqN:
        normal_xticks = [x for x in X[:-1] if x % interval == 0]
        ax.set_xticks(normal_xticks + [max(X)])
        ax.set_xticklabels(normal_xticks + [f'$\\geq{max(X)}$'])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(top=0.95, bottom=0.25, left=0.23, right=0.98)
    fig.show()
    if save_fig:
        fig.savefig(filename, dpi=dpi)
