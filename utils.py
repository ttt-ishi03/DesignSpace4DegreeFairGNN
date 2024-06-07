import argparse
import copy
import csv
import json
import math
import random
import sys

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
# import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))

    return sum_m

def ttest_p_value(A, B, var_p=0.05, alternative="two-sided"):
    if len(A) == 0 or len(B) == 0:
        return 1.0
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)
    A_df = len(A) - 1
    B_df = len(B) - 1
    f = np.var(A, ddof=1) / np.var(B, ddof=1)
    one_sided_pval1 = stats.f.cdf(f, A_df, B_df)
    one_sided_pval2 = stats.f.sf(f, A_df, B_df)
    two_sided_pval = min(one_sided_pval1, one_sided_pval2) * 2

    ret = stats.ttest_ind(A, B, equal_var=(two_sided_pval >= var_p), alternative=alternative)
    return ret.pvalue


def read_hyper_params_from_csv(csv_file, line):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                col = row.index("params")
            if i + 1 == line:
                params = row[col]
    params = params.replace("'", '"')
    return json.loads(params)


def fix_type2json(data):
    if isinstance(data, np.ndarray):
        try:
            data = data.item()
        except Exception:
            data = list(data)

    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = fix_type2json(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            data[i] = fix_type2json(v)
    elif isinstance(data, argparse.Namespace):
        data = vars(data)
        data = fix_type2json(data)
    elif isinstance(data, np.int64):
        data = int(data)
    elif isinstance(data, np.float64):
        data = float(data)
    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.float32 or data.dtype == torch.float64:
            data = float(data)
        elif data.dtype == torch.int64:
            data = int(data)
        else:
            print(f"Invalid dtype {data.dtype}", file=sys.stderr)
            exit(1)
    elif isinstance(data, (int, float, str)) or data is None:
        pass
    else:
        print(f"Invalid data type {type(data)}", file=sys.stderr)
        exit(1)

    return data


def compute_CI(out_list, name=None, log_file=None):
    # ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = (
        name
        + " Mean: {:.4f} ".format(np.mean(out_list))
        + "Std: {:.4f}".format(st.sem(out_list))
    )
    print(log)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def confidence_interval(arr=None):
    mean = np.mean(arr)
    ci = 1.96 * st.sem(arr) / math.sqrt(len(arr))

    print("Mean: {:.4f}, CI: {:.4f}".format(mean, ci))
    return


def get_str_info(adj, n_layers=2):

    str_info = adj
    for _ in range(n_layers):
        str_info = torch.matmul(str_info, adj)

    torch.set_printoptions(profile="full")
    print("struc: ", str_info)
    return torch.sum(str_info, dim=1)

def metrics_by_degree(pred, data, groups, degree_range):
    accs = []
    macfs = []
    weifs = []

    data_groups = groups[data.idx_test]
    pred = pred[data.idx_test]
    labels = data.labels[data.idx_test].cpu()
    for i in range(degree_range + 1):
        if i < degree_range:
            mask = np.where(data_groups == i)[0]
        else:
            mask = np.where(data_groups >= degree_range)[0]
        if sum(mask) != 0:
            acc = accuracy_score(pred[mask], labels[mask])
            macf = f1_score(pred[mask], labels[mask], average="macro")
            weif = f1_score(pred[mask], labels[mask], average="weighted")
        else:
            acc = None
            macf = None
            weif = None
        accs.append(acc)
        macfs.append(macf)
        weifs.append(weif)

    return accs, macfs, weifs

def evaluate_fairness_light(pred, data, groups):

    def fair_metric(p):
        groups = np.copy(data_groups)
        groups = -1

        # mask 2 groups
        groups = np.where(data_groups <= p[0], 0.0, groups)
        groups = np.where(data_groups > p[1], 1.0, groups)

        idx_s0 = np.where(groups == 0)[0]
        idx_s1 = np.where(groups == 1)[0]

        # print(idx_s0.shape, idx_s1.shape)

        SP = []
        EO = []

        for i in range(data.labels.max() + 1):
            # SP

            p_i0 = np.where(pred[idx_s0] == i)[0]
            p_i1 = np.where(pred[idx_s1] == i)[0]
            sp = abs(
                (p_i0.shape[0] / idx_s0.shape[0]) - (p_i1.shape[0] / idx_s1.shape[0])
            )
            SP.append(sp)

            # EO
            p_y0 = np.where(labels[idx_s0] == i)[0]
            p_y1 = np.where(labels[idx_s1] == i)[0]

            temp0 = pred[idx_s0]
            temp1 = pred[idx_s1]

            p_iy0 = np.where(temp0[p_y0] == i)[0]
            p_iy1 = np.where(temp1[p_y1] == i)[0]

            if p_y0.shape[0] == 0 or p_y1.shape[0] == 0:
                eo = 0
            else:
                eo = abs(
                    (p_iy0.shape[0] / p_y0.shape[0]) - (p_iy1.shape[0] / p_y1.shape[0])
                )
            EO.append(eo)


        SP = np.asarray(SP) * 100
        EO = np.asarray(EO) * 100

        return {
            "sp": SP,
            "eo": EO,
        }

    data_groups = groups[data.idx_test]
    pred = pred[data.idx_test]
    labels = data.labels[data.idx_test].cpu()
    np.set_printoptions(precision=3, suppress=True)

    portion = [20]

    for por in portion:
        formatted_data_groups = np.array(data_groups)
        formatted_data_groups = np.ravel(formatted_data_groups)
        p = np.percentile(formatted_data_groups, [por, 100 - por])
        # sp, eo, h_acc, t_acc, h_macf, t_macf, h_weif, t_weif, prob_pred0, prob_pred1, n_data0, n_data1, recall0, recall1 = fair_metric(p)
        dic = fair_metric(p)

        dic["mu_sp"] = np.mean(dic["sp"])
        dic["mu_eo"] = np.mean(dic["eo"])
    return dic


def evaluate_fairness(pred, data, groups, embed=None, print_debug=False):

    def fair_metric(p):
        groups = np.copy(data_groups)
        groups = -1

        # mask 2 groups
        groups = np.where(data_groups <= p[0], 0.0, groups)
        groups = np.where(data_groups > p[1], 1.0, groups)

        idx_s0 = np.where(groups == 0)[0]
        idx_s1 = np.where(groups == 1)[0]
        idx_other = np.where(groups == -1)[0]

        # print(idx_s0.shape, idx_s1.shape)

        SP = []
        EO = []
        prob_pred0 = []
        prob_pred1 = []
        prob_pred_other = []
        n_data0 = []
        n_data1 = []
        n_data_other = []
        recall0 = []
        recall1 = []
        recall_other = []

        for i in range(data.labels.max() + 1):
            # SP
            if print_debug:
                print("\nlabel =", i)

            p_i0 = np.where(pred[idx_s0] == i)[0]
            p_i1 = np.where(pred[idx_s1] == i)[0]
            p_other = np.where(pred[idx_other] == i)[0]
            sp = abs(
                (p_i0.shape[0] / idx_s0.shape[0]) - (p_i1.shape[0] / idx_s1.shape[0])
            )
            if print_debug:
                print("cnt of pred=={} (low) = {}".format(i, p_i0.shape[0]))
                print("cnt of pred=={} (high) = {}".format(i, p_i1.shape[0]))
                print(
                    "prob of pred=={} (low) = {:.3f}".format(
                        i, p_i0.shape[0] / idx_s0.shape[0]
                    )
                )
                print(
                    "prob of pred=={} (high) = {:.3f}".format(
                        i, p_i1.shape[0] / idx_s1.shape[0]
                    )
                )
            prob_pred0.append(p_i0.shape[0] / idx_s0.shape[0])
            prob_pred1.append(p_i1.shape[0] / idx_s1.shape[0])
            prob_pred_other.append(p_other.shape[0] / idx_other.shape[0])
            if print_debug:
                print("sp = {:.3f}".format(sp))
            SP.append(sp)

            # EO
            p_y0 = np.where(labels[idx_s0] == i)[0]
            p_y1 = np.where(labels[idx_s1] == i)[0]
            p_y_other = np.where(labels[idx_other] == i)[0]
            if print_debug:
                print(f"number of label=={i}(low) =", (p_y0.shape[0]))
                print(f"number of label=={i}(higth) =", (p_y1.shape[0]))
            n_data0.append(p_y0.shape[0])
            n_data1.append(p_y1.shape[0])
            n_data_other.append(p_y_other.shape[0])

            temp0 = pred[idx_s0]
            temp1 = pred[idx_s1]
            temp_other = pred[idx_other]

            p_iy0 = np.where(temp0[p_y0] == i)[0]
            p_iy1 = np.where(temp1[p_y1] == i)[0]
            p_iy_other = np.where(temp_other[p_y_other] == i)[0]

            if p_y0.shape[0] == 0 or p_y1.shape[0] == 0:
                eo = 0
                # print('eo=0, shape[0]=0')
                recall0.append(-1)
                recall1.append(-1)
                recall_other.append(-1)
            else:
                eo = abs(
                    (p_iy0.shape[0] / p_y0.shape[0]) - (p_iy1.shape[0] / p_y1.shape[0])
                )
                if print_debug:
                    print("recall(group 0) = {:.3f}".format(p_iy0.shape[0] / p_y0.shape[0]))
                    print("recall(group 1) = {:.3f}".format(p_iy1.shape[0] / p_y1.shape[0]))
                    print("eo = {:.3f}".format(eo))
                recall0.append(p_iy0.shape[0] / p_y0.shape[0])
                recall1.append(p_iy1.shape[0] / p_y1.shape[0])
                recall_other.append(p_iy_other.shape[0] / p_y_other.shape[0])
            EO.append(eo)

            # print(p_iy0.shape, p_y0.shape)
            # print(p_iy1.shape, p_y1.shape)

        """
        degree = data.degree[data.idx_test].cpu().detach().numpy()
        mean_degree = np.mean(degree)
        #K = mean_degree * self.k
        R = np.where(degree < mean_degree, 0, 1)

        acc_idx_s0 = np.where(R == 0)[0]
        acc_idx_s1 = np.where(R == 1)[0]
        """

        # group accuracy
        tail_acc = accuracy_score(labels[idx_s0], pred[idx_s0])
        tail_macf = f1_score(labels[idx_s0], pred[idx_s0], average="macro")
        tail_weif = f1_score(labels[idx_s0], pred[idx_s0], average="weighted")

        head_acc = accuracy_score(labels[idx_s1], pred[idx_s1])
        head_macf = f1_score(labels[idx_s1], pred[idx_s1], average="macro")
        head_weif = f1_score(labels[idx_s1], pred[idx_s1], average="weighted")

        other_acc = accuracy_score(labels[idx_other], pred[idx_other])
        other_macf = f1_score(labels[idx_other], pred[idx_other], average="macro")
        other_weif = f1_score(labels[idx_other], pred[idx_other], average="weighted")

        SP = np.asarray(SP) * 100
        EO = np.asarray(EO) * 100
        head_acc = head_acc * 100
        tail_acc = tail_acc * 100
        other_acc = other_acc * 100
        head_macf = head_macf * 100
        tail_macf = tail_macf * 100
        other_macf = other_macf * 100
        head_weif = head_weif * 100
        tail_weif = tail_weif * 100
        other_weif = other_weif * 100
        prob_pred0 = np.asarray(prob_pred0) * 100
        prob_pred1 = np.asarray(prob_pred1) * 100
        prob_pred_other = np.asarray(prob_pred_other) * 100
        recall0 = np.asarray(recall0) * 100
        recall1 = np.asarray(recall1) * 100
        recall_other = np.asarray(recall_other) * 100

        ret = {
            "sp": SP,
            "eo": EO,
            "head_acc": head_acc,
            "tail_acc": tail_acc,
            "other_acc": other_acc,
            "head_macf": head_macf,
            "tail_macf": tail_macf,
            "other_macf": other_macf,
            "head_weif": head_weif,
            "tail_weif": tail_weif,
            "other_weif": other_weif,
            "prob_pred0": prob_pred0,
            "prob_pred1": prob_pred1,
            "prob_pred_other": prob_pred_other,
            "n_data0": n_data0,
            "n_data1": n_data1,
            "n_data_other": n_data_other,
            "recall0": recall0,
            "recall1": recall1,
            "recall_other": recall_other,
        }
        return ret

    data_groups = groups[data.idx_test]
    # pred = pred.cpu()
    pred = pred[data.idx_test]
    labels = data.labels[data.idx_test].cpu()
    np.set_printoptions(precision=3, suppress=True)

    # Equal group: 20, 30
    portion = [20]  # 30
    # output = []

    for por in portion:
        formatted_data_groups = np.array(data_groups)
        formatted_data_groups = np.ravel(formatted_data_groups)
        p = np.percentile(formatted_data_groups, [por, 100 - por])
        # sp, eo, h_acc, t_acc, h_macf, t_macf, h_weif, t_weif, prob_pred0, prob_pred1, n_data0, n_data1, recall0, recall1 = fair_metric(p)
        dic = fair_metric(p)

        dic["mu_sp"] = np.mean(dic["sp"])
        dic["mu_eo"] = np.mean(dic["eo"])
        dic["diff_acc"] = dic["head_acc"] - dic["tail_acc"]
        dic["diff_macf"] = dic["head_macf"] - dic["tail_macf"]
        dic["diff_weif"] = dic["head_weif"] - dic["tail_weif"]
        dic["delta_f1"] = np.abs(dic["diff_macf"]) / dic["head_macf"]

        if print_debug:
            print(
                "{:d} split: Mean SP={:.2f}, Mean EO={:.2f}".format(
                    por, dic["mu_sp"], np.mean(dic["mu_eo"])
                )
            )
            print(
                "Head Acc: {:.4f}, Tail Acc: {:.4f}".format(
                    dic["head_acc"], dic["tail_acc"]
                )
            )

    return dic


def row_shuffle(x):
    return x[torch.randperm(x.size()[0])]


# def count_degree(adj):
#     edge = adj.coalesce().indices()
#     edge = np.array(edge.cpu())
#     n_node = edge.max() + 1
#     degrees = np.zeros(n_node, dtype='uint16')  # degree for each node

#     if edge.min() != 0:
#         print('id of node must be starting from 0, but got {}'.format(edge.min()), file=sys.stderr)
#         exit(1)

#     u, counts = np.unique(edge[0], return_counts=True)
#     degrees[u] = counts

#     return degrees

# def dsp_loss(output, degree):
#     def other_max(values):
#         values = [torch.max(v[])]

#     def differentiable_ratio(prob):
#         F.sigmoid()


def degree_contrastive_cos_similarity_loss(x, label, degree, p=0.2):
    label = label.cpu().detach().numpy()
    degree = np.ravel(degree.cpu().detach())

    head = dict()
    tail = dict()
    head_mask = degree > np.percentile(degree, (1 - p) * 100)
    tail_mask = degree <= np.percentile(degree, p * 100)
    classes = set(label)
    for i in classes:
        label_mask = label == i
        tail[i] = x[np.logical_and(tail_mask, label_mask)]
        head[i] = x[np.logical_and(head_mask, label_mask)]

    loss = 0
    n_couple = 0
    for i in classes:
        n_row = min(head[i].shape[0], tail[i].shape[0])
        head[i] = row_shuffle(head[i])
        tail[i] = row_shuffle(tail[i])
        # print('similar')
        # print('head :', head[i][0])
        # print('tail :', tail[i][0])
        loss += torch.sum(
            torch.exp(-torch.cosine_similarity(head[i][:n_row], tail[i][:n_row], dim=1))
        )
        # tmp = torch.mean(torch.cosine_similarity(head[i][:n_row], tail[i][:n_row], dim=1)).cpu().detach()
        # print('similarity(same label): {:.3f}'.format(tmp))
        n_couple += n_row

        j = random.choice(list(classes - set([i])))
        n_row = min(head[j].shape[0], tail[i].shape[0])
        head[j] = row_shuffle(head[j])
        tail[i] = row_shuffle(tail[i])
        # print('different')
        # print('head :', head[j][0])
        # print('tail :', tail[i][0])
        loss += torch.sum(
            torch.exp(torch.cosine_similarity(head[j][:n_row], tail[i][:n_row], dim=1))
        )
        # tmp = torch.mean(torch.cosine_similarity(head[j][:n_row], tail[i][:n_row], dim=1)).cpu().detach()
        # print('similarity(different label): {:.3f}'.format(tmp))
        n_couple += n_row

    loss = loss / n_couple
    return loss


def degree_contrastive_loss(x, label, degree, p=0.2):

    def sim(hu, hv):
        n_row = min(hu.shape[0], hv.shape[0])
        hu = row_shuffle(hu)
        hv = row_shuffle(hv)
        similarity = torch.cosine_similarity(hu[:n_row], hv[:n_row], dim=1)
        similarity = torch.mean(similarity)
        return similarity

    label = label.cpu().detach().numpy()
    degree = np.ravel(degree.cpu().detach())

    head = dict()
    tail = dict()
    head_mask = degree > np.percentile(degree, (1 - p) * 100)
    tail_mask = degree <= np.percentile(degree, p * 100)
    classes = set(label)
    for i in classes:
        label_mask = label == i
        tail[i] = x[np.logical_and(tail_mask, label_mask)]
        head[i] = x[np.logical_and(head_mask, label_mask)]

    loss = 0
    for i in classes:
        den = 0
        for j in classes:
            if i != j:
                den += torch.exp(sim(head[i], tail[j]))
        loss += -torch.log(torch.exp(sim(head[i], tail[i])) / den)

    loss = loss / len(classes)
    return loss


def degree_weighted_cross_entropy_loss(output, label, degree_label):
    cross_entropy_loss = nn.CrossEntropyLoss()
    n_degree_group = torch.max(degree_label) + 1
    mask = torch.zeros(n_degree_group, output.shape[0], dtype=torch.bool).cuda()
    for i in range(n_degree_group):
        mask[i] = degree_label == i
    loss_by_degree_group = torch.zeros(n_degree_group).cuda()
    for i in range(n_degree_group):
        loss_by_degree_group[i] = cross_entropy_loss(output[mask[i]], label[mask[i]])
    n_node_by_degree_group = torch.sum(mask, axis=1)
    weight_by_degree_group = loss_by_degree_group / n_node_by_degree_group
    weight_by_degree_group = weight_by_degree_group / torch.mean(weight_by_degree_group)
    loss = torch.dot(weight_by_degree_group, loss_by_degree_group)

    return loss


def low_degree_specific_loss(output, label, degree_label):
    mask = degree_label == 0
    loss = nn.CrossEntropyLoss()

    for i in range(len(output)):
        if mask[i] and loss(output[i], label[i]) > 10:
            print(f'{i}: {output[i]}, {label[i]}, {loss(output[i], label[i])}')
    print(sum(mask))
    return loss(output[mask], label[mask])


def create_dummy_adj(adj, degree, rng):
    degree = torch.flatten(degree).cpu().numpy()
    n_node = len(degree)
    p = np.percentile(degree, 20)
    delete_target_mask = degree > p
    delete_target_idx = np.where(delete_target_mask)
    adj = adj.coalesce()
    indices = copy.deepcopy(adj.indices())
    values = copy.deepcopy(adj.values())
    values = values.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    delete_indices = []
    offset = 0
    for d, target in zip(degree, delete_target_mask):
        if target:
            delete_indices.append(offset + rng.integers(d))
        offset += d

    cut_nodes = np.full(n_node, -1)
    cut_nodes[delete_target_idx] = indices[1, delete_indices]
    cut_nodes = torch.tensor(cut_nodes).cuda()
    dummy_indices = np.delete(indices, delete_indices, axis=1)
    dummy_values = np.delete(values, delete_indices, axis=0)

    dummy_indices = torch.tensor(dummy_indices).cuda()
    dummy_values = torch.tensor(dummy_values).cuda()
    dummy_adj = torch.sparse_coo_tensor(
        dummy_indices, dummy_values, size=adj.size()
    ).to_sparse_csr()

    return dummy_adj, cut_nodes


def create_tail_adj(adj, degree, train_idx, k, rng):
    degree = torch.flatten(degree).cpu().numpy()
    dummy_degree = np.copy(degree)
    n_node = len(degree)
    adj = adj.coalesce()
    indices = copy.deepcopy(adj.indices())
    values = copy.deepcopy(adj.values())
    values = values.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    delete_indices = []
    offset = 0
    for i, d in enumerate(degree):
        if i in train_idx:
            if d == 0:
                continue
            num_links = rng.integers(min(k, d))
            num_links += 1
            idcs = rng.choice(d, num_links, replace=False)
            delete_idcs = np.delete(np.arange(d), idcs)
            delete_indices.extend(offset + delete_idcs)
            dummy_degree[i] = num_links
        offset += d

    dummy_indices = np.delete(indices, delete_indices, axis=1)
    dummy_values = np.delete(values, delete_indices, axis=0)

    dummy_indices = torch.tensor(dummy_indices).cuda()
    dummy_values = torch.tensor(dummy_values).cuda()
    dummy_adj = torch.sparse_coo_tensor(
        dummy_indices, dummy_values, size=adj.size()
    )
    dummy_degree = torch.tensor(dummy_degree).cuda()
    # dummy_adj = dummy_adj.to_sparse_csr()
    return dummy_adj.coalesce(), dummy_degree


def calc_cosine_sim(g, h):
    sim = torch.mm(g, h.T)
    norm_g = torch.norm(g, dim=1).unsqueeze(0)
    norm_h = torch.norm(h, dim=1).unsqueeze(0)
    norm_gh = torch.mm(norm_g.T, norm_h)
    sim = sim / norm_gh

    return sim


def indices_sort(indices, weights=None):
    indices[1], idx = torch.sort(indices[1])
    indices[0] = indices[0][idx]
    if weights is not None:
        weights = weights[idx]
    indices[0], idx = torch.sort(indices[0])
    indices[1] = indices[1][idx]
    if weights is not None:
        weights = weights[idx]

    if weights is None:
        return indices
    else:
        return indices, weights


def add_new_links(
    emb,
    adj_t,
    low_degree_group,
    use_sim_as_edge_weight,
    r,
    max_n_add_edge=None,
    sim_threshold=None,
):
    if max_n_add_edge is None and sim_threshold is None:
        print("either argument max_n_add_edge or sim_threshold must be specified")
        exit(1)
    size = adj_t.size()
    non_low_degree_group = list(set(np.arange(emb.shape[0])) - set(low_degree_group))
    low_degree_group = torch.tensor(low_degree_group).cuda()
    sim = calc_cosine_sim(emb[low_degree_group], emb[non_low_degree_group])
    sim, target = torch.sort(sim, dim=1, descending=True)

    sim = sim[:, :max_n_add_edge].flatten()
    source = (
        low_degree_group.unsqueeze(1).expand(-1, max_n_add_edge).flatten().unsqueeze(0)
    )
    target = target[:, :max_n_add_edge].flatten().unsqueeze(0)

    if sim_threshold is not None:
        sim_mask = sim >= sim_threshold
        sim = sim[sim_mask]
        source = source[:, sim_mask]
        target = target[:, sim_mask]

    if not use_sim_as_edge_weight:
        sim = torch.ones_like(sim)
    sim = r * sim
    # sim = torch.sigmoid(r) * sim
    new_link = torch.concatenate([source, target], dim=0)
    values = torch.ones_like(target).flatten().cuda()
    indices = torch.concatenate([adj_t.indices(), new_link], dim=1)
    edge_weights = torch.ones_like(adj_t.values()).cuda()
    edge_weights = torch.concatenate([edge_weights, sim], dim=0)
    indices, edge_weights = indices_sort(indices, edge_weights)
    values = torch.concatenate([adj_t.values(), values], dim=0)
    adj_t = torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    return adj_t, edge_weights


def add_edges(x, adj, target_idx, new_nodes):
    dummy_nodes = new_nodes[target_idx]
    n_dummy_nodes = dummy_nodes.shape[0]
    offset = x.shape[0]
    x = torch.cat([x, dummy_nodes], dim=0)
    dummy_idx = offset + torch.arange(n_dummy_nodes).cuda()
    aug_edges = torch.cat([target_idx[None, :], dummy_idx[None, :]], dim=0)
    adj = adj.coalesce()
    indices = torch.cat([adj.indices(), aug_edges], dim=1)
    values = torch.cat([adj.values(), torch.ones(n_dummy_nodes).cuda()])
    size = (offset + n_dummy_nodes, offset + n_dummy_nodes)
    adj = torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    return x, adj


def node_generation_loss(truth_emb, predicted_emb):
    return -torch.mean(torch.cosine_similarity(truth_emb, predicted_emb))


def add_edge(adj, srcs, score, n_add_edge: int):
    size = adj.size()
    if not isinstance(srcs, torch.Tensor):
        srcs = torch.tensor(srcs).cuda()
    _, indices = torch.sort(score, dim=1, descending=True)
    pbar = tqdm.tqdm(indices)
    pbar.set_description("adding edge")
    srcs = []
    drts = []
    # TODO: 高速化（ソートして探索してやれば速くなるはず）
    for i, idc in enumerate(pbar):
        add_cnt = 0
        neighbors = adj[i].coalesce().indices()
        for idx in idc:
            if add_cnt >= n_add_edge:
                break
            if idx not in neighbors:
                srcs.append(i)
                drts.append(idx.item())
                add_cnt += 1
    srcs = torch.tensor(srcs).cuda().unsqueeze(0)
    drts = torch.tensor(drts).cuda().unsqueeze(0)
    new_link = torch.concatenate([srcs, drts], dim=0)
    values = torch.ones_like(srcs).flatten().cuda()
    indices = torch.concatenate([adj.indices(), new_link], dim=1)
    values = torch.concatenate([adj.values(), values], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    return adj

def calc_degree(adj):
    edge = adj.coalesce().indices()
    n_node = edge.max() + 1
    degrees = torch.zeros(n_node, dtype=torch.int64).cuda()  # degree for each node

    u, counts = torch.unique(edge[0], return_counts=True)
    degrees[u] = copy.deepcopy(counts)

    return degrees