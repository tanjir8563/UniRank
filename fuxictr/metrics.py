import numpy as np
from collections import OrderedDict
from sklearn.metrics import log_loss, roc_auc_score

def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred).astype(np.float64)

    return_dict = OrderedDict()
    group_metrics = []

    for metric in metrics:
        if metric in ["logloss", "binary_crossentropy"]:
            return_dict[metric] = float(log_loss(y_true, y_pred))
        elif metric == "AUC":
            return_dict[metric] = _auc(y_true, y_pred)
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))

    if group_metrics:
        assert group_id is not None, "group_index is required."
        group_res = _evaluate_group_metrics(y_true, y_pred, group_id, group_metrics)
        for m in group_metrics:
            return_dict[m] = float(group_res[m])

    return return_dict


def avgAUC(y_true, y_pred):
    """avgAUC used in MIND news recommendation."""
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    yb = (y_true > 0).astype(np.int8)
    pos = int(yb.sum())
    neg = len(yb) - pos
    if pos > 0 and neg > 0:
        auc = _binary_auc_rank(yb, y_pred)
        return (auc, 1)
    return (0, 0)


def gAUC(y_true, y_pred):
    """gAUC defined in DIN paper."""
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    yb = (y_true > 0).astype(np.int8)
    pos = int(yb.sum())
    neg = len(yb) - pos
    if pos > 0 and neg > 0:
        auc = _binary_auc_rank(yb, y_pred)
        n = len(yb)
        return (auc * n, n)
    return (0, 0)

def _to_1d(x):
    return np.asarray(x).reshape(-1)


def _binary_auc_rank(y_true, y_pred):
    """
    Fast AUC for binary labels using rank statistics (Mann-Whitney U),
    with tie handling via average ranks.
    """
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred).astype(np.float64)

    # map labels to {0,1}
    y = (y_true > 0).astype(np.int8)
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present in y_true. ROC AUC is undefined.")

    # stable sort by score
    order = np.argsort(y_pred, kind="mergesort")
    y_sorted = y[order]
    s_sorted = y_pred[order]

    # tie blocks
    starts = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1]
    ends = np.r_[starts[1:], n]  # exclusive

    # positives per block
    pos_counts = np.add.reduceat(y_sorted, starts).astype(np.float64)

    # average rank (1-based) for each tie block: (start+1 + end)/2
    avg_ranks = (starts + 1 + ends) * 0.5

    sum_pos_ranks = np.dot(pos_counts, avg_ranks)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg)
    return float(auc)


def _auc(y_true, y_pred):
    """Binary fast path; fallback to sklearn for non-binary labels."""
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    uniq = np.unique(y_true)
    if uniq.size <= 2:
        return _binary_auc_rank(y_true, y_pred)
    return float(roc_auc_score(y_true, y_pred))


def MRR(y_true, y_pred):
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    rr = y_true_sorted / (np.arange(len(y_true_sorted)) + 1)
    return float(np.sum(rr) / (np.sum(y_true_sorted) + 1e-12))


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = int(k)

    def dcg_score(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_topk = np.take(y_true, order[:self.topk])
        gains = 2 ** y_topk - 1
        discounts = np.log2(np.arange(len(y_topk)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return float(dcg / (idcg + 1e-12))


def _parse_ndcg(metric_name):
    """
    Support:
      - "NDCG"      -> k=1
      - "NDCG(5)"   -> k=5
    """
    if metric_name == "NDCG":
        return NDCG(k=1)
    if metric_name.startswith("NDCG(") and metric_name.endswith(")"):
        k = int(metric_name[5:-1])
        return NDCG(k=k)
    raise NotImplementedError("metrics={} not implemented.".format(metric_name))


def _evaluate_group_metrics(y_true, y_pred, group_id, group_metrics):
    """
    Compute grouped metrics via a single sort by group_id.
    """
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred).astype(np.float64)
    group_id = _to_1d(group_id)

    if not (len(y_true) == len(y_pred) == len(group_id)):
        raise ValueError("y_true, y_pred, group_id must have same length.")

    # sort once by group_id
    order = np.argsort(group_id, kind="mergesort")
    y_true = y_true[order]
    y_pred = y_pred[order]
    gid = group_id[order]

    n = len(gid)
    starts = np.r_[0, np.flatnonzero(gid[1:] != gid[:-1]) + 1]
    ends = np.r_[starts[1:], n]  # exclusive

    sums = {m: 0.0 for m in group_metrics}
    wts = {m: 0.0 for m in group_metrics}

    need_gauc = "gAUC" in group_metrics
    need_avgauc = "avgAUC" in group_metrics
    need_mrr = "MRR" in group_metrics
    ndcg_funcs = {m: _parse_ndcg(m) for m in group_metrics if m.startswith("NDCG")}

    for s, e in zip(starts, ends):
        yt = y_true[s:e]
        yp = y_pred[s:e]
        yb = (yt > 0).astype(np.int8)
        gsize = e - s

        pos = int(yb.sum())
        neg = gsize - pos
        valid_auc_group = (pos > 0 and neg > 0)

        if (need_gauc or need_avgauc) and valid_auc_group:
            auc = _binary_auc_rank(yb, yp)
            if need_gauc:
                sums["gAUC"] += auc * gsize
                wts["gAUC"] += gsize
            if need_avgauc:
                sums["avgAUC"] += auc
                wts["avgAUC"] += 1.0

        if need_mrr:
            sums["MRR"] += MRR(yb, yp)
            wts["MRR"] += 1.0

        for name, fn in ndcg_funcs.items():
            sums[name] += fn(yb, yp)
            wts[name] += 1.0

    return {m: (sums[m] / wts[m] if wts[m] > 0 else 0.0) for m in group_metrics}