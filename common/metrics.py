"""检索质量指标：Recall@K / MRR / nDCG@K（单黄金标注场景）。"""
import math


def recall_at_k(ranked_ids, gold_id, k):
    """ranked_ids: 排序后的候选 id 列表（不含分数）。gold_id 命中 top-k 则 1.0。"""
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr(ranked_ids, gold_id):
    """倒数排名。未命中返回 0.0。"""
    for i, idx in enumerate(ranked_ids):
        if idx == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids, gold_id, k):
    """单黄金标注：DCG@k = 1/log2(rank+1)，IDCG=1。"""
    for i, idx in enumerate(ranked_ids[:k]):
        if idx == gold_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate(ranked_ids_list, gold_ids, ks=(1, 3, 5, 10)):
    """对多条 query 的检索结果做批量评估，返回各指标均值。"""
    n = len(gold_ids)
    if n == 0:
        return {}
    out = {"n": n, "mrr": 0.0}
    for k in ks:
        out[f"recall@{k}"] = 0.0
        out[f"ndcg@{k}"] = 0.0
    for ranked, gold in zip(ranked_ids_list, gold_ids):
        out["mrr"] += mrr(ranked, gold)
        for k in ks:
            out[f"recall@{k}"] += recall_at_k(ranked, gold, k)
            out[f"ndcg@{k}"] += ndcg_at_k(ranked, gold, k)
    out["mrr"] /= n
    for k in ks:
        out[f"recall@{k}"] /= n
        out[f"ndcg@{k}"] /= n
    return out
