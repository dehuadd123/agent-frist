"""混合检索：BM25 稀疏 + TF-IDF 稠密 + RRF 融合（零依赖）。"""
from common.bm25 import BM25
from common.vector import TfidfVectorizer, cosine


def rrf_fuse(rankings_list, k=60):
    """RRF（Reciprocal Rank Fusion）融合多个排序列表。

    rankings_list: 每个元素是 [(doc_id, score), ...] 的降序列表。
    返回 [(doc_id, rrf_score), ...] 降序。
    """
    scores = {}
    for rankings in rankings_list:
        for rank, (doc_id, _) in enumerate(rankings):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """三种检索模式：bm25 / dense / hybrid(rrf)。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self.vectorizer = TfidfVectorizer().fit(chunks)
        self.vecs = self.vectorizer.transform(chunks)

    def _bm25_rank(self, query, limit=None):
        return self.bm25.search(query, top_k=limit)

    def _dense_rank(self, query, limit=None):
        qv = self.vectorizer._vec(query)
        scored = sorted(
            ((i, cosine(qv, self.vecs[i])) for i in range(len(self.chunks))),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored if limit is None else scored[:limit]

    def search(self, query, mode="hybrid", top_k=10):
        if mode == "bm25":
            return self._bm25_rank(query, limit=top_k)
        if mode == "dense":
            return self._dense_rank(query, limit=top_k)
        if mode == "hybrid":
            bm = self._bm25_rank(query, limit=50)
            de = self._dense_rank(query, limit=50)
            fused = rrf_fuse([bm, de])
            return fused[:top_k]
        raise ValueError(f"unknown mode: {mode}")
