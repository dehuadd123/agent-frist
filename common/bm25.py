"""BM25 稀疏检索（零依赖，Okapi BM25 实现）。"""
import math
from collections import Counter

from .tokenizer import tokenize


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.N else 0.0
        self.df = Counter()
        self.tf = []
        for d in self.docs:
            c = Counter(d)
            self.tf.append(c)
            for t in c:
                self.df[t] += 1

    def score(self, query, doc_idx):
        q = tokenize(query)
        d = self.docs[doc_idx]
        dl = len(d)
        s = 0.0
        for t in q:
            if t not in self.df:
                continue
            f = self.tf[doc_idx].get(t, 0)
            if f == 0:
                continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, top_k=None):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scores = scores[:top_k]
        return scores
