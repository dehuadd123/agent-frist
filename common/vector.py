"""TF-IDF 向量化 + 余弦相似度（零依赖，向量已 L2 归一化）。"""
import math
from collections import Counter

from .tokenizer import tokenize


class TfidfVectorizer:
    def __init__(self):
        self.idf = {}
        self.vocab = {}

    def fit(self, docs):
        df = Counter()
        n = len(docs)
        for d in docs:
            for t in set(tokenize(d)):
                df[t] += 1
        # 平滑 idf
        self.idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
        self.vocab = {t: i for i, t in enumerate(self.idf)}
        return self

    def _vec(self, text):
        tf = Counter(tokenize(text))
        vec = [0.0] * len(self.vocab)
        norm = 0.0
        for t, c in tf.items():
            if t in self.idf:
                i = self.vocab[t]
                w = c * self.idf[t]
                vec[i] = w
                norm += w * w
        norm = math.sqrt(norm) or 1.0
        return [v / norm for v in vec]

    def transform(self, docs):
        return [self._vec(d) for d in docs]

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


def cosine(a, b):
    """已归一化向量的余弦相似度（即点积）。"""
    return sum(x * y for x, y in zip(a, b))
