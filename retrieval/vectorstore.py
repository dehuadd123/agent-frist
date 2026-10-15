"""真实向量库（零依赖实现）：稠密向量索引 + 余弦检索 + 落盘持久化 + 增量索引 + 变更检测。

设计要点（对标生产向量库的存储/检索/运维骨架，语义模型可插拔）：

- 嵌入（embed）：默认使用「确定性特征哈希」——对中文 unigram/bigram/trigram 与英文词
  做 signed feature hashing 得到 DIM 维稠密向量并 L2 归一化。该方案跨进程/跨重启结果稳定
  （不依赖 PYTHONHASHSEED），具备可用的词面/浅层语义相似度。
  * 生产可替换：构造时传入 embed_fn，对接 BGE / text-embedding-3 等真实 Embedding 模型。
- 索引（index）：扁平索引 + 余弦相似度（归一化向量即点积），内存常驻；records 携带
  doc_id / text / meta（含 title、owner、acl、scene）供检索后过滤与溯源。
- 持久化（persist）：save()/load() 把向量、文本、元数据、manifest 写入 JSON 文件，
  重启后 load 即恢复，无需重新嵌入。
- 增量索引（incremental）：add_document() 先做 content hash 变更检测——未变化直接跳过，
  变化则先 remove_document() 旧切片再 upsert，只对受影响文档重新嵌入，绝不全量重建。
- 变更检测（change detection）：manifest 记录每篇文档的 {hash, chunk_count, title, updated_at}。
"""
import hashlib
import json
import math
import os
import time

from common.tokenizer import tokenize

DIM = 256  # 默认向量维度（可插拔 embedding 可覆盖）


def content_hash(text):
    """内容指纹：对文档/切片集合做 SHA-256，用于变更检测。"""
    if isinstance(text, (list, tuple)):
        text = json.dumps(list(text), ensure_ascii=False)
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _ngrams(text):
    """产出用于嵌入的 token 序列：英文/数字词 + 中文 unigram/bigram/trigram。"""
    toks = list(tokenize(text))
    zh = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    # 中文 trigram 补充（tokenize 已含 unigram+bigram）
    for i in range(len(zh) - 2):
        toks.append(zh[i] + zh[i + 1] + zh[i + 2])
    return toks


def hash_embed(text, dim=DIM):
    """确定性特征哈希嵌入：token -> signed bucket，求和后 L2 归一化。

    用 md5 保证跨进程稳定（hash() 受 PYTHONHASHSEED 影响，不可用）。
    """
    vec = [0.0] * dim
    for tok in _ngrams(text):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]


def cosine(a, b):
    """余弦相似度（向量已 L2 归一化，即点积）。"""
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """扁平向量索引 + 持久化 + 增量/变更检测。"""

    def __init__(self, dim=DIM, embed_fn=None):
        self.dim = dim
        self.embed = embed_fn or hash_embed
        self.records = []          # [{id, doc_id, text, meta, vec}]
        self._pos = {}             # id -> records 下标
        self.manifest = {}         # doc_id -> {hash, chunk_count, title, updated_at}

    # ---------------- 写入 / 增量 ----------------
    def _append(self, doc_id, text, meta):
        rec_id = "%s#%d" % (doc_id, len(self.records))
        rec = {"id": rec_id, "doc_id": doc_id, "text": text,
               "meta": dict(meta or {}), "vec": self.embed(text)}
        self.records.append(rec)
        self._pos[rec_id] = len(self.records) - 1
        return rec_id

    def add_document(self, doc_id, title, chunks, meta=None):
        """增量索引一篇文档（变更检测）。

        返回 {changed, added, removed, doc_id}。content hash 未变 -> changed=False 跳过。
        """
        meta = dict(meta or {})
        meta["title"] = title
        new_hash = content_hash(chunks)
        old = self.manifest.get(doc_id)
        if old and old.get("hash") == new_hash:
            return {"changed": False, "added": 0, "removed": 0, "doc_id": doc_id}
        removed = self.remove_document(doc_id)
        for text in chunks:
            self._append(doc_id, text, meta)
        self.manifest[doc_id] = {
            "hash": new_hash, "chunk_count": len(chunks),
            "title": title, "updated_at": int(time.time()),
        }
        return {"changed": True, "added": len(chunks), "removed": removed, "doc_id": doc_id}

    def remove_document(self, doc_id):
        """删除一篇文档的全部切片，返回移除数量。"""
        before = len(self.records)
        self.records = [r for r in self.records if r["doc_id"] != doc_id]
        removed = before - len(self.records)
        self.manifest.pop(doc_id, None)
        self._reindex()
        return removed

    def _reindex(self):
        self._pos = {r["id"]: i for i, r in enumerate(self.records)}

    # ---------------- 检索 ----------------
    def search(self, query, k=10, filter_fn=None):
        """余弦相似度 top-k。filter_fn(rec) -> bool 用于 ACL / 场景过滤。"""
        qv = self.embed(query)
        scored = []
        for rec in self.records:
            if filter_fn is not None and not filter_fn(rec):
                continue
            scored.append((cosine(qv, rec["vec"]), rec))
        scored.sort(key=lambda x: -x[0])
        return [{"id": r["id"], "doc_id": r["doc_id"], "text": r["text"],
                 "meta": r["meta"], "score": round(float(s), 6)}
                for s, r in scored[:k]]

    # ---------------- 持久化 ----------------
    def save(self, path):
        """落盘：向量 + 文本 + 元数据 + manifest。"""
        data = {"dim": self.dim, "manifest": self.manifest,
                "records": [{k: r[k] for k in ("id", "doc_id", "text", "meta", "vec")}
                            for r in self.records]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)  # 原子替换，避免写一半损坏

    def load(self, path):
        """加载落盘索引，恢复 _pos 与 manifest。"""
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data.get("dim", self.dim)
        self.records = data.get("records", [])
        self.manifest = data.get("manifest", {})
        self._reindex()
        return len(self.records)

    # ---------------- 观测 ----------------
    def stats(self):
        docs = len(self.manifest)
        chunks = len(self.records)
        dim = self.dim
        return {"docs": docs, "chunks": chunks, "dim": dim,
                "manifest": {k: {"chunk_count": v.get("chunk_count"),
                                 "updated_at": v.get("updated_at")}
                             for k, v in self.manifest.items()}}
