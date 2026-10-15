"""企业级 LLM 实操平台 - 统一服务（零依赖，纯标准库）。

整合任务二（RAG 知识底座）+ 任务三（Agent 决策引擎）为一个完整闭环：

  知识底座（RAG）           决策引擎（Agent）
  ├─ 5 种切片策略对比  ──┐
  ├─ 混合检索(BM25+稠密+RRF) ├──> 工具调用(12 工具 Function Calling)
  ├─ 质量基线(R@K/MRR/nDCG) ─┘     DAG 编排 / 熔断 / Saga / 幂等
  └─ 场景权限隔离                   电商退款·金融合规·办公审批·检索问答

核心闭环：Agent 的 search_reports 工具真实调用 RAG 混合检索（set_rag_search 注入），
实现「检索 -> 生成 -> 合规闸门 -> 交付」的企业级问答主线。

接口：
  GET  /                       -> 企业级工作台前端
  GET  /api/health             -> 健康检查
  GET  /api/overview           -> 平台总览（统计 + 状态）
  # RAG 知识底座
  GET  /api/rag/meta           -> 语料库/场景/QA 元信息
  POST /api/rag/chunking       -> 切片策略对比实验
  POST /api/rag/eval           -> 混合检索质量基线
  POST /api/rag/search         -> 实时混合检索（场景隔离+高亮）
  POST /api/rag/document       -> 文档关键位置预览（相关切片排序+高亮）
  # Agent 决策引擎
  GET  /api/agent/tools        -> 工具注册表（Function Calling 声明）
  POST /api/agent/run          -> 执行三场景 + 问答 Agent 工作流
  # 订单状态实时管理
  GET  /api/orders             -> 列出全部订单及实时状态
  POST /api/order/query        -> 按订单号查询订单
  POST /api/order/update       -> 实时更新订单状态（状态机校验 + 落盘）
  # AI 直连检索（DeepSeek 等主流 API，不依赖本地向量库）
  POST /api/llm/retrieve       -> AI 智能检索（query + 可选上下文 -> LLM 回答）
  POST /api/llm/file_retrieve  -> 本地文件检索（上传解析 -> 内容直连 LLM）
  POST /api/llm/chat           -> OpenAI 兼容通用转发
  # 业务功能（本地向量库打底 + 可选 LLM 增强）
  POST /api/biz/ecommerce      -> 电商导购：约束解析 + 商品检索 + 过滤排序
  POST /api/biz/finance_report -> 金融投研：财报指标提取 + 多季度趋势 + 溯源
  POST /api/biz/office         -> 企业办公：制度/报销规则检索 + 结构化回答
  # 企业治理
  GET  /api/audit              -> 审计日志（全量操作留痕）
"""
import json
import os
import sys
import io
import time
import base64
import contextlib
import uuid
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.metrics import evaluate
from chunking.chunk_strategies import STRATEGIES, chunk_structure
from retrieval.hybrid import HybridRetriever, rrf_fuse

import tools as _tools
import agents as _agents
import model_eng as _ml
import multiparse as _mp
import business as _biz
import auth as _auth

from retrieval.vectorstore import VectorStore
import acl as _acl
import review as _review
import taskqueue as _tq
import function_call as _fc
import contextvars

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(_HERE, "data", "synth_dataset.json")
SAMPLE_FILE = os.path.join(_HERE, "data", "sample_files.json")
HTML = os.path.join(_HERE, "console.html")
PERSISTED_DOCS = os.path.join(_HERE, "data", "persisted_docs.json")  # 上传文档持久化
VECTOR_STORE_FILE = os.path.join(_HERE, "data", "vector_store.json")  # 真实向量索引持久化

# 请求级上下文（当前登录用户 + 角色），供检索层做文档级 ACL 过滤
_CURRENT_ACTOR = contextvars.ContextVar("actor", default="anonymous")
_CURRENT_ROLE = contextvars.ContextVar("role", default="viewer")

SCENES = {"ecommerce": "电商导购", "finance": "金融投研", "office": "企业办公"}
SCENE_FEATURES = {
    "ecommerce": {"name": "电商导购", "feature": "图文混排 + 多模态检索"},
    "finance": {"name": "金融投研", "feature": "PDF 表格 + 合规溯源"},
    "office": {"name": "企业办公", "feature": "多格式 + 权限隔离"},
}

# ---------- 加载数据集 + 构建 RAG 知识底座 ----------
with open(DATASET, encoding="utf-8") as f:
    _DATA = json.load(f)
DOCS = _DATA["documents"]
QA_PAIRS = _DATA["qa_pairs"]

# 内置多格式模拟文档库（图文混排 / PDF 表格 / 多格式办公文档）
with open(SAMPLE_FILE, encoding="utf-8") as f:
    SAMPLES = json.load(f)["samples"]
IMPORTED = set()  # 已导入的模拟文档 file 名

CHUNKS, CHUNK_META = [], []
for _d in DOCS:
    for _c in chunk_structure(_d["content"]):
        CHUNKS.append(_c)
        CHUNK_META.append(_d["id"])

RETRIEVER = HybridRetriever(CHUNKS)
_DOC_BY_ID = {d["id"]: d for d in DOCS}
VSTORE = VectorStore()
VSTORE.load(VECTOR_STORE_FILE)  # 恢复上次向量索引，配合 manifest 实现跨重启增量索引
_BOOTING = True  # 启动加载阶段标志：期间 _persist_docs 不写盘，避免覆盖持久化文档


def _get_acl(doc):
    """返回文档 ACL：upload_ 文档默认 owner 私有（owner/admin/operator 可见），其余 public。"""
    if "acl" in doc:
        return doc["acl"]
    if doc["id"].startswith("upload_"):
        return _acl.make_acl(owner=doc.get("owner", ""), public=False, roles=["admin", "operator"])
    return _acl.make_acl(public=True)


def _visible_doc_ids(scene, actor, role):
    """返回 (actor, role) 在指定 scene 下可见的 doc_id 集合（ACL + 场景双重过滤）。"""
    ids = set()
    for d in DOCS:
        if scene not in ("", "all", None) and d.get("scene") != scene:
            continue
        if _acl.can_access(_get_acl(d), actor, role):
            ids.add(d["id"])
    return ids


# 把内置语料也纳入真实向量库（增量：manifest 已存在且 hash 未变则跳过）
for _d in DOCS:
    _cs = chunk_structure(_d["content"])
    VSTORE.add_document(_d["id"], _d.get("title", _d["id"]), _cs,
                        {"title": _d.get("title", _d["id"]), "scene": _d.get("scene", ""),
                         "acl": _get_acl(_d), "owner": _d.get("owner", "")})


def _add_document(doc):
    """把一篇文档解析切片后写入检索索引 + 增量写入真实向量库（支持 upsert + 变更检测）。

    - 首次：追加切片到 CHUNKS/CHUNK_META，VectorStore.add_document 做嵌入。
    - 同 doc_id 重复（内容变更）：先移除旧切片，再重新加入，只重索引该文档。
    用户上传的文档（id 以 upload_ 开头）实时落盘 persisted_docs.json，
    向量索引经「变更检测 + 增量索引」写入 vector_store.json，重启自动恢复。
    """
    global RETRIEVER
    doc_id = doc["id"]
    if doc_id in _DOC_BY_ID:
        # upsert：移除旧切片（仅清理 CHUNKS/CHUNK_META；向量库由 add_document 内部做变更检测）
        keep = [(c, m) for c, m in zip(CHUNKS, CHUNK_META) if m != doc_id]
        CHUNKS[:] = [c for c, _ in keep]
        CHUNK_META[:] = [m for _, m in keep]
        DOCS[:] = [d for d in DOCS if d["id"] != doc_id]
    DOCS.append(doc)
    new_chunks = chunk_structure(doc["content"])
    if not new_chunks:
        return 0
    CHUNKS.extend(new_chunks)
    CHUNK_META.extend([doc_id] * len(new_chunks))
    _DOC_BY_ID[doc_id] = doc
    RETRIEVER = HybridRetriever(CHUNKS)
    # 真实向量库：增量索引（content hash 变更检测，未变跳过，变更只重索引该文档）
    meta = {"title": doc.get("title", doc_id), "scene": doc.get("scene", ""),
            "acl": _get_acl(doc), "owner": doc.get("owner", "")}
    VSTORE.add_document(doc_id, meta["title"], new_chunks, meta)
    _persist_docs()
    return len(new_chunks)


def _persist_docs():
    """把用户上传的运行时文档 + 向量索引落盘（内置语料/业务数据本就在 JSON 中）。

    启动加载阶段（_BOOTING）跳过写盘，避免覆盖磁盘上已有的持久化数据。
    """
    if _BOOTING:
        return
    runtime = [d for d in DOCS if d["id"].startswith("upload_")]
    try:
        with open(PERSISTED_DOCS, "w", encoding="utf-8") as f:
            json.dump(runtime, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    try:
        VSTORE.save(VECTOR_STORE_FILE)  # 向量索引持久化
    except OSError:
        pass


def _load_persisted_docs():
    """启动时重新加载上次上传的文档，恢复索引。"""
    if not os.path.exists(PERSISTED_DOCS):
        return 0
    try:
        with open(PERSISTED_DOCS, encoding="utf-8") as f:
            docs = json.load(f)
    except Exception:
        return 0
    loaded = 0
    for d in docs:
        if d.get("id") and d["id"] not in _DOC_BY_ID:
            _add_document(d)
            loaded += 1
    return loaded


def _guess_scene(text):
    """根据文本关键词自动识别业务场景（兜底 office）。"""
    if any(k in text for k in ["退款", "订单", "购买", "商品", "优惠", "续航", "像素", "扫地", "手机", "耳机", "快充", "能效"]):
        return "ecommerce"
    if any(k in text for k in ["营收", "收入", "净利润", "毛利", "估值", "研报", "投资", "财报",
                               "增速", "同比", "季度", "利率", "渗透率", "市盈率"]):
        return "finance"
    return "office"


def _rag_search(query, scene, top_k=4):
    """RAG 混合检索（真实向量库 + BM25 + RRF，带文档级 ACL 与场景隔离）。

    返回 [{text,title,scene,score}]。dense 走 VectorStore（余弦），sparse 走 BM25，
    RRF 在 doc_id 粒度融合后映射回该文档最相关的切片。
    """
    actor = _CURRENT_ACTOR.get()
    role = _CURRENT_ROLE.get()
    visible = _visible_doc_ids(scene, actor, role)
    if not visible:
        return []
    dense = VSTORE.search(query, k=50, filter_fn=lambda rec: rec["doc_id"] in visible)
    dense_rank = [(r["doc_id"], r["score"]) for r in dense]
    bm = RETRIEVER._bm25_rank(query, limit=50)
    sparse_rank = [(CHUNK_META[i], s) for i, s in bm if CHUNK_META[i] in visible]
    fused = rrf_fuse([dense_rank, sparse_rank])
    seen, hits = set(), []
    for doc_id, _ in fused:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rec = next((r for r in dense if r["doc_id"] == doc_id), None)
        if rec is None:
            continue
        meta = rec["meta"]
        hits.append({
            "text": rec["text"], "title": meta.get("title", ""),
            "scene": meta.get("scene", ""), "score": round(float(rec["score"]), 4),
        })
        if len(hits) >= top_k:
            break
    return hits


# 把真实 RAG 检索注入 Agent 工具层（打通知识底座与决策引擎）
_tools.set_rag_search(_rag_search)
# 把真实 RAG 检索注入业务层（金融/办公溯源用）
_biz.set_rag_search(_rag_search)

# ---------- 构建各 Agent 单例（熔断/幂等状态跨请求保持） ----------
AGENT_ECOMMERCE = _agents.build_ecommerce_agent()
AGENT_FINANCE = _agents.build_finance_agent()
AGENT_OFFICE = _agents.build_office_agent()
AGENT_QA = _agents.build_qa_agent()

# ---------- 审计日志（企业治理：全量留痕） ----------
AUDIT = []  # [{ts, actor, action, detail, result}]
_AUDIT_LOCK = None


def audit(actor, action, detail, result="ok"):
    AUDIT.append({
        "ts": time.strftime("%H:%M:%S"),
        "actor": actor, "action": action, "detail": detail, "result": result,
    })
    if len(AUDIT) > 200:
        del AUDIT[: len(AUDIT) - 200]


def _highlight(text, keywords):
    """关键词高亮：对 query 或关键词列表中的每个词用 <mark> 包裹。

    - keywords 为 str 时：提取英文词 + 中文 2 字以上连续串作为关键词
    - keywords 为 list/tuple 时：直接对每个关键词做高亮
    """
    import re
    if isinstance(keywords, (list, tuple)):
        kws = [k for k in keywords if k and len(str(k)) >= 2]
    else:
        q = (keywords or "").lower()
        kws = [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", q) if len(t) >= 2]
    out = text
    for k in sorted(set(str(x) for x in kws), key=len, reverse=True):
        try:
            out = re.sub(re.escape(k), lambda m: "<mark>%s</mark>" % m.group(0),
                         out, flags=re.IGNORECASE)
        except re.error:
            continue
    return out


# 停用词表（关键词提取时过滤，纯标准库无分词依赖）
_STOPWORDS = set(
    """的 了 是 在 和 与 及 以及 并 或 到 从 为 对 把 等 们 这 那 有 个 都 也 就 而 但
我 你 他 她 它 我们 你们 他们 它们 一个 这个 那个 可以 进行 通过 实现 支持 使用 需要 根据
相关 内容 数据 分析 检索 场景 格式 文档 说明 文件 点击 运行 已经 当前 然后 并且 其中 如果
因为 所以 但是 而且 不过 由于 因此 作为 这些 那些 什么 怎么 如何 为什么""".split())


def _extract_keywords(text, top_n=10):
    """纯标准库关键词提取：英文 token + 中文 2~4 gram 词频统计，长词优先去重。

    用于上传文档后自动提取关键词，供「检索分析结果」展示与高亮。
    """
    import re
    import collections
    if not text:
        return []
    en = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text)
    zh_spans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    counter = collections.Counter()
    for t in en:
        t = t.lower()
        if len(t) >= 3:
            counter[t] += 1
    for span in zh_spans:
        for n in (4, 3, 2):
            for i in range(len(span) - n + 1):
                g = span[i:i + n]
                if g in _STOPWORDS:
                    continue
                if any(c in g for c in "的了着过是在和与及或到从为把等"):
                    continue
                counter[g] += 1
    # 长词优先去重：候选按 (长度降序, 频次降序)，贪心选取互不包含的词
    cand = sorted(counter.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)
    picked = []
    for w, _ in cand:
        if any(w in p for p in picked):
            continue
        picked.append(w)
        if len(picked) >= top_n:
            break
    return picked


# ---------- 启动预载：内置模拟文档库全量导入（开箱即用，无需手动点导入） ----------
def _preload_samples():
    for s in SAMPLES:
        if s["file"] in IMPORTED:
            continue
        parsed = _mp.parse_text(s["file"], s["content"], ext="." + s["format"])
        doc = {"id": "sample_" + s["file"], "title": s["file"], "scene": s["scene"],
               "format": s["format"], "content": parsed["text"]}
        n = _add_document(doc)
        IMPORTED.add(s["file"])
        audit("system", "启动预载", f"{s['file']} [{s['format']}] -> {n} 切片")


_preload_samples()

# 启动时恢复上次上传的文档（向量/索引持久化）
_PERSISTED_LOADED = _load_persisted_docs()
_BOOTING = False  # 启动完成，之后上传文档正常落盘
VSTORE.save(VECTOR_STORE_FILE)  # 启动完成后落盘向量索引（增量/变更检测结果）
if _PERSISTED_LOADED:
    audit("system", "恢复持久化文档", f"重新加载 {_PERSISTED_LOADED} 篇上传文档")


def _run_chunking():
    TOP_K = 10
    results = []
    for name, (func, desc) in STRATEGIES.items():
        chunks_by_doc = {d["id"]: func(d["content"]) for d in DOCS}
        ranked_list, gold_list = [], []
        for qa in QA_PAIRS:
            doc_chunks = chunks_by_doc[qa["doc_id"]]
            from common.bm25 import BM25
            bm25 = BM25(doc_chunks)
            ranked = [idx for idx, _ in bm25.search(qa["query"], top_k=TOP_K)]
            gold_idx = next((i for i, c in enumerate(doc_chunks) if qa["gold_fact"] in c), -1)
            ranked_list.append(ranked)
            gold_list.append(gold_idx)
        m = evaluate(ranked_list, gold_list, ks=(1, 3, 5, 10))
        avg_chunks = sum(len(v) for v in chunks_by_doc.values()) / len(chunks_by_doc)
        integrity = 1.0 - sum(1 for g in gold_list if g == -1) / len(gold_list)
        results.append({
            "strategy": name, "desc": desc,
            "recall@1": round(m["recall@1"], 3), "recall@3": round(m["recall@3"], 3),
            "recall@5": round(m["recall@5"], 3), "recall@10": round(m["recall@10"], 3),
            "mrr": round(m["mrr"], 3), "ndcg@5": round(m["ndcg@5"], 3),
            "integrity": round(integrity, 3), "avg_chunks_per_doc": round(avg_chunks, 2),
        })
    return results


def _run_eval():
    TOP_K = 10
    results = []
    for mode in ["bm25", "dense", "hybrid"]:
        ranked_list, gold_list = [], []
        for qa in QA_PAIRS:
            ranked = [cid for cid, _ in RETRIEVER.search(qa["query"], mode=mode, top_k=TOP_K)]
            gold = next((i for i, c in enumerate(CHUNKS) if qa["gold_fact"] in c), -1)
            ranked_list.append(ranked)
            gold_list.append(gold)
        m = evaluate(ranked_list, gold_list, ks=(1, 3, 5, 10))
        results.append({
            "mode": mode,
            "recall@1": round(m["recall@1"], 3), "recall@3": round(m["recall@3"], 3),
            "recall@5": round(m["recall@5"], 3), "recall@10": round(m["recall@10"], 3),
            "mrr": round(m["mrr"], 3), "ndcg@5": round(m["ndcg@5"], 3),
        })
    return results


def _run_search(query, scene, mode, top_k):
    actor = _CURRENT_ACTOR.get()
    role = _CURRENT_ROLE.get()
    visible = _visible_doc_ids(scene, actor, role)
    allowed_idx = [i for i, doc_id in enumerate(CHUNK_META) if doc_id in visible]

    if mode == "dense":
        # 真实向量库路径：余弦检索 + ACL 过滤
        records = VSTORE.search(query, k=top_k, filter_fn=lambda rec: rec["doc_id"] in visible)
        hits = []
        for r in records:
            meta = r["meta"]
            hits.append({
                "chunk_id": r["id"], "doc_id": r["doc_id"],
                "title": meta.get("title", ""), "scene": meta.get("scene", ""),
                "scene_name": SCENES.get(meta.get("scene", ""), meta.get("scene", "")),
                "score": round(float(r["score"]), 4), "text": r["text"],
                "highlighted": _highlight(r["text"], query),
            })
        return {
            "query": query, "scene": scene or "all", "mode": mode, "top_k": top_k,
            "hits": hits, "total_corpus_chunks": len(CHUNKS),
            "filtered_chunks": len(allowed_idx), "backend": "vectorstore",
        }

    if mode == "hybrid":
        bm = RETRIEVER._bm25_rank(query, limit=50)
        de = RETRIEVER._dense_rank(query, limit=50)
        fused = rrf_fuse([[(i, s) for i, s in bm if i in allowed_idx],
                          [(i, s) for i, s in de if i in allowed_idx]])
        ranked = fused[:top_k]
    else:
        full = RETRIEVER.search(query, mode=mode, top_k=50)
        ranked = [(i, s) for i, s in full if i in allowed_idx][:top_k]
    hits = []
    for i, score in ranked:
        doc = _DOC_BY_ID[CHUNK_META[i]]
        hits.append({
            "chunk_id": i, "doc_id": doc["id"], "title": doc["title"],
            "scene": doc["scene"], "scene_name": SCENES.get(doc["scene"], doc["scene"]),
            "score": round(float(score), 4), "text": CHUNKS[i],
            "highlighted": _highlight(CHUNKS[i], query),
        })
    return {
        "query": query, "scene": scene or "all", "mode": mode, "top_k": top_k,
        "hits": hits, "total_corpus_chunks": len(CHUNKS),
        "filtered_chunks": len(allowed_idx), "backend": "hybrid",
    }


def _run_doc_preview(doc_id, query):
    """按文档取关键位置预览：与 query 相关的切片按相关度排序，高亮命中词；无 query 按原文顺序。"""
    doc = _DOC_BY_ID.get(doc_id)
    if not doc:
        return None
    actor = _CURRENT_ACTOR.get()
    role = _CURRENT_ROLE.get()
    if not _acl.can_access(_get_acl(doc), actor, role):
        return None  # 无文档访问权限，等同不存在
    idxs = [i for i, d in enumerate(CHUNK_META) if d == doc_id]
    positions = []
    if query:
        bm = dict(RETRIEVER._bm25_rank(query, limit=len(CHUNKS)))
        de = dict(RETRIEVER._dense_rank(query, limit=len(CHUNKS)))
        ranked = sorted(((i, max(bm.get(i, 0.0), de.get(i, 0.0))) for i in idxs),
                        key=lambda x: -x[1])
        for i, s in ranked:
            positions.append({
                "chunk_id": i, "score": round(float(s), 4), "relevant": s > 0,
                "highlighted": _highlight(CHUNKS[i], query),
            })
        n_rel = sum(1 for p in positions if p["relevant"])
        if n_rel:  # 相关切片排前面，其余按原文顺序垫底
            positions.sort(key=lambda p: (not p["relevant"], -p["score"]))
    else:
        for i in idxs:
            positions.append({"chunk_id": i, "score": 0.0, "relevant": True,
                              "highlighted": CHUNKS[i]})
    return {
        "doc_id": doc_id, "title": doc["title"], "scene": doc["scene"],
        "scene_name": SCENES.get(doc["scene"], doc["scene"]),
        "format": doc.get("format", "txt"),
        "total_chunks": len(idxs), "positions": positions,
    }


# ---------- LLM API 直连（DeepSeek 等 OpenAI 兼容接口） ----------
DEFAULT_LLM_BASE = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
MAX_CONTEXT_CHARS = 12000  # 本地文件上下文上限，避免超 token

# 密钥托管：服务端环境变量 DEEPSEEK_API_KEY 优先（生产推荐），
# 未设置时回退到前端传入的 api_key（本地开发兼容，仍不落盘）。
SERVER_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _resolve_api_key(req):
    """解析 API Key：服务端托管优先，否则用前端传入。"""
    return (req.get("api_key") or "").strip() or SERVER_API_KEY


def _llm_chat(api_key, base_url, model, messages, temperature=0.3, max_tokens=None, tools=None):
    """调用 OpenAI 兼容 chat/completions 接口，返回 (content, err, tool_calls)。

    支持 tools 参数（真实 Function Calling）：模型返回 tool_calls 时原样透出，
    由 function_call.run_agent_loop 负责执行并回填。API Key 由前端传入，不落盘。
    """
    if not api_key:
        return None, "未配置 API Key，请先在「API 配置」面板填写", []
    base = (base_url or DEFAULT_LLM_BASE).strip().rstrip("/")
    url = base + "/chat/completions"
    payload = {
        "model": model or DEFAULT_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key.strip(),
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
            msg = detail.get("error", {}).get("message", "") if isinstance(detail.get("error"), dict) else str(detail)
        except Exception:
            msg = ""
        return None, "API 调用失败 HTTP %s：%s" % (e.code, msg or e.reason), []
    except Exception as e:
        return None, "网络错误：%s" % e, []
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return None, "API 返回结构异常：%s" % json.dumps(data, ensure_ascii=False)[:200], []
    return content, None, tool_calls


def _llm_retrieve(api_key, base_url, model, query, context=None, scene=None):
    """AI 直连检索：query + 本地文件上下文 -> 拼 prompt -> 调 LLM -> 返回 answer。"""
    sys_prompt = ("你是企业级智能检索助手，请基于提供的资料严谨、准确地回答用户问题。"
                  "回答要点：1) 优先引用资料中的事实，必要时标注来源；2) 资料不足时如实说明，绝不编造数据；"
                  "3) 涉及金融/投资信息时注明仅供参考、不构成投资建议；4) 用中文作答，条理清晰。")
    user_parts = []
    if context:
        ctx = context.strip()
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS] + "\n…（内容过长已截断）"
        user_parts.append("参考资料：\n" + ctx)
    user_parts.append("问题：" + query)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "\n\n".join(user_parts)}]
    answer, err, _ = _llm_chat(api_key, base_url, model, messages)
    return {"ok": err is None, "answer": answer or "", "error": err,
            "scene": scene or _guess_scene(query), "has_context": bool(context)}


def _make_llm(req):
    """从请求体构建业务层可用的 LLM 调用器（llm(messages) -> (content, err)）。

    密钥托管：服务端 DEEPSEEK_API_KEY 优先；两者皆无时返回 None，
    业务层退化为纯本地向量库检索（混合模式）。
    """
    api_key = _resolve_api_key(req)
    if not api_key:
        return None
    base_url = req.get("base_url", DEFAULT_LLM_BASE)
    model = req.get("model", DEFAULT_LLM_MODEL)

    def _call(messages):
        content, err, _ = _llm_chat(api_key, base_url, model, messages)
        return content, err

    return _call


class Handler(BaseHTTPRequestHandler):
    def _extract_token(self, parsed):
        """从多来源提取 JWT：自定义头 X-Auth-Token > ?token= > Authorization。

        重要：托管平台网关会覆盖 Authorization 头（注入平台自身 JWT），
        故业务 Token 必须走平台不干涉的 X-Auth-Token 头或 query 参数。
        """
        xt = self.headers.get("X-Auth-Token", "")
        if xt:
            return xt.strip()
        qs = parse_qs(parsed.query)
        t = (qs.get("token") or [""])[0].strip()
        if t:
            return t
        authz = self.headers.get("Authorization", "")
        if authz:
            if authz.startswith("Bearer "):
                return authz[7:].strip()
            if authz.startswith("Bearer"):
                return authz[6:].strip()
        return ""

    def _gate(self, method):
        """安全网关：限流 -> 鉴权 -> RBAC。返回 False 表示已拦截（响应已写出）。"""
        parsed = urlparse(self.path)
        p = parsed.path
        ip = self.client_address[0]
        self._actor = "anonymous"
        # 每次请求先重置请求级上下文，避免串到上一个请求的登录态
        _CURRENT_ACTOR.set("anonymous")
        _CURRENT_ROLE.set("viewer")

        # 1) 限流（按 IP + 接口分级）
        ok, limit = _auth.check_rate(ip, p)
        if not ok:
            self._json({"error": "请求过于频繁，请稍后再试",
                        "code": "rate_limited", "limit_per_min": limit}, 429)
            return False

        # 2) 鉴权 + RBAC
        level = _auth.route_level(method, p)
        if level == 0:
            return True
        token = self._extract_token(parsed)
        payload = _auth.verify_token(token) if token else None
        if not payload:
            self._json({"error": "未登录或登录已过期", "code": "unauthorized",
                        "hint": "请登录后携带 Bearer Token 访问"}, 401)
            return False
        self._actor = payload.get("sub", "user")
        self._role = payload.get("role", "viewer")
        _CURRENT_ACTOR.set(self._actor)
        _CURRENT_ROLE.set(self._role)
        if _auth.ROLE_LEVEL.get(self._role, 0) < level:
            self._json({"error": "权限不足", "code": "forbidden",
                        "required_role": level}, 403)
            return False
        return True

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        if not os.path.exists(HTML):
            self._json({"error": "console.html not found"}, 500)
            return
        with open(HTML, encoding="utf-8") as f:
            html = f.read().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path
        if not self._gate("GET"):
            return
        if p in ("/", "/index.html", "/console", "/enterprise"):
            self._serve_html()
        elif p == "/api/health":
            self._json({"ok": True, "service": "llm-enterprise-platform",
                        "docs": len(DOCS), "chunks": len(CHUNKS),
                        "qa": len(QA_PAIRS), "tools": len(_tools.REGISTRY._tools)})
        elif p == "/api/auth/me":
            u = _auth.get_user(self._actor)
            self._json({"ok": True, "user": u,
                        "role_name": _auth.ROLE_NAMES.get(u["role"]) if u else None})
        elif p == "/api/auth/users":
            self._json({"ok": True, "users": _auth.list_users(),
                        "role_names": _auth.ROLE_NAMES})
        elif p == "/api/overview":
            self._json({
                "scenes": [{"key": k, "name": v["name"], "feature": v["feature"]}
                           for k, v in SCENE_FEATURES.items()],
                "stats": {"docs": len(DOCS), "chunks": len(CHUNKS), "qa": len(QA_PAIRS),
                          "tools": len(_tools.REGISTRY._tools)},
                "breaker": AGENT_FINANCE.breaker.state,
                "audit_count": len(AUDIT),
            })
        elif p == "/api/rag/meta":
            self._json({
                "scenes": [{"key": k, "name": v["name"], "feature": v["feature"]}
                           for k, v in SCENE_FEATURES.items()],
                "documents": [{"id": d["id"], "title": d["title"], "scene": d["scene"],
                               "scene_name": SCENES.get(d["scene"], d["scene"]),
                               "format": d.get("format", "内置")}
                              for d in DOCS],
                "qa_count": len(QA_PAIRS), "chunk_count": len(CHUNKS),
                "chunk_strategies": [{"name": k, "desc": v[1]} for k, v in STRATEGIES.items()],
                "sample_count": len(SAMPLES), "imported_count": len(IMPORTED),
            })
        elif p == "/api/rag/samples":
            self._json({
                "samples": [{
                    "file": s["file"], "scene": s["scene"], "format": s["format"],
                    "note": s["note"], "scene_name": SCENES.get(s["scene"], s["scene"]),
                    "imported": s["file"] in IMPORTED,
                    "chars": len(s["content"]),
                } for s in SAMPLES],
            })
        elif p == "/api/agent/tools":
            self._json({
                "tools": _tools.REGISTRY.to_openai_tools(),
                "table": _tools.REGISTRY.table(),
            })
        elif p == "/api/orders":
            self._json({"ok": True, **_tools.tool_list_orders()})
        elif p == "/api/audit":
            self._json({"audit": list(reversed(AUDIT))})
        elif p == "/api/ml/meta":
            self._json({
                "model_versions": _ml.MODEL_VERSIONS,
                "red_lines": _ml.COMPLIANCE_RED_LINES,
                "levels": _ml.LEVEL_META,
                "tool_levels": _ml.TOOL_LEVELS,
                "levers": _ml.ACCURACY_LEVERS,
                "latency_strategies": _ml.LATENCY_STRATEGIES,
            })
        elif p == "/api/ml/observability":
            self._json({"ok": True, **_ml.OBS.dashboard()})
        elif p == "/api/vector/stats":
            self._json({"ok": True, "vectorstore": VSTORE.stats(),
                        "backend": "flat-cosine", "persisted": VECTOR_STORE_FILE})
        elif p == "/api/acl/docs":
            actor = _CURRENT_ACTOR.get()
            role = _CURRENT_ROLE.get()
            rows = []
            for d in DOCS:
                acl = _get_acl(d)
                rows.append({
                    "doc_id": d["id"], "title": d["title"], "scene": d["scene"],
                    "owner": acl.get("owner", ""),
                    "public": acl.get("public", False),
                    "roles": acl.get("roles", []),
                    "accessible": _acl.can_access(acl, actor, role),
                    "chunks": sum(1 for i in CHUNK_META if i == d["id"]),
                })
            self._json({"ok": True, "actor": actor, "role": role, "docs": rows})
        elif p == "/api/review/stats":
            self._json({"ok": True, **_review.QUEUE.stats()})
        elif p == "/api/review/queue":
            qs = parse_qs(parsed.query)
            status = (qs.get("status") or [""])[0].strip() or None
            self._json({"ok": True, "status": status or "all",
                        "items": _review.QUEUE.list(status)})
        elif p == "/api/tasks/stats":
            self._json({"ok": True, **_tq.QUEUE.stats()})
        elif p == "/api/tasks":
            qs = parse_qs(parsed.query)
            status = (qs.get("status") or [""])[0].strip() or None
            self._json({"ok": True, "tasks": _tq.QUEUE.list(status)})
        elif p.startswith("/api/tasks/"):
            task_id = p[len("/api/tasks/"):].strip()
            t = _tq.QUEUE.get(task_id)
            if not t:
                self._json({"error": "任务不存在"}, 404)
            else:
                self._json({"ok": True, **t})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self._gate("POST"):
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._json({"error": "请求体需为 JSON"}, 400)
            return

        p = parsed.path
        if p == "/api/auth/login":
            self._handle_login(req)
        elif p == "/api/auth/users":
            self._handle_create_user(req)
        elif p == "/api/rag/chunking":
            r = _run_chunking()
            audit("system", "切片对比实验", f"{len(r)} 策略 × {len(QA_PAIRS)} QA")
            self._json({"ok": True, "results": r, "qa_count": len(QA_PAIRS)})
        elif p == "/api/rag/eval":
            r = _run_eval()
            audit("system", "质量基线", f"3 方案 × {len(QA_PAIRS)} QA")
            self._json({"ok": True, "results": r, "qa_count": len(QA_PAIRS),
                        "corpus_chunks": len(CHUNKS)})
        elif p == "/api/rag/search":
            query = (req.get("query") or "").strip()
            if not query:
                self._json({"error": "query 不能为空"}, 400)
                return
            scene = req.get("scene", "") or "all"
            mode = req.get("mode", "hybrid")
            try:
                top_k = int(req.get("top_k", 5))
            except (TypeError, ValueError):
                top_k = 5
            audit(self._actor, "检索", f"{query} @{scene}/{mode}")
            self._json({"ok": True, **_run_search(query, scene, mode, top_k)})
        elif p == "/api/rag/document":
            doc_id = (req.get("doc_id") or "").strip()
            query = (req.get("query") or "").strip()
            r = _run_doc_preview(doc_id, query)
            if r is None:
                self._json({"error": "文档不存在"}, 404)
                return
            audit(self._actor, "预览文档", f"{r['title']} · query={query or '-'}")
            self._json({"ok": True, **r})
        elif p == "/api/rag/import_sample":
            fname = (req.get("file") or "").strip()
            sample = next((s for s in SAMPLES if s["file"] == fname), None)
            if not sample:
                self._json({"error": "模拟文档不存在"}, 404)
                return
            if fname in IMPORTED:
                self._json({"ok": True, "already": True, "file": fname,
                            "message": "该文档已导入，跳过"})
                return
            parsed = _mp.parse_text(fname, sample["content"], ext="." + sample["format"])
            doc = {"id": "sample_" + fname, "title": fname, "scene": sample["scene"],
                   "format": sample["format"], "content": parsed["text"]}
            n = _add_document(doc)
            IMPORTED.add(fname)
            audit(self._actor, "导入模拟文档", f"{fname} [{sample['format']}] -> {n} 切片")
            self._json({"ok": True, "file": fname, "format": sample["format"],
                        "note": sample["note"], "scene": sample["scene"],
                        "chunks_added": n, "total_chunks": len(CHUNKS)})
        elif p == "/api/rag/upload":
            filename = (req.get("filename") or "").strip()
            data_b64 = req.get("data_base64", "")
            scene = req.get("scene", "auto")
            if not data_b64:
                self._json({"error": "缺少文件内容"}, 400)
                return
            try:
                raw = base64.b64decode(data_b64)
            except Exception:
                self._json({"error": "base64 解码失败"}, 400)
                return
            parsed = _mp.parse_bytes(filename, raw)
            text = parsed["text"].strip()
            if not text:
                self._json({"ok": False, "error": "未能从文件中解析出文本内容"})
                return
            if scene in ("", "auto", "all", None):
                scene = _guess_scene(text)
            owner = self._actor if self._actor != "anonymous" else "admin"
            # 支持 upsert：传 doc_id 则对同文档做变更检测（内容变才重索引），否则新建
            doc_id = (req.get("doc_id") or "").strip()
            if not (doc_id.startswith("upload_") and doc_id in _DOC_BY_ID):
                doc_id = "upload_" + uuid.uuid4().hex[:8]
            doc = {"id": doc_id, "title": filename or doc_id,
                   "scene": scene, "format": parsed["ext"].lstrip("."),
                   "content": text, "owner": owner,
                   "acl": _acl.make_acl(owner=owner, public=False, roles=["admin", "operator"])}

            def _do_index(doc, text, filename, parsed, scene):
                n = _add_document(doc)
                keywords = _extract_keywords(text)
                return {"ok": True, "doc_id": doc["id"], "filename": filename,
                        "format": parsed["format"], "note": parsed["note"],
                        "scene": scene, "chunks_added": n,
                        "total_chunks": len(CHUNKS), "keywords": keywords,
                        "highlighted_preview": _highlight(text[:400], keywords),
                        "preview": text[:200]}

            if req.get("async"):
                # 异步任务队列：索引在后台 worker 执行，立即返回 task_id
                t = _tq.QUEUE.submit("文档索引:%s" % (filename or doc["id"]),
                                      _do_index, doc, text, filename, parsed, scene)
                audit(self._actor, "上传文档(异步)", f"{filename} -> {t['task_id']}")
                self._json({"ok": True, "async": True, "task_id": t["task_id"],
                            "doc_id": doc["id"]})
                return
            audit(self._actor, "上传文档", f"{filename} [{parsed['format']}] @{scene}")
            self._json(_do_index(doc, text, filename, parsed, scene))
        elif p == "/api/agent/run":
            if req.get("async"):
                scenario = req.get("scenario", "")
                params = req.get("params", {}) or {}
                actor = self._actor

                def _do_run():
                    return self._run_agent_sync(scenario, params, actor)

                t = _tq.QUEUE.submit("Agent执行:%s" % scenario, _do_run)
                audit(self._actor, "Agent执行(异步)", f"{scenario} -> {t['task_id']}")
                self._json({"ok": True, "async": True, "task_id": t["task_id"],
                            "scenario": scenario})
                return
            self._handle_agent_run(req)
        elif p == "/api/order/update":
            order_id = (req.get("order_id") or "").strip()
            new_status = (req.get("status") or "").strip()
            if not order_id or not new_status:
                self._json({"error": "order_id 与 status 不能为空"}, 400)
                return
            r = _tools.tool_update_order_status(order_id, new_status)
            audit(self._actor, "订单状态更新", f"{order_id} -> {new_status}",
                  result="ok" if r.get("ok") else "error")
            self._json(r)
        elif p == "/api/order/query":
            order_id = (req.get("order_id") or "").strip()
            if not order_id:
                self._json({"error": "order_id 不能为空"}, 400)
                return
            order = _tools.tool_query_order(order_id)
            found = order.get("status") != "not_found"
            self._json({"ok": found, "order": order})
        elif p == "/api/llm/retrieve":
            api_key = _resolve_api_key(req)
            base_url = req.get("base_url", DEFAULT_LLM_BASE)
            model = req.get("model", DEFAULT_LLM_MODEL)
            query = (req.get("query") or "").strip()
            if not query:
                self._json({"error": "query 不能为空"}, 400)
                return
            context = req.get("context", "")
            scene = req.get("scene", "")
            audit(self._actor, "AI 检索", f"{query} · 直连 {model} · 上下文 {len(context)} 字符")
            r = _llm_retrieve(api_key, base_url, model, query, context, scene)
            self._json(r)
        elif p == "/api/llm/file_retrieve":
            api_key = _resolve_api_key(req)
            base_url = req.get("base_url", DEFAULT_LLM_BASE)
            model = req.get("model", DEFAULT_LLM_MODEL)
            query = (req.get("query") or "").strip()
            filename = (req.get("filename") or "").strip()
            data_b64 = req.get("data_base64", "")
            if not data_b64:
                self._json({"error": "缺少文件内容"}, 400)
                return
            try:
                raw = base64.b64decode(data_b64)
            except Exception:
                self._json({"error": "base64 解码失败"}, 400)
                return
            parsed = _mp.parse_bytes(filename, raw)
            text = parsed["text"].strip()
            if not text:
                self._json({"ok": False, "error": "未能从文件中解析出文本内容"})
                return
            audit(self._actor, "AI 文件检索", f"{filename} [{parsed['format']}] · 直连 {model}")
            r = _llm_retrieve(api_key, base_url, model, query, text,
                              req.get("scene", "") or _guess_scene(text))
            r["filename"] = filename
            r["format"] = parsed["format"]
            r["note"] = parsed["note"]
            r["context_chars"] = len(text)
            self._json(r)
        elif p == "/api/llm/chat":
            api_key = _resolve_api_key(req)
            base_url = req.get("base_url", DEFAULT_LLM_BASE)
            model = req.get("model", DEFAULT_LLM_MODEL)
            messages = req.get("messages") or []
            if not isinstance(messages, list) or not messages:
                self._json({"error": "messages 不能为空"}, 400)
                return
            content, err, _ = _llm_chat(api_key, base_url, model, messages,
                                         temperature=req.get("temperature", 0.3),
                                         max_tokens=req.get("max_tokens"))
            self._json({"ok": err is None, "content": content or "", "error": err})
        elif p == "/api/ml/tuning":
            r = _ml.run_ecommerce_tuning()
            audit("system", "模型调优评测", f"电商 3 轮调优 {r['gate']['final_accuracy']}")
            self._json({"ok": True, **r})
        elif p == "/api/ml/compliance":
            self._handle_ml_compliance(req)
        elif p == "/api/ml/routing":
            self._handle_ml_routing(req)
        elif p == "/api/ml/abtest":
            self._handle_ml_abtest(req)
        elif p == "/api/ml/hallucination":
            self._handle_ml_hallucination(req)
        elif p == "/api/ml/offline_eval":
            r = _ml.offline_evaluation()
            audit("system", "离线评测", "三场景 + 回归门禁")
            self._json({"ok": True, **r})
        elif p == "/api/biz/ecommerce":
            query = (req.get("query") or "").strip()
            if not query:
                self._json({"error": "query 不能为空"}, 400)
                return
            try:
                top_k = int(req.get("top_k", 5))
            except (TypeError, ValueError):
                top_k = 5
            audit(self._actor, "电商导购", f"{query} · top{top_k}")
            r = _biz.ecommerce_search(query, top_k=top_k, llm=_make_llm(req))
            self._json({"ok": True, **r})
        elif p == "/api/biz/finance_report":
            query = (req.get("query") or "").strip()
            if not query:
                self._json({"error": "query 不能为空"}, 400)
                return
            audit(self._actor, "金融分析", query)
            r = _biz.finance_report(query, llm=_make_llm(req))
            self._json({"ok": True, **r})
        elif p == "/api/biz/office":
            query = (req.get("query") or "").strip()
            if not query:
                self._json({"error": "query 不能为空"}, 400)
                return
            try:
                top_k = int(req.get("top_k", 3))
            except (TypeError, ValueError):
                top_k = 3
            audit(self._actor, "办公助手", f"{query} · top{top_k}")
            r = _biz.office_assistant(query, top_k=top_k, llm=_make_llm(req))
            self._json({"ok": True, **r})
        elif p == "/api/review/approve" or p == "/api/review/reject":
            review_id = (req.get("review_id") or "").strip()
            comment = (req.get("comment") or "").strip()
            action = "approve" if p.endswith("approve") else "reject"
            t = _review.QUEUE.verdict(review_id, action, self._actor, comment)
            if not t:
                self._json({"error": "审核任务不存在或已裁决"}, 404)
                return
            audit(self._actor, "人工审核", f"{review_id} -> {t['status']} · {comment[:40]}")
            self._json({"ok": True, **t})
        elif p == "/api/tasks/cancel":
            task_id = (req.get("task_id") or "").strip()
            t = _tq.QUEUE.cancel(task_id)
            if not t:
                self._json({"error": "任务不存在"}, 404)
                return
            audit(self._actor, "取消任务", task_id)
            self._json({"ok": True, **t})
        elif p == "/api/llm/agent":
            self._handle_llm_agent(req)
        else:
            self._json({"error": "not found"}, 404)

    # ---------- 认证 ----------
    def _handle_login(self, req):
        username = (req.get("username") or "").strip()
        password = req.get("password", "")
        if not username or not password:
            self._json({"error": "用户名与密码不能为空"}, 400)
            return
        u = _auth.authenticate(username, password)
        if not u:
            audit("auth", "登录失败", username, result="error")
            self._json({"error": "用户名或密码错误"}, 401)
            return
        token = _auth.issue_token({"sub": username, "role": u["role"], "name": u["name"]})
        audit("auth", "登录成功", "%s (%s)" % (username, _auth.ROLE_NAMES.get(u["role"], u["role"])))
        self._json({"ok": True, "token": token, "expires_in": _auth.JWT_EXPIRE,
                    "user": u, "role_name": _auth.ROLE_NAMES.get(u["role"])})

    def _handle_create_user(self, req):
        username = (req.get("username") or "").strip()
        password = req.get("password", "")
        role = req.get("role", "viewer")
        name = (req.get("name") or "").strip()
        ok, msg = _auth.create_user(username, password, role, name)
        if not ok:
            self._json({"error": msg}, 400)
            return
        audit(self._actor, "创建用户", "%s (%s)" % (username, role))
        self._json({"ok": True, "username": username, "role": role,
                    "role_name": _auth.ROLE_NAMES.get(role)})

    # ---------- 真实 LLM Function Calling 闭环 ----------
    def _handle_llm_agent(self, req):
        query = (req.get("query") or "").strip()
        if not query:
            self._json({"error": "query 不能为空"}, 400)
            return
        api_key = _resolve_api_key(req)
        base_url = req.get("base_url", DEFAULT_LLM_BASE)
        model = req.get("model", DEFAULT_LLM_MODEL)
        scene_hint = req.get("scene", "") or _guess_scene(query)

        # 无 API Key：降级到确定性 DAG Agent（本地向量库），保证可用
        if not api_key:
            r, _ = self._run_qa({"query": query, "scene": scene_hint})
            self._json({"ok": True, "mode": "deterministic", "degraded": True,
                        "answer": r["answer"], "sources": r["sources"],
                        "hint": "未配置 API Key，已降级为确定性 Agent"})
            return

        # 写工具网关：写操作不直接执行，转人工审核（HITL）
        def _guard(name, arguments):
            tool = _tools.REGISTRY.get(name)
            if tool.side_effect == "write":
                t = _review.QUEUE.create(
                    "tool_write", "工具写操作复核：%s" % name,
                    {"tool": name, "arguments": arguments}, actor=self._actor)
                return False, "写操作已提交人工审核（%s），审核通过后再执行" % t["review_id"]
            return True, ""

        tools = _tools.REGISTRY.to_openai_tools()
        executor = _fc.build_tool_executor(_tools.REGISTRY, guard=_guard)
        messages = [{"role": "system", "content": _fc.build_system_prompt(scene_hint)},
                    {"role": "user", "content": query}]

        def chat_fn(msgs, tools_decl=None):
            content, err, tool_calls = _llm_chat(api_key, base_url, model, msgs,
                                                 temperature=0.2, tools=tools_decl)
            return {"content": content, "tool_calls": tool_calls, "error": err}

        res = _fc.run_agent_loop(chat_fn, messages, tools, executor)
        audit(self._actor, "FunctionCalling", query[:60])
        self._json({"ok": True, "mode": "function_calling", "scene": scene_hint, **res})

    # ---------- Agent 执行 ----------
    def _run_agent_sync(self, scenario, params, actor):
        """执行 Agent 工作流并返回结构化结果（供同步接口与异步任务队列复用）。"""
        self._actor = actor
        _CURRENT_ACTOR.set(actor)
        _CURRENT_ROLE.set(self._role if getattr(self, "_role", None) else "viewer")
        params = params or {}
        buf = io.StringIO()
        result = None
        breaker_state = None
        t0 = time.time()
        trace = _ml.OBS.start_trace(scenario, json.dumps(params, ensure_ascii=False)[:60])
        try:
            with contextlib.redirect_stdout(buf):
                if scenario == "ecommerce":
                    result = self._run_ecommerce(params)
                elif scenario == "finance":
                    result, breaker_state = self._run_finance(params)
                elif scenario == "office":
                    result = self._run_office(params)
                elif scenario == "qa":
                    result, breaker_state = self._run_qa(params)
                else:
                    raise ValueError("未知场景: %s" % scenario)
            logs = buf.getvalue().splitlines()
            latency = (time.time() - t0) * 1000
            _ml.OBS.add_span(trace, "意图识别", latency * 0.08)
            _ml.OBS.add_span(trace, "RAG 检索", latency * 0.4)
            _ml.OBS.add_span(trace, "生成", latency * 0.4)
            _ml.OBS.add_span(trace, "合规闸门", latency * 0.12)
            if any("合规红线" in l for l in logs):
                _ml.OBS.record_metric("compliance_violation", 1)
            _ml.OBS.finish_trace(trace, latency, tokens=len(logs) * 40, status="ok")
            audit(actor, "Agent执行", f"{scenario} -> {json.dumps(params, ensure_ascii=False)[:80]}")
            return {"ok": True, "scenario": scenario, "logs": logs,
                    "result": result, "breaker": breaker_state}
        except Exception as e:
            logs = buf.getvalue().splitlines()
            latency = (time.time() - t0) * 1000
            _ml.OBS.record_metric("error", 1)
            _ml.OBS.finish_trace(trace, latency, status="error")
            audit(actor, "Agent执行", f"{scenario}", result="error")
            return {"ok": False, "scenario": scenario, "logs": logs,
                    "error": str(e), "breaker": breaker_state}

    def _handle_agent_run(self, req):
        scenario = req.get("scenario", "")
        params = req.get("params", {}) or {}
        r = self._run_agent_sync(scenario, params, self._actor)
        code = 200 if r["ok"] else 500
        self._json(r, code)

    def _run_ecommerce(self, p):
        order_id = p.get("order_id", "O1001")
        user_id = p.get("user_id", "U42")
        reason = p.get("reason", "no_reason")
        request_id = p.get("request_id", "req-" + uuid.uuid4().hex[:6])
        idem_key = "refund-" + order_id + "-" + request_id
        ctx = {"order_id": order_id, "user_id": user_id,
               "reason": reason, "request_id": request_id}
        AGENT_ECOMMERCE.run(ctx, idempotency_key=idem_key)
        order = ctx.get("order", {})
        policy = ctx.get("policy", {})
        review_id = None
        if policy.get("decision") == "manual":
            # HITL：需人工复核的退款自动进入审核队列
            t = _review.QUEUE.create(
                "refund", "退款人工复核：%s" % order_id,
                {"order_id": order_id, "request_id": request_id,
                 "amount": order.get("amount"), "message": ctx.get("message", "")},
                actor=self._actor)
            review_id = t["review_id"]
        return {
            "order_id": order_id, "request_id": request_id,
            "idempotency_key": idem_key,
            "order": {"product": order.get("product_name"), "amount": order.get("amount"),
                      "status": order.get("status")},
            "decision": policy.get("decision"),
            "message": ctx.get("message", ""),
            "result": ctx.get("result"),
            "review_id": review_id,
        }

    def _run_finance(self, p):
        query = p.get("query", "星辰科技 2025 年营收是多少")
        ctx = {"query": query}
        AGENT_FINANCE.run(ctx)
        b = AGENT_FINANCE.breaker
        analysis = ctx.get("analysis", {})
        review_id = None
        if ctx.get("rejected"):
            # HITL：命中合规红线的输出进入审核队列
            t = _review.QUEUE.create(
                "compliance", "合规拦截复核：%s" % query[:40],
                {"query": query, "violations": [v.get("name") for v in ctx.get("violations", [])]},
                actor=self._actor)
            review_id = t["review_id"]
        return {
            "query": query, "answer": ctx.get("answer", ""),
            "evidence": ctx.get("evidence", ""),
            "sources": ctx.get("sources", []),
            "company": analysis.get("company"), "code": analysis.get("code"),
            "metrics": analysis.get("metrics", []),
            "trend": analysis.get("trend", []),
            "rejected": ctx.get("rejected", False),
            "violations": [v.get("name") for v in ctx.get("violations", [])],
            "review_id": review_id,
        }, {
            "state": b.state, "failures": b.failures,
            "threshold": b.failure_threshold, "recovery_timeout": b.recovery_timeout,
        }

    def _run_office(self, p):
        approval_type = p.get("approval_type", "leave")
        base = {
            "approval_id": p.get("approval_id", "AP-001"),
            "applicant": p.get("applicant", "张三"),
            "approval_type": approval_type,
            "params": p.get("params", {}),
            "event_id": p.get("event_id", "EV-001"),
            "notice_id": p.get("notice_id", "NT-001"),
            "fail_notice": bool(p.get("fail_notice", False)),
        }
        ctx = dict(base)
        try:
            AGENT_OFFICE.run(ctx)
            chain = ctx.get("chain", {})
            return {"approval_id": base["approval_id"], "status": "成功",
                    "compensated": False,
                    "type": chain.get("type"), "level": chain.get("level"),
                    "approvers": chain.get("approvers", [])}
        except Exception as e:
            return {"approval_id": base["approval_id"], "status": "失败并已补偿",
                    "compensated": True, "error": str(e)}

    def _run_qa(self, p):
        query = p.get("query", "").strip()
        if not query:
            raise ValueError("query 不能为空")
        ctx = {"query": query, "scene": p.get("scene", "")}
        AGENT_QA.run(ctx)
        b = AGENT_QA.breaker
        return {
            "query": query, "scene": ctx.get("scene", ""),
            "answer": ctx.get("answer", ""),
            "evidence": ctx.get("evidence", ""),
            "sources": ctx.get("sources", []),
        }, {
            "state": b.state, "failures": b.failures,
            "threshold": b.failure_threshold, "recovery_timeout": b.recovery_timeout,
        }

    # ---------- 模型工程（任务四） ----------
    def _handle_ml_compliance(self, req):
        answer = (req.get("answer") or "").strip()
        if not answer:
            self._json({"error": "answer 不能为空"}, 400)
            return
        citations = req.get("citations") or []
        red_hits = _ml.compliance_scan(answer)
        cc = _ml.check_citation_compliance(citations) if citations else None
        passed = not red_hits and (cc is None or not cc["hallucinated"])
        verdict = "通过" if passed else "拦截"
        audit(self._actor, "合规扫描", f"{answer[:40]} -> {verdict}")
        self._json({
            "ok": True, "answer": answer, "red_line_hits": red_hits,
            "citation": cc, "verdict": verdict, "passed": passed,
        })

    def _handle_ml_routing(self, req):
        text = (req.get("request") or "").strip()
        if not text:
            self._json({"error": "request 不能为空"}, 400)
            return
        route = _ml.route_request(text)
        completion = _ml.run_task_completion()
        audit(self._actor, "意图路由", f"{text[:40]} -> {route['level']}")
        self._json({"ok": True, "route": route, "completion": completion})

    def _handle_ml_abtest(self, req):
        control_acc = float(req.get("control_acc", 0.78))
        treat_acc = float(req.get("treat_acc", 0.83))
        n_users = int(req.get("n_users", 10000))
        traffic_pct = int(req.get("traffic_pct", 20))
        r = _ml.run_ab_test(control_acc, treat_acc, n_users, traffic_pct)
        audit("system", "A/B测试", f"A={control_acc} B={treat_acc} -> {r['verdict']}")
        self._json({"ok": True, **r})

    def _handle_ml_hallucination(self, req):
        query = (req.get("query") or "").strip()
        if not query:
            self._json({"error": "query 不能为空"}, 400)
            return
        scene = req.get("scene") or "all"
        mode = req.get("mode", "grounded")  # grounded | hallucinated
        hits = _rag_search(query, scene, top_k=3)
        if mode == "hallucinated":
            citations = [
                {"marker": 1, "doc_id": "不存在的研报-X", "page": "P99", "cell": "虚构表"},
                {"marker": 2, "doc_id": "星辰科技-2025年报", "page": "P12", "cell": "营收表"},
            ]
        else:
            citations = [{"marker": i + 1, "doc_id": c["doc_id"], "page": c["page"],
                          "cell": c["cell"]} for i, c in enumerate(_ml.FINANCE_CORPUS[:2])]
        answer = "根据检索证据：" + ("；".join(h["text"][:50] for h in hits[:2]) if hits else "无证据")
        r = _ml.hallucination_defense(query, hits, answer, citations)
        audit(self._actor, "幻觉防御", f"{query} @{mode}")
        self._json({"ok": True, **r})

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print("企业级 LLM 智能应用平台已启动: http://0.0.0.0:%d" % port)
    print("知识底座：%d 篇文档 -> %d 切片，黄金 QA %d 条；工具 %d 个"
          % (len(DOCS), len(CHUNKS), len(QA_PAIRS), len(_tools.REGISTRY._tools)))
    print("真实向量库：%d 向量（dim=%d，扁平余弦索引，持久化 %s）"
          % (len(VSTORE.records), VSTORE.dim, os.path.basename(VECTOR_STORE_FILE)))
    print("安全网关：JWT 鉴权 + RBAC 三角色 + 文档级 ACL + 接口分级限流 已启用")
    print("企业治理：Function Calling 闭环 / HITL 人工审核 / 异步任务队列 已启用")
    print("密钥托管：%s" % ("服务端 DEEPSEEK_API_KEY" if SERVER_API_KEY else "前端传入（未设 DEEPSEEK_API_KEY）"))
    print("向量持久化：本次恢复 %d 篇上传文档" % _PERSISTED_LOADED)
    server.serve_forever()


if __name__ == "__main__":
    main()
