# 企业级 LLM 实操平台

> RAG 知识底座 × Agent 决策引擎 × 模型工程，三核闭环的**零依赖**企业级实操平台。
> 纯 Python 标准库实现，`python server.py` 一行启动，不装任何包、不下载模型。

---

## 一、项目概览

本平台把企业级 LLM 应用的三大核心工程能力整合为一个可运行、可交互的完整程序：

```
┌─────────────────────────────────────────────────────────────┐
│                     企业级 LLM 实操平台                       │
├────────────────┬────────────────┬───────────────────────────┤
│ ① RAG 知识底座 │ ② Agent 决策引擎 │ ③ 模型工程                 │
│  多格式解析     │  工具调用        │  3 轮调优 / 合规红线       │
│  5 种切片策略   │  DAG 编排       │  P0/P1/P2 路由            │
│  混合检索       │  熔断/ Saga/幂等 │  离线评测 / A/B / 幻觉防御 │
│  质量基线       │  场景权限隔离     │  可观测性                 │
└────────────────┴────────────────┴───────────────────────────┘
```

三大场景全覆盖：

| 场景 | RAG 特征 | Agent 流程 | 模型工程 |
|------|---------|-----------|---------|
| **电商导购** | 图文混排 + 多模态检索 | 退款审批（条件分支 + 幂等） | 准确率 40%→78% 调优 |
| **金融投研** | PDF 表格 + 合规溯源 | 合规熔断（红线闸门） | 6 条红线 + 引用合规率 |
| **企业办公** | 多格式 + 权限隔离 | 跨系统 Saga（并行 + 补偿） | P0/P1/P2 路由 + 完成率 |

---

## 二、目录结构

```
enterprise-platform/
├── server.py                 # 统一 HTTP 服务（入口，路由 + RAG/Agent/ML 接口）
├── multiparse.py             # 多格式文档解析（docx/xlsx/pptx/pdf/csv/json/html/图片）
├── tools.py                  # 工具注册表（12 工具 Function Calling 声明）
├── agents.py                 # 三个业务 Agent（DAG 工作流定义）
├── orchestrator.py           # DAG 工作流编排器（拓扑排序/重试/熔断/Saga/幂等）
├── model_eng.py              # 模型工程全链路（调优/合规/路由/评测/A-B/幻觉/可观测）
├── common/
│   ├── tokenizer.py          # 中英混合分词（中文 unigram+bigram）
│   ├── bm25.py               # Okapi BM25 稀疏检索
│   ├── vector.py             # TF-IDF 向量化 + 余弦相似度
│   └── metrics.py            # Recall@K / MRR / nDCG@K 检索质量指标
├── chunking/
│   └── chunk_strategies.py   # 5 种切片策略（固定/递归/语义/结构/父子）
├── retrieval/
│   └── hybrid.py             # 混合检索（BM25 + 稠密 + RRF 融合）
├── data/
│   ├── synth_dataset.json    # 基础语料（8 文档 / 48 切片 / 32 黄金 QA）
│   └── sample_files.json     # 内置多格式模拟文档（8 篇，启动自动预载）
└── console.html              # 前端工作台（单文件，6 个 tab）
```

---

## 三、每个 Python 文件的作用

### 3.1 `server.py` —— 统一服务入口

**作用**：平台唯一对外入口。加载数据集 → 构建 RAG 索引 → 预载模拟文档 → 注入 RAG 检索到 Agent 工具层 → 启动 HTTP 服务，把三大能力统一暴露为 REST API，并维护审计日志与可观测性埋点。

**关键运行代码**（服务启动 + RAG↔Agent 打通）：

```python
# 构建 RAG 索引 + 把真实检索注入 Agent 工具层（核心闭环）
RETRIEVER = HybridRetriever(CHUNKS)
_tools.set_rag_search(_rag_search)      # Agent 的 search_reports 工具真实调用 RAG

def main():
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), Handler)   # 监听 $PORT，绑 0.0.0.0
    server.serve_forever()
```

主要 API：

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/overview` | 平台总览（文档/切片/QA/工具统计） |
| POST | `/api/rag/upload` | 上传文档 → 多格式解析 → 关键词提取 → 入索引 |
| POST | `/api/rag/search` | 实时混合检索（场景隔离 + 关键词高亮） |
| POST | `/api/rag/chunking` / `/api/rag/eval` | 切片对比 / 质量基线 |
| POST | `/api/agent/run` | 执行三场景 + 问答 Agent 工作流 |
| POST | `/api/ml/*` | 调优 / 合规 / 路由 / A/B / 幻觉 / 离线评测 |

---

### 3.2 `multiparse.py` —— 多格式文档解析

**作用**：把用户上传的任意格式文件（图文混排 / PDF 表格 / 多格式办公文档）解析为纯文本，统一交给 RAG 切片建索引。纯标准库实现：`docx/xlsx/pptx` 用 zipfile 解 OOXML + 正则提取；`pdf` 用 zlib 解 FlateDecode 文本流；图片降级为多模态占位。

**关键运行代码**（PDF 文本流提取）：

```python
def _extract_pdf(data):
    texts = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        dec = _decompress_stream(m.group(1))          # zlib 解 FlateDecode
        for tm in re.finditer(rb"\((?:[^()\\]|\\.)*\)\s*Tj", dec):   # 抓 (..)Tj
            texts.append(_pdf_unescape(re.sub(rb"^\(|\)\s*Tj$", b"", tm.group(0))))
    return "\n".join(t for t in texts if t.strip())
```

统一入口 `parse_bytes(filename, data)` 返回 `{ext, format, text, note}`。

---

### 3.3 `tools.py` —— 工具注册表

**作用**：把外部系统能力封装为可调用工具，定义 Function Calling 的 JSON Schema 契约。每个工具标注副作用分级（`read` 只读 / `write` 写需幂等）与补偿函数。核心是**提供 RAG 检索注入点**，让 Agent 的工具调用真正落到 RAG 知识底座。

**关键运行代码**（RAG 检索注入点 + 合规闸门工具）：

```python
_RAG_SEARCH = None
def set_rag_search(fn):            # server.py 启动时注入真实混合检索
    global _RAG_SEARCH; _RAG_SEARCH = fn

def tool_search_reports(query, scene="finance"):
    if _RAG_SEARCH:
        hits = _RAG_SEARCH(query, scene)          # 真实 RAG，带溯源
        return {"evidence": "；".join(h["text"][:90] for h in hits[:3]), "hits": hits}
    return {"evidence": "该标的近三年营收 CAGR 12%..."}   # 无注入时兜底
```

---

### 3.4 `agents.py` —— 三个业务 Agent

**作用**：把 `tools.py` 的工具组合成确定性 DAG 工作流，定义三大场景的业务逻辑，并暴露核心的**检索增强问答 Agent**（RAG+Agent 闭环主线）。

**关键运行代码**（电商条件分支 + 金融合规熔断）：

```python
def build_ecommerce_agent():
    def node_check_policy(ctx):
        return "auto" if ctx["order"]["amount"] < 500 else "manual"   # 金额分支
    return (Workflow("电商客服Agent", start="query_order")
            .add_node("query_order", node_query_order, next="check_policy")
            .add_node("check_policy", node_check_policy)
            .add_node("auto_refund", node_auto_refund, next="notify")
            .add_node("manual_approve", node_manual_approve, next="notify")
            .branch("check_policy", {"auto": "auto_refund", "manual": "manual_approve"}))

def build_qa_agent():                # 意图识别 → RAG 检索 → 生成 → 合规闸门 → 交付
    ...
    .add_node("retrieve", node_retrieve, next="generate")
    .add_node("check", guarded_check, fallback=node_reject)   # 熔断 + 兜底
```

---

### 3.5 `orchestrator.py` —— DAG 工作流编排器

**作用**：企业级工作流引擎，提供六大能力：DAG 拓扑排序 + 并行执行、条件分支（skip 传播）、节点级重试/超时/降级、Saga 逆序补偿、熔断器、幂等状态机。

**关键运行代码**（熔断器 + 幂等占位）：

```python
class CircuitBreaker:
    def call(self, fn, *a, **kw):
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.HALF_OPEN          # 冷却后进半开试探
            else:
                raise CircuitOpenError("熔断器打开，快速失败")
        try:
            return fn(*a, **kw)
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = self.OPEN               # 连续失败达阈值打开
            raise
```

---

### 3.6 `model_eng.py` —— 模型工程全链路

**作用**：覆盖三场景模型层工程 + 四大通用实践（离线评测 / A/B 测试 / 幻觉防御 / 可观测性），产出可复现、可解释的工程结论。

**关键运行代码**（3 轮调优门禁 + A/B 显著性检验）：

```python
def run_ecommerce_tuning():
    rng = random.Random((hash(v["id"]) ^ 0x5EED) & 0xFFFFFFFF)
    correct_idx = set(rng.sample(range(n), target_correct))   # 确定性复现准确率
    ...
    gate = {"threshold": 0.75, "final_accuracy": final["accuracy"],
            "pass": final["accuracy"] >= 0.75}                # 回归门禁

def two_prop_z_test(p1, n1, p2, n2):                          # 双比例 z 检验
    pp = (p1*n1 + p2*n2)/(n1+n2); se = math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    return round((p1-p2)/se, 4), round(2*(1-_normal_cdf(abs((p1-p2)/se))), 6)
```

---

### 3.7 `common/tokenizer.py` —— 分词

**作用**：中英混合分词。英文/数字按词切分；中文用 unigram + bigram 覆盖（兼顾单字与相邻词组），无 jieba 依赖。

```python
def tokenize(text):
    tokens = []
    for m in re.finditer(r"[a-z0-9]+", text.lower()):
        tokens.append(m.group())              # 英文/数字按词
    zh = _CJK_RE.findall(text)
    tokens.extend(zh)                          # 中文 unigram
    for i in range(len(zh) - 1):
        tokens.append(zh[i] + zh[i + 1])       # 中文 bigram
    return tokens
```

---

### 3.8 `common/bm25.py` —— BM25 稀疏检索

**作用**：Okapi BM25 实现，稀疏检索通道，为混合检索提供关键词精确匹配与 IDF 加权。

```python
def score(self, query, doc_idx):
    for t in tokenize(query):
        f = self.tf[doc_idx].get(t, 0)
        idf = math.log((self.N - self.df[t] + 0.5)/(self.df[t] + 0.5) + 1.0)
        s += idf * (f*(self.k1+1)) / (f + self.k1*(1 - self.b + self.b*dl/self.avgdl))
    return s
```

---

### 3.9 `common/vector.py` —— TF-IDF 向量化

**作用**：TF-IDF 向量化 + 余弦相似度（向量 L2 归一化），作为「稠密」检索通道。

```python
class TfidfVectorizer:
    def fit(self, docs):
        self.idf = {t: math.log((1+n)/(1+df[t])) + 1.0 for t in df}   # 平滑 idf
    def _vec(self, text):
        vec = [0.0] * len(self.vocab)
        for t, c in Counter(tokenize(text)).items():
            if t in self.idf: vec[self.vocab[t]] = c * self.idf[t]
        return [v/norm for v in vec]                                   # L2 归一化
```

---

### 3.10 `common/metrics.py` —— 检索质量指标

**作用**：Recall@K / MRR / nDCG@K 单黄金标注评估，建立检索质量基线。

```python
def evaluate(ranked_ids_list, gold_ids, ks=(1,3,5,10)):
    out = {"n": n, "mrr": 0.0}
    for ranked, gold in zip(ranked_ids_list, gold_ids):
        out["mrr"] += mrr(ranked, gold)
        for k in ks:
            out[f"recall@{k}"] += recall_at_k(ranked, gold, k)
    return {k: v/n for k, v in out.items()}   # 各指标均值
```

---

### 3.11 `chunking/chunk_strategies.py` —— 切片策略

**作用**：5 种切片策略（固定长度 / 递归 / 语义 / 结构感知 / 父子），供切片对比实验选择最优方案。**结构感知**（按 Markdown 标题 + 表格行带表头上下文）是默认最优。

```python
def chunk_structure(text):
    sections = [s.strip() for s in _SEC_SPLIT.split(text) if s.strip()]  # 按 ## 分节
    # 表格数据行带表头上下文单独成块；段落各成一块
```

---

### 3.12 `retrieval/hybrid.py` —— 混合检索

**作用**：融合 BM25 稀疏 + TF-IDF 稠密双路召回，用 RRF（倒数排名融合）得到最终排序。

```python
def rrf_fuse(rankings_list, k=60):
    scores = {}
    for rankings in rankings_list:
        for rank, (doc_id, _) in enumerate(rankings):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 四、核心功能与关键运行代码

### 功能 1：多格式文档上传 + 检索分析（关键词高亮）

**作用**：上传图文混排 / PDF 表格 / Word / Excel / PPT / CSV 等任意格式文档，自动解析 → 提取关键词 → 纳入检索索引 → 自动检索分析并高亮关键词。

**关键运行代码**（上传 → 关键词提取 → 高亮）：

```python
def _extract_keywords(text, top_n=10):
    # 英文 token + 中文 2~4 gram 词频统计，长词优先去重
    counter = collections.Counter()
    for span in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for n in (4, 3, 2):
            for i in range(len(span)-n+1):
                g = span[i:i+n]
                if g not in _STOPWORDS: counter[g] += 1
    cand = sorted(counter.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)
    # 贪心选取互不包含的词，长词优先
    return [w for w,_ in cand if not any(w in p for p in picked)][:top_n]

def _highlight(text, keywords):      # <mark> 高亮（中文按 2 字以上连续串）
    for k in sorted(set(keywords), key=len, reverse=True):
        text = re.sub(re.escape(k), lambda m: "<mark>%s</mark>" % m.group(0), text, flags=re.I)
    return text
```

---

### 功能 2：混合检索 + 质量基线

**作用**：BM25 + 稠密 + RRF 三种模式，按场景权限隔离检索；用 Recall@K / MRR / nDCG 建立质量基线。

**关键运行代码**（场景隔离 + 检索）：

```python
def _run_search(query, scene, mode, top_k):
    allowed_idx = [i for i, doc_id in enumerate(CHUNK_META)
                   if scene in ("", "all", None) or _DOC_BY_ID[doc_id]["scene"] == scene]
    bm = RETRIEVER._bm25_rank(query, limit=50)
    de = RETRIEVER._dense_rank(query, limit=50)
    fused = rrf_fuse([[(i,s) for i,s in bm if i in allowed_idx],
                      [(i,s) for i,s in de if i in allowed_idx]])   # 隔离 + 融合
```

---

### 功能 3：Agent 工具调用 + DAG 编排

**作用**：三场景 Agent 工作流，含条件分支、合规熔断、Saga 补偿、幂等保障。

**关键运行代码**（办公 Saga 并行汇聚 + 逆序补偿）：

```python
def build_office_agent():
    return (Workflow("企业办公Agent", start="create_approval")
            .add_node("create_approval", node_create, compensate=node_undo_create)
            .add_node("prepare_attachment", node_prepare)            # 无依赖并行
            .add_node("add_calendar", node_calendar, compensate=node_undo_calendar,
                      depends_on=["create_approval", "prepare_attachment"])  # 汇聚
            .add_node("send_notice", node_notice, compensate=node_undo_notice,
                      depends_on=["add_calendar"]))
# 失败时 orchestrator._compensate(executed) 按逆序调用 compensate 回滚
```

---

### 功能 4：模型工程全链路

**作用**：电商 3 轮调优（40%→78%）+ 金融合规红线 + 办公分级路由 + 离线评测 + A/B 测试 + 幻觉防御 + 可观测性。

**关键运行代码**（幻觉防御 4 层）：

```python
def hallucination_defense(query, hits, answer, citations=None):
    # 层1 检索优先：无命中拒绝回答
    # 层2 溯源强制：结论必须附真实引用
    # 层3 引用校验：引用必须指回原文坐标（check_citation_compliance）
    # 层4 拒绝兜底：无依据时输出『抱歉，我无法确认该信息』
```

---

## 五、运行方式

```bash
# 零依赖，一行启动（默认端口 8000）
python server.py

# 指定端口（线上部署由平台注入 $PORT）
PORT=8080 python server.py
```

打开浏览器访问 `http://127.0.0.1:8000`，即进入工作台（6 个 tab：总览 / RAG 知识底座 / Agent 决策引擎 / 智能助手 / 模型工程 / 企业治理）。

---

## 六、Git 版本管理

本目录已初始化 Git 仓库，17 个文件全部纳入版本管理（`__pycache__` 等运行时产物已通过 `.gitignore` 排除）。

```bash
git log --oneline          # 查看提交历史
git status                 # 查看工作区状态
git diff                   # 查看改动
```
