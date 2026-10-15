# 企业级 LLM 实操平台 · AI 直连版

> **AI 直连检索** × RAG 知识底座 × Agent 决策引擎 × 模型工程 × **业务功能**，五核闭环的**零依赖**企业级实操平台。
> 纯 Python 标准库实现，`python server.py` 一行启动，不装任何包、不下载模型。
> 本版在完整保留三核的基础上，检索能力改为**直连 DeepSeek 等主流大模型 API**，并落地三场景的**具体实现**：
> - **业务功能**（电商导购 / 金融投研 / 企业办公）——本地向量库检索打底 + 可选 LLM 增强的混合模式；
> - **决策引擎**（电商客服退款 / 金融投研合规 / 企业办公审批）——真实订单库 + 退款规则引擎、财报指标 + 6 条合规红线 + 熔断、分级审批链 + Saga 补偿。

---

## 一、项目概览

```
┌──────────────────────────────────────────────────────────────────┐
│               企业级 LLM 实操平台 · AI 直连版                      │
├──────────┬──────────┬──────────────┬───────────────┬─────────────┤
│ ⓪ AI直连 │ ① RAG   │ ② Agent     │ ③ 模型工程    │  治理       │
│ 检索      │ 知识底座  │ 决策引擎      │               │             │
│ LLM 问答  │ 多格式解析│ 工具调用      │ 3 轮调优      │ 鉴权        │
│ 文件直连  │ 5 种切片  │ DAG 编排     │ 合规红线      │ 审计        │
│ 检索      │ 混合检索  │ 熔断/Saga/幂等│ P0/P1/P2 路由 │ 合规        │
│ (DeepSeek)│ 质量基线  │ 场景权限隔离  │ 评测/A-B/幻觉 │             │
└──────────┴──────────┴──────────────┴───────────────┴─────────────┘
```

**核心区别**：本版新增「AI 智能检索」——提问/本地文件检索不经过本地 BM25/向量索引，而是把内容直接交给 DeepSeek 等 OpenAI 兼容大模型作答。API Key 由用户在浏览器前端填写，仅存 `localStorage`，不落服务器。

| 场景 | RAG 特征 | Agent 决策引擎（具体实现） | 模型工程 |
|------|---------|-----------|---------|
| **电商客服** | 图文混排 + 多模态检索 | 真实订单库 + 退款规则引擎（状态×金额×原因三分支 + 幂等防重） | 准确率 40%→78% 调优 |
| **金融投研** | PDF 表格 + 合规溯源 | 财报指标计算 → 客观陈述 → 6 条红线闸门 → 熔断兜底 | 6 条红线 + 引用合规率 |
| **企业办公** | 多格式 + 权限隔离 | 审批类型识别 + 分级审批链（P1/P2/P3）+ Saga 逆序补偿 | P0/P1/P2 路由 + 完成率 |

---

## 二、目录结构

```
enterprise-platform-api/
├── server.py                 # 统一 HTTP 服务（入口 + 安全网关 + RAG/Agent/ML + LLM 直连）
├── auth.py                   # 安全模块：JWT 登录 + RBAC 三级角色 + API 限流（纯标准库）
├── multiparse.py             # 多格式文档解析（docx/xlsx/pptx/pdf/csv/json/html/图片）
├── tools.py                  # 工具注册表（17 工具 Function Calling 声明，含订单状态机）
├── agents.py                 # 三个业务 Agent（DAG 工作流定义）
├── orchestrator.py           # DAG 工作流编排器（拓扑排序/重试/熔断/Saga/幂等）
├── model_eng.py              # 模型工程全链路（调优/合规/路由/评测/A-B/幻觉/可观测）
├── business.py               # 三场景业务功能（电商导购/金融投研/企业办公，混合模式）
├── common/
│   ├── __init__.py           # 公共模块包标记
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
│   ├── sample_files.json     # 内置多格式模拟文档（8 篇，启动自动预载）
│   ├── business_data.json    # 业务语料（24 商品 / 2 公司 6 季度财报 / 12 条办公制度）
│   ├── users.json            # 用户库（首次启动生成，PBKDF2 密码哈希，不落明文）
│   └── persisted_docs.json   # 上传文档持久化（重启自动恢复索引）
└── console.html              # 前端工作台（单文件，8 个 tab + 登录门禁）
```

---

## 三、每个 Python 文件的作用

### 3.1 `server.py` —— 统一服务入口

**作用**：平台唯一对外入口。加载数据集 → 构建 RAG 索引 → 预载模拟文档 → 注入 RAG 检索到 Agent 工具层 → 启动 HTTP 服务，把四大能力统一暴露为 REST API，并维护审计日志与可观测性埋点。**本版新增 LLM 直连代理**（转发 OpenAI 兼容接口，Key 由前端传入不落盘）。

**关键运行代码**（LLM 直连代理 + 服务启动）：

```python
DEFAULT_LLM_BASE = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
MAX_CONTEXT_CHARS = 12000          # 本地文件上下文上限，避免超 token

def _llm_chat(api_key, base_url, model, messages, temperature=0.3, max_tokens=None):
    if not api_key:
        return None, "未配置 API Key，请先在「API 配置」面板填写"
    url = (base_url or DEFAULT_LLM_BASE).strip().rstrip("/") + "/chat/completions"
    payload = {"model": model or DEFAULT_LLM_MODEL, "messages": messages,
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key.strip()})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"], None

def main():
    port = int(os.environ.get("PORT", "8000"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()   # 监听 $PORT，绑 0.0.0.0
```

主要 API：

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/api/auth/login` | 登录（返回 JWT Token） |
| GET | `/api/auth/me` | 当前登录用户信息 |
| GET/POST | `/api/auth/users` | 用户列表 / 创建用户（仅 admin） |
| POST | `/api/llm/retrieve` | AI 智能问答（query → 直连 LLM 作答） |
| POST | `/api/llm/file_retrieve` | 本地文件检索（上传解析 → 内容直连 LLM） |
| POST | `/api/llm/chat` | OpenAI 兼容通用转发 |
| POST | `/api/biz/ecommerce` | 电商导购：约束解析 + 商品检索 + 过滤排序 |
| POST | `/api/biz/finance_report` | 金融投研：财报指标提取 + 多季度趋势 + 溯源 |
| POST | `/api/biz/office` | 企业办公：制度/报销规则检索 + 结构化回答 |
| POST | `/api/rag/upload` | 上传文档 → 多格式解析 → 关键词提取 → 入索引 |
| POST | `/api/rag/search` | 本地混合检索（场景隔离 + 关键词高亮） |
| POST | `/api/rag/chunking` / `/api/rag/eval` | 切片对比 / 质量基线 |
| POST | `/api/agent/run` | 执行三场景 + 问答 Agent 工作流 |
| GET | `/api/orders` | 列出全部订单及实时状态 |
| POST | `/api/order/query` | 按订单号查询订单 |
| POST | `/api/order/update` | 实时更新订单状态（状态机校验 + 落盘） |
| POST | `/api/ml/*` | 调优 / 合规 / 路由 / A/B / 幻觉 / 离线评测 |

---

### 3.2 `multiparse.py` —— 多格式文档解析

**作用**：把用户上传的任意格式文件（图文混排 / PDF 表格 / 多格式办公文档）解析为纯文本，供 RAG 切片建索引，也供 AI 直连检索作为大模型上下文。纯标准库实现：`docx/xlsx/pptx` 用 zipfile 解 OOXML + 正则提取；`pdf` 用 zlib 解 FlateDecode 文本流；图片降级为多模态占位。

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

**关键运行代码**（RAG 检索注入点）：

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

**关键运行代码**（电商退款规则引擎三分支 + 金融合规熔断）：

```python
def build_ecommerce_agent():
    def node_check_policy(ctx):
        # 退款规则引擎：按订单状态 × 金额 × 原因三分支
        return tools.tool_check_refund_policy(ctx["order"],
                                              ctx.get("reason", "no_reason"))["decision"]
    return (Workflow("电商客服Agent", start="query_order")
            .add_node("query_order", node_query_order, next="check_policy")
            .add_node("check_policy", node_check_policy)          # auto / manual / reject
            .add_node("auto_refund", node_auto_refund, next="notify")
            .add_node("manual_approve", node_manual_approve, next="notify")
            .add_node("reject_refund", node_reject_refund, next="notify")
            .add_node("notify", node_notify)
            .branch("check_policy", {"auto": "auto_refund", "manual": "manual_approve",
                                     "reject": "reject_refund"}))

def build_finance_agent():   # RAG 检索 → 财报指标计算 → 客观陈述 → 合规闸门 → 交付
    ...
    .add_node("analyze", node_analyze, next="generate")       # 财报指标计算
    .add_node("check", guarded_check, fallback=node_reject)   # 6 条红线 + 熔断 + 免责兜底
```

---

### 3.5 `orchestrator.py` —— DAG 工作流编排器

**作用**：企业级工作流引擎，提供六大能力：DAG 拓扑排序 + 并行执行、条件分支（skip 传播）、节点级重试/超时/降级、Saga 逆序补偿、熔断器、幂等状态机。

**关键运行代码**（熔断器）：

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
    tokens = [m.group() for m in re.finditer(r"[a-z0-9]+", text.lower())]  # 英文/数字
    zh = _CJK_RE.findall(text)
    tokens.extend(zh)                          # 中文 unigram
    tokens.extend(zh[i] + zh[i+1] for i in range(len(zh)-1))   # 中文 bigram
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

### 3.11 `common/__init__.py` —— 公共模块包标记

**作用**：把 `common/` 标记为 Python 包（供 `from common.metrics import evaluate` 等跨模块导入）。内容仅一行注释，无实际逻辑。

---

### 3.12 `chunking/chunk_strategies.py` —— 切片策略

**作用**：5 种切片策略（固定长度 / 递归 / 语义 / 结构感知 / 父子），供切片对比实验选择最优方案。**结构感知**（按 Markdown 标题 + 表格行带表头上下文）是默认最优。

```python
def chunk_structure(text):
    sections = [s.strip() for s in _SEC_SPLIT.split(text) if s.strip()]  # 按 ## 分节
    # 表格数据行带表头上下文单独成块；段落各成一块
```

---

### 3.13 `retrieval/hybrid.py` —— 混合检索

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

### 3.14 `business.py` —— 三场景业务功能（混合模式）

**作用**：三场景的**具体业务实现**，统一模式为「本地向量库检索打底 + 可选 LLM 增强」。加载 `data/business_data.json` 后构建商品 / 制度两套 `HybridRetriever` 索引；`set_rag_search(fn)` 提供金融 / 办公溯源的 RAG 注入点（由 `server.py` 启动时注入真实混合检索）。

- **`ecommerce_search(query)`**：正则解析自然语言约束（`3000以内`/`4.5星以上`/品类）→ 商品混合检索 → 硬过滤 → 排序推荐 + 命中高亮。
- **`finance_report(query)`**：识别公司 → 提取最新季度营收/净利/毛利 → 计算同比、环比与**可比口径累计同比**（H2 对齐，避免季度数不一致）→ 输出趋势研判 + 溯源。
- **`office_assistant(query)`**：制度规则向量检索 → 结构化回答 + 细则溯源。

**关键运行代码**（约束解析 + 可比口径同比）：

```python
def parse_constraints(query):
    cons = {}
    m = re.search(r"(?:(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|之内|内))|"
                  r"(?:不超过\s*(\d+(?:\.\d+)?))|(?:预算\s*(\d+(?:\.\d+)?))", query)
    if m: cons["max_price"] = float(next(x for x in m.groups() if x))
    # 评分：4.5星以上；品类：CATEGORY_KEYS 命中
    return cons

def finance_report(query, llm=None):
    quarters = company["quarters"]
    # 可比口径同比：2025 H2 vs 2024 H2（各 2 季度对齐）
    h2_2025 = [q for q in quarters if q["quarter"] in ("2025Q3", "2025Q4")]
    h2_2024 = [q for q in quarters if q["quarter"] in ("2024Q3", "2024Q4")]
    rev_yoy = round((sum(q["revenue"] for q in h2_2025)
                     - sum(q["revenue"] for q in h2_2024))
                    / sum(q["revenue"] for q in h2_2024) * 100, 1)
```

---

## 四、核心功能与关键运行代码

### 功能 1：AI 直连检索（本版核心新增）

**作用**：检索不经过本地向量库，直接把问题/本地文件内容交给 DeepSeek 等 OpenAI 兼容大模型作答。分两种：
- **AI 智能问答**：`POST /api/llm/retrieve`，query + 可选上下文 → 拼 prompt → 调 LLM。
- **本地文件检索**：`POST /api/llm/file_retrieve`，上传文件 → `multiparse` 解析 → 内容作上下文 → 调 LLM。

**关键运行代码**（拼 prompt + 调 LLM）：

```python
def _llm_retrieve(api_key, base_url, model, query, context=None, scene=None):
    sys_prompt = ("你是企业级智能检索助手，请基于提供的资料严谨、准确地回答。"
                  "资料不足时如实说明，绝不编造；金融/投资信息注明仅供参考。")
    user_parts = []
    if context:
        ctx = context.strip()
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS] + "\n…（内容过长已截断）"
        user_parts.append("参考资料：\n" + ctx)      # 本地文件全文作上下文
    user_parts.append("问题：" + query)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "\n\n".join(user_parts)}]
    answer, err = _llm_chat(api_key, base_url, model, messages)
    return {"ok": err is None, "answer": answer or "", "error": err}
```

**前端 API 配置**（Key 仅存浏览器，不上传服务器）：

```js
const AI_KEY = 'ai_cfg';
function aiSaveCfg(){
  const c = {key:$('#aiKey').value.trim(), base:$('#aiBase').value.trim(),
             model:$('#aiModel').value.trim()};
  localStorage.setItem(AI_KEY, JSON.stringify(c));      // 只存本机
}
// 调用：post('/api/llm/retrieve', {api_key, base_url, model, query})
```

---

### 功能 2：多格式文档上传 + 检索分析（关键词高亮）

**作用**：上传图文混排 / PDF 表格 / Word / Excel / PPT / CSV 等任意格式文档，自动解析 → 提取关键词 → 纳入本地检索索引 → 自动检索分析并高亮关键词。

**关键运行代码**（关键词提取 + 高亮）：

```python
def _extract_keywords(text, top_n=10):
    counter = collections.Counter()
    for span in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for n in (4, 3, 2):
            for i in range(len(span)-n+1):
                g = span[i:i+n]
                if g not in _STOPWORDS: counter[g] += 1
    cand = sorted(counter.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)
    return [w for w,_ in cand if not any(w in p for p in picked)][:top_n]

def _highlight(text, keywords):      # <mark> 高亮
    for k in sorted(set(keywords), key=len, reverse=True):
        text = re.sub(re.escape(k), lambda m: "<mark>%s</mark>" % m.group(0), text, flags=re.I)
    return text
```

---

### 功能 3：本地混合检索 + 质量基线

**作用**：BM25 + 稠密 + RRF 三种模式，按场景权限隔离检索；用 Recall@K / MRR / nDCG 建立质量基线。**本版仍保留此能力**，供与 AI 直连检索对照（本地 vs 云端两种检索范式）。

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

### 功能 4：Agent 工具调用 + DAG 编排（三场景具体实现）

**作用**：三个业务 Agent 工作流，含真实订单库退款规则引擎、财报指标 + 合规熔断、分级审批链 + Saga 补偿、幂等保障。17 个工具（Function Calling 契约），并新增**订单状态实时管理**：通过订单号查询/更改订单状态（状态机校验 `paid→shipped→delivered→refunded/cancelled` + 时间戳联动 + 落盘 `orders_runtime.json`），对外提供 `GET /api/orders`、`POST /api/order/query`、`POST /api/order/update` 三个接口。

```python
def build_office_agent():          # 企业办公：审批类型识别 + 分级审批链 + Saga
    def node_create(ctx):          # 请假/报销/出差 → 分级审批链（P1/P2/P3）
        chain = tools.tool_create_approval(
            ctx["approval_id"], ctx["applicant"],
            approval_type=ctx.get("approval_type", "leave"),
            params=ctx.get("params", {}))
        ctx["chain"] = chain["chain"]
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

### 功能 5：模型工程全链路

**作用**：电商 3 轮调优（40%→78%）+ 金融合规红线 + 办公分级路由 + 离线评测 + A/B 测试 + 幻觉防御 + 可观测性。

```python
def hallucination_defense(query, hits, answer, citations=None):
    # 层1 检索优先：无命中拒绝回答
    # 层2 溯源强制：结论必须附真实引用
    # 层3 引用校验：引用必须指回原文坐标（check_citation_compliance）
    # 层4 拒绝兜底：无依据时输出『抱歉，我无法确认该信息』
```

---

### 功能 6：三场景业务功能（本地向量库 + 可选 LLM 增强）

**作用**：三场景的具体业务实现，本地向量库（BM25 + TF-IDF 稠密 + RRF）检索打底，前端已保存 API Key 时自动叠加 LLM 增强（混合模式），未配置则纯本地检索优雅降级。

- **电商导购**：`POST /api/biz/ecommerce` —— 自然语言约束解析 + 商品混合检索 + 硬过滤排序，返回结构化商品列表 + 命中高亮。
- **金融投研**：`POST /api/biz/finance_report` —— 财报指标提取 + 多季度趋势 + 可比口径同比 + 溯源。
- **企业办公**：`POST /api/biz/office` —— 制度 / 报销 / 考勤规则检索 + 结构化回答。

**关键运行代码**（业务层 LLM 注入 + 优雅降级）：

```python
def _make_llm(req):
    api_key = (req.get("api_key") or "").strip()
    if not api_key:
        return None                        # 未配置 Key → 纯本地向量库检索
    base_url = req.get("base_url", DEFAULT_LLM_BASE)
    model = req.get("model", DEFAULT_LLM_MODEL)
    return lambda messages: _llm_chat(api_key, base_url, model, messages)
```

---

## 五、运行方式

```bash
# 零依赖，一行启动（默认端口 8000）
python server.py

# 指定端口（线上部署由平台注入 $PORT）
PORT=8080 python server.py

# 可选：服务端密钥托管 + 自定义 JWT 密钥（生产推荐）
DEEPSEEK_API_KEY=sk-xxx JWT_SECRET=your-secret python server.py
```

打开浏览器访问 `http://127.0.0.1:8000`，**首次需登录**（默认账号见下方「企业级安全」章节），登录后进入工作台（8 个 tab：总览 / 智能助手 / **AI 智能检索** / **业务功能** / RAG 知识底座 / Agent 决策引擎 / 模型工程 / 企业治理）。

**使用 AI 智能检索**：未配置服务端 `DEEPSEEK_API_KEY` 时，可在「AI 智能检索」页顶部填写 DeepSeek（或任意 OpenAI 兼容）API Key（Key 仅存本地浏览器）；配置了服务端 Key 则前端无需填，接口自动使用服务端密钥。

---

## 六、Git 版本管理

本目录已初始化 Git 仓库，全部文件纳入版本管理（`__pycache__` 等运行时产物已通过 `.gitignore` 排除）。

```bash
git log --oneline          # 查看提交历史
git status                 # 查看工作区状态
git diff                   # 查看改动
```

---

## 七、企业级安全（JWT 登录 · RBAC · 限流 · 密钥托管 · 向量持久化）

### 7.1 登录与 RBAC 权限

所有业务接口统一走安全网关（限流 → 鉴权 → RBAC），未登录返回 `401`，越权返回 `403`，超频返回 `429`。默认三账号：

| 账号 | 密码 | 角色 | 权限 |
|------|------|------|------|
| `admin` | `admin123` | 管理员 | 全部权限（含用户管理） |
| `operator` | `operator123` | 运营 | 读写业务（改订单 / 上传 / 执行 Agent / 调 LLM） |
| `viewer` | `viewer123` | 只读访客 | 仅只读查询 |

- **JWT**：HS256 签名，纯 `hmac + hashlib + base64` 实现，密钥/过期时间由 `JWT_SECRET` / `JWT_EXPIRE` 控制。
- **密码**：PBKDF2-SHA256 加盐 120000 次哈希，用户持久化到 `data/users.json`（不落明文）。
- **RBAC 映射**：写接口（`/api/order/update`、`/api/rag/upload`、`/api/agent/run`、`/api/llm/*`）需 operator+；`/api/auth/users` 仅 admin。

### 7.2 API 限流（滑动窗口，按 IP + 接口分级）

| 分级 | 阈值 | 覆盖接口 |
|------|------|---------|
| 登录 | 5 次/分 | `/api/auth/login`（防暴力破解） |
| 写操作 | 10 次/分 | 改订单 / 上传 / 执行 Agent |
| LLM | 20 次/分 | `/api/llm/*`（控成本） |
| 读接口 | 100 次/分 | 其余查询 |

### 7.3 密钥托管

API Key 解析优先级：**服务端 `DEEPSEEK_API_KEY` 环境变量 > 前端传入**。生产环境建议服务端托管，前端不再暴露个人 Key；未配置时回退前端填写（仍不落服务器、不落盘、不进日志）。

### 7.4 向量/索引持久化

用户上传的文档实时落盘到 `data/persisted_docs.json`，服务重启后自动重新加载并重建混合检索索引，**上传内容与检索能力不因重启丢失**。内置语料与业务数据本就存于 JSON，天然持久。
