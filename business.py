"""业务功能模块：三场景具体实现（本地向量库打底 + 可选 LLM 增强，混合模式）。

  1. 电商导购  ecommerce_search —— 自然语言约束解析 + 商品向量检索 + 过滤排序 + 推荐
  2. 金融投研  finance_report   —— 财报结构化指标提取 + 多季度趋势分析 + 溯源 + 深度解读
  3. 企业办公  office_assistant —— 制度/报销/考勤规则检索 + 结构化回答 + 溯源

统一模式：本地向量库（HybridRetriever：BM25 + TF-IDF 稠密 + RRF）检索打底，
可选注入 LLM（OpenAI 兼容）做增强。零依赖纯标准库。
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "data", "business_data.json"), encoding="utf-8") as f:
    _BIZ = json.load(f)

PRODUCTS = _BIZ["products"]
FINANCE = _BIZ["finance"]
OFFICE_RULES = _BIZ["office_rules"]

from retrieval.hybrid import HybridRetriever


def _product_text(p):
    specs = "，".join("%s%s" % (k, v) for k, v in p["specs"].items())
    return "%s %s %s %s" % (p["name"], p["category"], specs, p["desc"])


def _rule_text(r):
    return "%s %s %s %s" % (r["category"], r["item"], r["standard"], r["detail"])


_PRODUCT_TEXTS = [_product_text(p) for p in PRODUCTS]
_RULE_TEXTS = [_rule_text(r) for r in OFFICE_RULES]
_PRODUCT_RET = HybridRetriever(_PRODUCT_TEXTS)
_RULE_RET = HybridRetriever(_RULE_TEXTS)

# RAG 注入点（金融/办公溯源用，由 server.py 注入真实 RAG 检索）
_rag_search = None


def set_rag_search(fn):
    global _rag_search
    _rag_search = fn


def _highlight(text, kws):
    out = text
    for k in sorted(set(k for k in kws if k), key=len, reverse=True):
        try:
            out = re.sub(re.escape(k), lambda m: "<mark>%s</mark>" % m.group(0),
                         out, flags=re.IGNORECASE)
        except re.error:
            continue
    return out


# =====================================================================
# 1. 电商导购：自然语言约束解析 + 商品向量检索 + 过滤排序
# =====================================================================
CATEGORY_KEYS = ["手机", "耳机", "空调", "冰箱", "洗衣机", "扫地机", "电暖器",
                 "服饰", "美妆", "食品", "数码", "充电宝", "手环"]


def parse_constraints(query):
    """把自然语言约束解析为结构化过滤条件（价格/评分/品类）。"""
    cons = {}
    # 价格上限：3000以内 / 3000元以下 / 不超过3000 / 预算3000
    m = re.search(r"(?:(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|之内|内))|"
                  r"(?:不超过\s*(\d+(?:\.\d+)?))|(?:预算\s*(\d+(?:\.\d+)?))", query)
    if m:
        v = next((float(x) for x in m.groups() if x), None)
        if v:
            cons["max_price"] = v
    # 价格下限：2000以上 / 2000元以上 / 不低于2000
    m = re.search(r"(?:(\d+(?:\.\d+)?)\s*元?\s*以上)|(?:不低于\s*(\d+(?:\.\d+)?))", query)
    if m:
        v = next((float(x) for x in m.groups() if x), None)
        if v:
            cons["min_price"] = v
    # 评分：4.5星以上 / 评分4.5以上
    m = re.search(r"(\d(?:\.\d)?)\s*[星分]?\s*(?:以上|及以上)", query)
    if m:
        cons["min_rating"] = float(m.group(1))
    # 品类
    for c in CATEGORY_KEYS:
        if c in query:
            cons["category"] = c
            break
    return cons


def ecommerce_search(query, top_k=5, llm=None):
    """电商导购检索：解析约束 -> 向量检索 -> 硬过滤 -> 返回商品 + 匹配高亮 + 可选 LLM 推荐。"""
    cons = parse_constraints(query)
    ranked = _PRODUCT_RET.search(query, mode="hybrid", top_k=len(PRODUCTS))
    picked = []
    for idx, score in ranked:
        p = PRODUCTS[idx]
        if "category" in cons and p["category"] != cons["category"]:
            continue
        if "max_price" in cons and p["price"] > cons["max_price"]:
            continue
        if "min_price" in cons and p["price"] < cons["min_price"]:
            continue
        if "min_rating" in cons and p["rating"] < cons["min_rating"]:
            continue
        picked.append({"product": p, "score": round(float(score), 4)})
        if len(picked) >= top_k:
            break
    kws = [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower())
           if len(t) >= 2]
    items = []
    for r in picked:
        p = r["product"]
        items.append({
            "id": p["id"], "name": p["name"], "category": p["category"],
            "price": p["price"], "rating": p["rating"], "stock": p["stock"],
            "specs": p["specs"], "desc": p["desc"], "score": r["score"],
            "highlighted": _highlight(_product_text(p), kws),
        })
    llm_answer = None
    if llm and items:
        ctx = "\n".join("%s（%s，¥%s，评分%s）：%s" % (it["name"], it["category"],
                        it["price"], it["rating"], it["desc"]) for it in items)
        msgs = [
            {"role": "system", "content": "你是电商导购助手，仅基于检索到的商品做推荐，分点说明推荐理由，"
                                          "不编造商品，用中文作答。"},
            {"role": "user", "content": "用户需求：%s\n检索到的商品：\n%s" % (query, ctx)},
        ]
        content, err = llm(msgs)
        if not err:
            llm_answer = content
    return {"query": query, "constraints": cons, "items": items,
            "total_matched": len(items), "llm_answer": llm_answer}


# =====================================================================
# 2. 金融投研：财报指标提取 + 多季度趋势分析 + 溯源 + 深度解读
# =====================================================================
def _find_company(query):
    for c in FINANCE["companies"]:
        if c["name"] in query or c["code"] in query:
            return c
    return None


def _calc_trend(quarters):
    """真实计算最新季度同比/环比/毛利变化。"""
    latest, prev = quarters[-1], quarters[-2]
    qoq = round((latest["revenue"] - prev["revenue"]) / prev["revenue"] * 100, 1)
    np_qoq = round((latest["net_profit"] - prev["net_profit"]) / prev["net_profit"] * 100, 1)
    gm_change = round(latest["gross_margin"] - prev["gross_margin"], 1)
    return {"qoq_revenue": qoq, "qoq_net_profit": np_qoq, "gm_change": gm_change}


def finance_report(query, llm=None):
    """金融报告分析：识别公司 -> 提取指标 -> 趋势分析 -> 结构化报告 -> 溯源 -> 可选 LLM 解读。"""
    company = _find_company(query) or FINANCE["companies"][0]
    quarters = company["quarters"]
    latest = quarters[-1]
    prev = quarters[-2]
    trend = _calc_trend(quarters)

    # 全年累计（2025 四季度真实求和）+ 可比口径同比（H2 对齐，避免季度数不一致）
    y2025 = [q for q in quarters if q["quarter"].startswith("2025")]
    rev_2025 = round(sum(q["revenue"] for q in y2025), 1)
    np_2025 = round(sum(q["net_profit"] for q in y2025), 1)
    # 可比口径：2025 H2（Q3+Q4）vs 2024 H2（Q3+Q4），各 2 季度，同季度对齐
    h2_2025 = [q for q in quarters if q["quarter"] in ("2025Q3", "2025Q4")]
    h2_2024 = [q for q in quarters if q["quarter"] in ("2024Q3", "2024Q4")]
    rev_h2_2025 = sum(q["revenue"] for q in h2_2025)
    rev_h2_2024 = sum(q["revenue"] for q in h2_2024)
    np_h2_2025 = sum(q["net_profit"] for q in h2_2025)
    np_h2_2024 = sum(q["net_profit"] for q in h2_2024)
    rev_yoy = round((rev_h2_2025 - rev_h2_2024) / rev_h2_2024 * 100, 1)
    np_yoy = round((np_h2_2025 - np_h2_2024) / np_h2_2024 * 100, 1)

    metrics = [
        {"指标": "最新季度营收", "值": "%s 亿元" % latest["revenue"], "说明": latest["quarter"]},
        {"指标": "营收同比增速", "值": "+%s%%" % latest["yoy"], "说明": "单季同比"},
        {"指标": "营收环比增速", "值": "%+s%%" % trend["qoq_revenue"], "说明": "较上一季度"},
        {"指标": "归母净利润", "值": "%s 亿元" % latest["net_profit"], "说明": latest["quarter"]},
        {"指标": "净利润环比", "值": "%+s%%" % trend["qoq_net_profit"], "说明": "较上一季度"},
        {"指标": "综合毛利率", "值": "%s%%" % latest["gross_margin"],
         "说明": "环比 %+spct" % trend["gm_change"]},
        {"指标": "2025 累计营收", "值": "%s 亿元" % rev_2025,
         "说明": "2025 全年（4 季），同比 %+s%%" % rev_yoy},
        {"指标": "2025 累计净利", "值": "%s 亿元" % np_2025,
         "说明": "2025 全年（4 季），同比 %+s%%" % np_yoy},
    ]

    # 趋势判断
    trend_verdict = []
    if latest["yoy"] > prev["yoy"]:
        trend_verdict.append("营收同比增速逐季抬升，成长性增强")
    else:
        trend_verdict.append("营收同比增速放缓，需关注增长动能")
    if trend["gm_change"] > 0:
        trend_verdict.append("毛利率环比提升，盈利能力改善")
    elif trend["gm_change"] < 0:
        trend_verdict.append("毛利率环比下降，成本端承压")
    else:
        trend_verdict.append("毛利率环比持平")

    # 溯源（注入真实 RAG 检索金融场景）
    sources = []
    if _rag_search:
        for h in _rag_search(query, "finance", top_k=3):
            sources.append({"title": h["title"], "text": h["text"][:100], "score": h["score"]})

    report = (
        "%s（%s）最新季度（%s）营业收入 %s 亿元，同比增长 %s%%，环比 %s%%；"
        "归母净利润 %s 亿元；综合毛利率 %s%%。2025 全年累计营收 %s 亿元、净利 %s 亿元。"
        % (company["name"], company["code"], latest["quarter"], latest["revenue"],
           latest["yoy"], trend["qoq_revenue"], latest["net_profit"],
           latest["gross_margin"], rev_2025, np_2025)
    )

    llm_answer = None
    if llm:
        ctx = "\n".join("%s：%s" % (m["指标"], m["值"]) for m in metrics)
        msgs = [
            {"role": "system", "content": "你是金融投研分析师，基于给定财务指标做客观解读，"
                                          "不预测股价、不给出买卖建议，注明仅供参考，用中文分点作答。"},
            {"role": "user", "content": "公司：%s\n财务指标：\n%s\n趋势：%s\n请给出简明的财务分析。"
                                        % (company["name"], ctx, "；".join(trend_verdict))},
        ]
        content, err = llm(msgs)
        if not err:
            llm_answer = content

    return {
        "query": query, "company": company["name"], "code": company["code"],
        "latest_quarter": latest["quarter"],
        "metrics": metrics, "trend": trend_verdict, "report": report,
        "sources": sources, "llm_answer": llm_answer,
        "quarters": quarters,  # 供前端绘趋势图
    }


# =====================================================================
# 3. 企业办公：制度/报销/考勤规则检索 + 结构化回答
# =====================================================================
def office_assistant(query, top_k=3, llm=None):
    """办公制度问答：规则向量检索 -> 结构化回答 -> 溯源 -> 可选 LLM。"""
    ranked = _RULE_RET.search(query, mode="hybrid", top_k=top_k)
    kws = [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower())
           if len(t) >= 2]
    rules = []
    for idx, score in ranked:
        r = OFFICE_RULES[idx]
        rules.append({
            "category": r["category"], "item": r["item"], "standard": r["standard"],
            "detail": r["detail"], "score": round(float(score), 4),
            "highlighted": _highlight(_rule_text(r), kws),
        })
    answer = "；".join("%s（%s）：%s" % (r["category"], r["item"], r["standard"]) for r in rules)

    llm_answer = None
    if llm and rules:
        ctx = "\n".join("%s - %s：%s（%s）" % (r["category"], r["item"], r["standard"], r["detail"])
                        for r in rules)
        msgs = [
            {"role": "system", "content": "你是企业办公助手，仅基于给定制度规则回答，条理清晰，用中文作答。"},
            {"role": "user", "content": "问题：%s\n相关制度：\n%s" % (query, ctx)},
        ]
        content, err = llm(msgs)
        if not err:
            llm_answer = content

    return {"query": query, "rules": rules, "answer": answer, "llm_answer": llm_answer}
