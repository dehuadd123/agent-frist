"""工具注册表：把外部系统能力统一封装为可调用工具（Function Calling 契约）。

设计要点：
- 每个工具声明 JSON Schema（OpenAI function calling 的 parameters 格式），供 LLM 决定调用。
- 副作用分级：read（只读，可并发/重试/无幂等负担） vs write（写，必须幂等）。
- 写工具一律要求携带 request_id / 业务键，由编排器保证「最多执行一次」。

真实业务中，func 内部应替换为 HTTP/RPC 调用；此处为演示用模拟实现（print 副作用 + 返回值）。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import business as _business
import model_eng as _model_eng

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "data", "business_data.json"), encoding="utf-8") as _f:
    _BIZ = json.load(_f)

# 订单库：加载到内存（模拟订单中心，状态机跨请求保持）
_ORDER_DB = {o["order_id"]: dict(o) for o in _BIZ.get("orders", [])}
APPROVAL_RULES = _BIZ.get("approval_rules", {})

# 退款原因枚举 -> 中文
REFUND_REASONS = {"quality": "质量问题", "logistics": "物流问题", "no_reason": "无理由退货"}


# =====================================================================
# 订单状态机（实时状态管理）：合法流转 + 中文映射 + 落盘持久化
# =====================================================================
ORDER_STATUS = ("paid", "shipped", "delivered", "refunded", "cancelled")
ORDER_STATUS_CN = {
    "paid": "已付款/待发货", "shipped": "已发货", "delivered": "已签收",
    "refunded": "已退款", "cancelled": "已取消",
}
# 合法状态流转：当前状态 -> 允许进入的下一状态（终态 refunded/cancelled 不可再变）
_VALID_TRANSITIONS = {
    "paid": ("shipped", "refunded", "cancelled"),
    "shipped": ("delivered", "refunded"),
    "delivered": ("refunded",),
    "refunded": (),
    "cancelled": (),
}
_ORDERS_RUNTIME = os.path.join(_HERE, "data", "orders_runtime.json")


def _persist_orders():
    """把内存订单库落盘，使状态实时更新在服务重启后依然生效。"""
    try:
        with open(_ORDERS_RUNTIME, "w", encoding="utf-8") as f:
            json.dump(list(_ORDER_DB.values()), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"      [工具] 订单状态落盘失败: {e}")


def _load_orders_runtime():
    """启动时加载运行时订单快照，覆盖内置订单，实现跨重启状态保持。"""
    if not os.path.exists(_ORDERS_RUNTIME):
        return
    try:
        with open(_ORDERS_RUNTIME, encoding="utf-8") as f:
            for o in json.load(f):
                _ORDER_DB[o["order_id"]] = o
        print(f"      [工具] 已加载运行时订单状态（{len(_ORDER_DB)} 单）")
    except Exception:
        pass


_load_orders_runtime()


class Tool:
    def __init__(self, name, fn, description, parameters, side_effect="read",
                 idempotent=True, compensate=None):
        assert side_effect in ("read", "write")
        self.name = name
        self.fn = fn
        self.description = description
        self.parameters = parameters          # JSON Schema (dict)
        self.side_effect = side_effect
        self.idempotent = idempotent          # 写工具是否具备幂等能力
        self.compensate = compensate          # 写工具的反向补偿函数（Saga 用）

    def call(self, **kwargs):
        return self.fn(**kwargs)

    def to_openai(self):
        """导出为 OpenAI function calling 格式的 tools 声明。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool
        return tool

    def get(self, name):
        return self._tools[name]

    def to_openai_tools(self):
        return [t.to_openai() for t in self._tools.values()]

    def dump_json(self):
        return json.dumps(self.to_openai_tools(), ensure_ascii=False, indent=2)

    def table(self):
        """副作用分级 + 幂等要求总览。"""
        rows = []
        for t in self._tools.values():
            rows.append({
                "工具": t.name,
                "副作用": t.side_effect.upper(),
                "幂等": "是" if t.idempotent else "否（需 request_id 去重）",
                "描述": t.description,
            })
        return rows


# =====================================================================
# 通用注册表（三个 Agent 共用）
# =====================================================================
REGISTRY = ToolRegistry()


def register(name, fn, description, parameters, side_effect="read", idempotent=True,
             compensate=None):
    return REGISTRY.register(
        Tool(name, fn, description, parameters, side_effect, idempotent, compensate))


# =====================================================================
# 场景 1：电商客服 —— 工具
# =====================================================================
def tool_query_order(order_id: str):
    """查询订单（真实订单库）。真实实现：调订单中心 HTTP API。"""
    order = _ORDER_DB.get(order_id)
    if not order:
        print(f"      [工具] query_order({order_id}) -> 订单不存在")
        return {"id": order_id, "status": "not_found", "amount": 0}
    print(f"      [工具] query_order({order_id}) -> {order['product_name']} ¥{order['amount']} [{order['status']}]")
    return dict(order)


def tool_check_refund_policy(order: dict, reason: str = "no_reason"):
    """退款规则引擎（确定性规则，参考主流电商售后流程）。

    返回 {decision: auto|manual|reject, message, reason}
    规则：
    - 已退款 -> reject（重复退款防护）
    - 金额 > 5000 -> manual（风控复核）
    - 未发货(paid) -> auto（全额退款）
    - 已发货未签收(shipped)：质量/物流 -> auto；无理由 -> manual
    - 已签收(delivered)：质量/物流 -> manual（验货）；无理由：≤7天 auto（扣运费），>7天 reject
    """
    status = order.get("status")
    amount = order.get("amount", 0)
    if status == "refunded":
        return {"decision": "reject", "message": "该订单已退款，不可重复退款", "reason": reason}
    if status == "not_found":
        return {"decision": "reject", "message": "订单不存在，拒绝退款", "reason": reason}
    if amount > 5000:
        return {"decision": "manual", "message": "金额超过 ¥5000，转人工风控复核", "reason": reason}
    if status == "paid":
        return {"decision": "auto", "message": "订单未发货，全额退款到原支付渠道", "reason": reason}
    if status == "shipped":
        if reason in ("quality", "logistics"):
            return {"decision": "auto", "message": "已发货未签收，%s全额退款" % REFUND_REASONS[reason], "reason": reason}
        return {"decision": "manual", "message": "已发货订单无理由退款，需人工核实物流", "reason": reason}
    if status == "delivered":
        if reason in ("quality", "logistics"):
            return {"decision": "manual", "message": "已签收商品%s，需人工验货后退款" % REFUND_REASONS[reason], "reason": reason}
        # 无理由退货：计算签收至今的天数
        try:
            from datetime import datetime
            sign = datetime.strptime(order.get("sign_time") or "", "%Y-%m-%d %H:%M")
            days = (datetime.now() - sign).days
        except Exception:
            days = 0
        if days <= 7:
            return {"decision": "auto", "message": "签收 %d 天，符合 7 天无理由退货，自动退款（扣除运费）" % days, "reason": reason}
        return {"decision": "reject", "message": "签收 %d 天，超出 7 天无理由退货期，拒绝退款" % days, "reason": reason}
    return {"decision": "manual", "message": "未知订单状态，转人工", "reason": reason}


def tool_refund(order_id: str, request_id: str, reason: str):
    """执行退款（写，资金操作）。状态机置为 refunded，重复调用幂等返回首次结果。"""
    order = _ORDER_DB.get(order_id)
    if order and order.get("status") == "refunded":
        print(f"      [工具] refund({order_id}) 已退款，幂等返回")
        return {"refund_id": "RF-" + request_id[-4:], "status": "refunded", "duplicate": True}
    if order:
        order["status"] = "refunded"
        order["refund_reason"] = reason
    refund_id = "RF-" + request_id[-4:]
    print(f"      [工具] refund({order_id}, req={request_id}) 已退款到原支付渠道 -> {refund_id}")
    return {"refund_id": refund_id, "status": "refunded"}


def tool_create_manual_task(order_id: str, reason: str):
    """转人工审批（写）。"""
    print(f"      [工具] create_manual_task({order_id}) 已转人工审批队列")
    return {"task_id": "MT-" + order_id, "status": "pending_review"}


def tool_notify_user(user_id: str, message: str):
    """通知用户（写）。"""
    print(f"      [工具] notify_user({user_id}) -> {message}")
    return {"sent": True}


def tool_list_orders():
    """列出全部订单及其「实时」状态（只读）。"""
    rows = []
    for oid in sorted(_ORDER_DB):
        o = _ORDER_DB[oid]
        st = o.get("status")
        rows.append({
            "order_id": oid, "user_id": o.get("user_id"),
            "product_name": o.get("product_name"), "amount": o.get("amount"),
            "status": st, "status_cn": ORDER_STATUS_CN.get(st, st),
            "pay_time": o.get("pay_time"), "ship_time": o.get("ship_time"),
            "sign_time": o.get("sign_time"), "refund_time": o.get("refund_time"),
            "cancel_time": o.get("cancel_time"),
        })
    return {"orders": rows, "total": len(rows)}


def tool_update_order_status(order_id: str, new_status: str):
    """实时更新订单状态（写）。状态机校验 + 关联时间戳自动联动 + 落盘持久化。

    合法流转：paid -> shipped/refunded/cancelled；shipped -> delivered/refunded；
             delivered -> refunded；refunded/cancelled 为终态不可再变。
    """
    order = _ORDER_DB.get(order_id)
    if not order:
        return {"ok": False, "error": "订单不存在: %s" % order_id}
    cur = order.get("status")
    if new_status not in ORDER_STATUS:
        return {"ok": False, "error": "非法状态: %s（可选: %s）" % (new_status, "、".join(ORDER_STATUS))}
    if new_status == cur:
        return {"ok": True, "changed": False, "order_id": order_id, "status": cur,
                "status_cn": ORDER_STATUS_CN.get(cur, cur)}
    if new_status not in _VALID_TRANSITIONS.get(cur, ()):
        allow = "、".join(_VALID_TRANSITIONS.get(cur, ())) or "无（终态）"
        return {"ok": False, "error": "非法状态流转: %s -> %s（允许: %s）" % (cur, new_status, allow)}
    # 更新状态 + 关联时间戳
    now = time.strftime("%Y-%m-%d %H:%M")
    order["status"] = new_status
    if new_status == "shipped":
        order["ship_time"] = now
    elif new_status == "delivered":
        order["sign_time"] = now
    elif new_status == "refunded":
        order["refund_time"] = now
    elif new_status == "cancelled":
        order["cancel_time"] = now
    _persist_orders()
    print(f"      [工具] update_order_status({order_id}) {cur} -> {new_status}")
    return {"ok": True, "changed": True, "order_id": order_id, "from": cur,
            "to": new_status, "status_cn": ORDER_STATUS_CN.get(new_status, new_status)}


register("query_order", tool_query_order, "查询订单详情（金额/状态/商品），用于退款审批判定",
         {"type": "object",
          "properties": {"order_id": {"type": "string", "description": "订单号"}},
          "required": ["order_id"]}, side_effect="read")

register("check_refund_policy", tool_check_refund_policy,
         "退款规则引擎：按订单状态+金额+原因判定 auto/manual/reject",
         {"type": "object",
          "properties": {"order": {"type": "object"},
                         "reason": {"type": "string", "enum": ["quality", "logistics", "no_reason"],
                                    "description": "退款原因：质量问题/物流问题/无理由退货"}},
          "required": ["order"]}, side_effect="read")

register("refund", tool_refund, "执行退款到原支付渠道（资金写操作，幂等）",
         {"type": "object",
          "properties": {"order_id": {"type": "string"},
                         "request_id": {"type": "string", "description": "幂等键，重复请求去重"},
                         "reason": {"type": "string"}},
          "required": ["order_id", "request_id"]}, side_effect="write", idempotent=True)

register("create_manual_task", tool_create_manual_task, "创建人工审批任务",
         {"type": "object",
          "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["order_id"]}, side_effect="write", idempotent=True)

register("notify_user", tool_notify_user, "发送消息通知用户",
         {"type": "object",
          "properties": {"user_id": {"type": "string"}, "message": {"type": "string"}},
          "required": ["user_id", "message"]}, side_effect="write", idempotent=True)

register("list_orders", tool_list_orders, "列出全部订单及实时状态（只读）",
         {"type": "object", "properties": {}}, side_effect="read")

register("update_order_status", tool_update_order_status,
         "实时更新订单状态（状态机校验 + 时间戳联动 + 落盘）",
         {"type": "object",
          "properties": {"order_id": {"type": "string", "description": "订单号"},
                         "new_status": {"type": "string", "enum": list(ORDER_STATUS),
                                        "description": "目标状态"}},
          "required": ["order_id", "new_status"]}, side_effect="write", idempotent=True)


# =====================================================================
# 场景 2：金融投研 —— 工具
# =====================================================================
# RAG 检索注入点：由 server.py 在启动时注入真实的混合检索函数，
# 使 Agent 的工具调用真正落到 RAG 知识底座上（而非硬编码字符串）。
_RAG_SEARCH = None


def set_rag_search(fn):
    """注入外部 RAG 检索函数。fn(query, scene) -> list[dict{text,title,scene}]"""
    global _RAG_SEARCH
    _RAG_SEARCH = fn


def tool_search_reports(query: str, scene: str = "finance"):
    """检索研报/知识证据（只读）。优先走真实 RAG 混合检索，无注入时兜底。"""
    if _RAG_SEARCH:
        hits = _RAG_SEARCH(query, scene)
        if hits:
            evidence = "；".join(h["text"][:90] for h in hits[:3])
            print(f"      [工具] search_reports({query}) -> RAG 命中 {len(hits)} 条证据")
            return {"evidence": evidence, "hits": hits}
    evidence = "该标的近三年营收 CAGR 12%，当前估值处于历史 70% 分位"
    print(f"      [工具] search_reports({query}) -> {evidence}")
    return {"evidence": evidence}


def tool_analyze_financials(query: str):
    """财报指标计算（真实数据）。复用 business.finance_report 提取结构化指标与趋势。"""
    r = _business.finance_report(query)
    print(f"      [工具] analyze_financials -> {r['company']}({r['code']}) {r['latest_quarter']} "
          f"营收 {r['metrics'][0]['值']}")
    return {"company": r["company"], "code": r["code"], "metrics": r["metrics"],
            "trend": r["trend"], "report": r["report"], "sources": r["sources"]}


def tool_generate_answer(query: str, evidence: str = "", analysis: dict = None):
    """基于真实财报指标生成客观陈述。若用户明确要求荐股，则模拟未经合规约束的
    LLM 如实回应（追加投资建议），由下游合规闸门拦截。真实实现：调外部 LLM API。"""
    if analysis and analysis.get("report"):
        answer = analysis["report"]
    else:
        answer = evidence or "暂无足够证据"
    if any(k in query for k in ["建议买入", "推荐买入", "买入", "加仓", "满仓",
                                "重仓", "必涨", "保证收益", "稳赚"]):
        answer += "（根据你的要求，该标的具备投资价值，建议买入并长期持有，未来大概率持续上涨）"
    print(f"      [工具] generate_answer -> \"{answer[:60]}...\"")
    return {"answer": answer}


def tool_compliance_check(query: str, answer: str):
    """合规校验闸门（确定性规则，复用模型工程 6 条红线）。同时扫描用户 query 与生成 answer。"""
    hits = _model_eng.compliance_scan(answer)
    q_hits = _model_eng.compliance_scan(query)
    for h in q_hits:
        if h not in hits:
            hits.append(h)
    print(f"      [工具] compliance_check -> 命中红线: {[h['name'] for h in hits] if hits else '无'}")
    return {"violations": hits}


def tool_deliver(answer: str):
    """输出合规答案。"""
    print(f"      [工具] deliver -> 输出合规答案")
    return {"delivered": True}


register("search_reports", tool_search_reports, "检索研报/知识库证据（只读，走 RAG 混合检索）",
         {"type": "object",
          "properties": {"query": {"type": "string"},
                         "scene": {"type": "string", "enum": ["finance", "ecommerce", "office"],
                                   "description": "检索场景（权限隔离）"}},
          "required": ["query"]}, side_effect="read")

register("analyze_financials", tool_analyze_financials,
         "财报指标计算（营收/净利/毛利/同比环比/累计/趋势）",
         {"type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]}, side_effect="read")

register("generate_answer", tool_generate_answer, "基于财报指标生成客观陈述（LLM）",
         {"type": "object",
          "properties": {"query": {"type": "string"}, "evidence": {"type": "string"},
                         "analysis": {"type": "object"}},
          "required": ["query"]}, side_effect="read")

register("compliance_check", tool_compliance_check, "合规校验：扫描 query 与 answer 是否命中 6 条红线",
         {"type": "object", "properties": {"query": {"type": "string"}, "answer": {"type": "string"}},
          "required": ["answer"]}, side_effect="read")

register("deliver", tool_deliver, "输出最终答案",
         {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
         side_effect="read")


# =====================================================================
# 场景 3：企业办公 —— 工具（含补偿，Saga 用）
# =====================================================================
def tool_parse_approval(approval_type: str, params: dict = None):
    """审批类型识别 + 分级审批链计算（参考钉钉/飞书审批流）。

    approval_type: leave(请假) / expense(报销) / trip(出差)
    params: 请假/出差传 {days: 天数}，报销传 {amount: 金额}
    返回 {type, level, approvers, ...}
    """
    params = params or {}
    rules = APPROVAL_RULES.get(approval_type, [])
    if approval_type == "leave":
        days = int(params.get("days", 1))
        for r in rules:
            if days <= r["max_days"]:
                return {"type": "请假", "level": r["level"], "days": days,
                        "approvers": r["approvers"], "rule": "请假 %d 天" % days}
    elif approval_type == "expense":
        amount = float(params.get("amount", 0))
        for r in rules:
            if amount <= r["max_amount"]:
                return {"type": "报销", "level": r["level"], "amount": amount,
                        "approvers": r["approvers"], "rule": "报销 ¥%g" % amount}
    elif approval_type == "trip":
        days = int(params.get("days", 1))
        for r in rules:
            if days <= r["max_days"]:
                return {"type": "出差", "level": r["level"], "days": days,
                        "approvers": r["approvers"], "rule": "出差 %d 天" % days}
    return {"type": approval_type, "level": "P1", "approvers": ["直属主管"], "rule": "默认"}


def tool_create_approval(approval_id: str, applicant: str, approval_type: str = "leave",
                         params: dict = None):
    """创建审批单（写）。按审批类型计算分级审批链，写入审批单。"""
    chain = tool_parse_approval(approval_type, params)
    print(f"      [工具] create_approval({approval_id}) 类型={chain['type']} 等级={chain['level']} "
          f"审批链={' → '.join(chain['approvers'])}")
    return {"approval_id": approval_id, "chain": chain}


def tool_undo_approval(approval_id: str):
    print(f"      [工具] undo_approval({approval_id}) 已撤销审批单")


def tool_prepare_attachment(approval_id: str):
    """准备审批附件（只读/独立，可与审批单创建并行）。"""
    print(f"      [工具] prepare_attachment({approval_id}) 附件已就绪")
    return {"attachment_ready": True}


def tool_add_calendar(approval_id: str, event_id: str):
    print(f"      [工具] add_calendar({event_id}) 已写入日历事件")
    return {"event_id": event_id}


def tool_undo_calendar(event_id: str):
    print(f"      [工具] undo_calendar({event_id}) 已删除日历事件")


def tool_send_notice(approval_id: str, notice_id: str, fail: bool = False):
    """发送通知（可注入失败用于演示 Saga 补偿）。"""
    if fail:
        raise RuntimeError("通知服务不可用")
    print(f"      [工具] send_notice({notice_id}) 已发送通知")
    return {"notice_id": notice_id}


def tool_undo_notice(notice_id: str):
    print(f"      [工具] undo_notice({notice_id}) 已撤回通知")


register("parse_approval", tool_parse_approval, "审批类型识别 + 分级审批链计算",
         {"type": "object",
          "properties": {"approval_type": {"type": "string", "enum": ["leave", "expense", "trip"]},
                         "params": {"type": "object"}},
          "required": ["approval_type"]}, side_effect="read")

register("create_approval", tool_create_approval, "创建审批单（写），按类型计算分级审批链",
         {"type": "object",
          "properties": {"approval_id": {"type": "string"}, "applicant": {"type": "string"},
                         "approval_type": {"type": "string", "enum": ["leave", "expense", "trip"]},
                         "params": {"type": "object"}},
          "required": ["approval_id"]}, side_effect="write", compensate=tool_undo_approval)

register("prepare_attachment", tool_prepare_attachment, "准备审批附件（只读）",
         {"type": "object", "properties": {"approval_id": {"type": "string"}},
          "required": ["approval_id"]}, side_effect="read")

register("add_calendar", tool_add_calendar, "写入日历事件（写）",
         {"type": "object",
          "properties": {"approval_id": {"type": "string"}, "event_id": {"type": "string"}},
          "required": ["event_id"]}, side_effect="write", compensate=tool_undo_calendar)

register("send_notice", tool_send_notice, "发送通知（写）",
         {"type": "object",
          "properties": {"approval_id": {"type": "string"}, "notice_id": {"type": "string"},
                         "fail": {"type": "boolean"}},
          "required": ["notice_id"]}, side_effect="write", compensate=tool_undo_notice)


if __name__ == "__main__":
    print("=== 工具注册表（Function Calling 声明）===")
    print(REGISTRY.dump_json())
    print("\n=== 副作用分级总览 ===")
    for r in REGISTRY.table():
        print(f"  {r['副作用']:<5} | 幂等:{r['幂等']:<12} | {r['工具']:<18} | {r['描述']}")
