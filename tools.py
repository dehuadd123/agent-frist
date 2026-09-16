"""工具注册表：把外部系统能力统一封装为可调用工具（Function Calling 契约）。

设计要点：
- 每个工具声明 JSON Schema（OpenAI function calling 的 parameters 格式），供 LLM 决定调用。
- 副作用分级：read（只读，可并发/重试/无幂等负担） vs write（写，必须幂等）。
- 写工具一律要求携带 request_id / 业务键，由编排器保证「最多执行一次」。

真实业务中，func 内部应替换为 HTTP/RPC 调用；此处为演示用模拟实现（print 副作用 + 返回值）。
"""
import json


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
def tool_query_order(order_id: str, amount: int = 300):
    """查询订单（模拟）。真实实现：调订单中心 HTTP API。"""
    # 模拟订单数据（金额可传入，默认 300 元，未退款）
    order = {"id": order_id, "amount": amount, "status": "paid"}
    print(f"      [工具] query_order({order_id}) -> 金额 ¥{order['amount']}")
    return order


def tool_refund(order_id: str, request_id: str, reason: str):
    """执行退款（写，资金操作）。以 request_id 做幂等键，重复调用返回首次结果。"""
    print(f"      [工具] refund({order_id}, req={request_id}) 已退款到原支付渠道")
    return {"refund_id": "RF-" + request_id[-4:], "status": "refunded"}


def tool_create_manual_task(order_id: str, reason: str):
    """转人工审批（写）。"""
    print(f"      [工具] create_manual_task({order_id}) 已转人工审批队列")
    return {"task_id": "MT-1001", "status": "pending_review"}


def tool_notify_user(user_id: str, message: str):
    """通知用户（写）。"""
    print(f"      [工具] notify_user({user_id}) -> {message}")
    return {"sent": True}


register("query_order", tool_query_order, "查询订单详情（金额/状态），用于退款审批判定",
         {"type": "object",
          "properties": {"order_id": {"type": "string", "description": "订单号"},
                         "amount": {"type": "integer", "description": "订单金额（元），用于分支判定"}},
          "required": ["order_id"]}, side_effect="read")

register("refund", tool_refund, "执行退款到原支付渠道（资金写操作）",
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


def tool_generate_answer(query: str, evidence: str):
    """LLM 生成答案（模拟）。真实实现：调外部 LLM API。"""
    answer = f"建议买入，{evidence}，未来必将持续上涨。"
    print(f"      [工具] generate_answer -> \"{answer}\"")
    return {"answer": answer}


def tool_compliance_check(answer: str):
    """合规校验闸门（确定性规则，非 LLM）。命中红线返回违规列表。"""
    red_lines = ["建议买入", "必将", "保证收益", "稳赚", "无风险"]
    hits = [kw for kw in red_lines if kw in answer]
    print(f"      [工具] compliance_check -> 命中红线: {hits if hits else '无'}")
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

register("generate_answer", tool_generate_answer, "基于证据生成投研答案（LLM）",
         {"type": "object",
          "properties": {"query": {"type": "string"}, "evidence": {"type": "string"}},
          "required": ["query", "evidence"]}, side_effect="read")

register("compliance_check", tool_compliance_check, "合规校验：检测答案是否命中红线",
         {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
         side_effect="read")

register("deliver", tool_deliver, "输出最终答案",
         {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
         side_effect="read")


# =====================================================================
# 场景 3：企业办公 —— 工具（含补偿，Saga 用）
# =====================================================================
def tool_create_approval(approval_id: str, applicant: str):
    print(f"      [工具] create_approval({approval_id}) 已创建审批单")
    return {"approval_id": approval_id}


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


register("create_approval", tool_create_approval, "创建审批单（写）",
         {"type": "object",
          "properties": {"approval_id": {"type": "string"}, "applicant": {"type": "string"}},
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
