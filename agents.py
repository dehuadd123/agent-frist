"""三个业务 Agent：把工具（tools.py）组合成确定性 DAG 工作流。

- 电商客服 Agent：退款审批（条件分支 + 幂等）
- 金融投研 Agent：合规熔断（红线闸门 + 兜底免责声明）
- 企业办公 Agent：跨系统 Saga（并行汇聚 + 逆序补偿）
- 通用演示：节点级重试退避 + 超时降级
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import Workflow, CircuitBreaker
import tools


# =====================================================================
# 场景 1：电商客服 Agent —— 退款审批（条件分支 + 幂等）
# =====================================================================
def build_ecommerce_agent():
    def node_query_order(ctx):
        ctx["order"] = tools.tool_query_order(ctx["order_id"], amount=ctx.get("amount", 300))

    def node_check_policy(ctx):
        # 确定性规则（非 LLM）：金额 < 500 自动退款，否则转人工
        return "auto" if ctx["order"]["amount"] < 500 else "manual"

    def node_auto_refund(ctx):
        ctx["result"] = tools.tool_refund(ctx["order_id"], ctx["request_id"], "用户申请退款")
        ctx["message"] = "已自动退款到原支付渠道"

    def node_manual_approve(ctx):
        ctx["result"] = tools.tool_create_manual_task(ctx["order_id"], "金额超阈值")
        ctx["message"] = "已转人工审批，待风控复核"

    def node_notify(ctx):
        tools.tool_notify_user(ctx["user_id"], ctx["message"])

    return (
        Workflow("电商客服Agent", start="query_order")
        .add_node("query_order", node_query_order, next="check_policy")
        .add_node("check_policy", node_check_policy)  # 条件分支节点
        .add_node("auto_refund", node_auto_refund, next="notify")
        .add_node("manual_approve", node_manual_approve, next="notify")
        .add_node("notify", node_notify)
        .branch("check_policy", {"auto": "auto_refund", "manual": "manual_approve"})
    )


# =====================================================================
# 场景 2：金融投研 Agent —— 合规熔断（红线闸门 + 兜底）
# =====================================================================
def build_finance_agent():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    def node_retrieve(ctx):
        res = tools.tool_search_reports(ctx["query"], scene="finance")
        ctx["evidence"] = res["evidence"]
        ctx["sources"] = [h.get("title", "") for h in res.get("hits", [])]

    def node_generate(ctx):
        ctx["answer"] = tools.tool_generate_answer(ctx["query"], ctx["evidence"])["answer"]

    def node_check(ctx):
        violations = tools.tool_compliance_check(ctx["answer"])["violations"]
        if violations:
            raise ValueError(f"命中合规红线：{violations[0]}")
        return "pass"

    def node_deliver(ctx):
        tools.tool_deliver(ctx["answer"])

    def node_reject(ctx):
        ctx["answer"] = "（合规免责声明）本内容仅供参考，不构成投资建议，市场有风险，投资需谨慎。"
        print("      [兜底] 输出被拦截，返回标准免责声明")

    def guarded_check(ctx):
        return breaker.call(node_check, ctx)

    wf = (
        Workflow("金融投研Agent", start="retrieve")
        .add_node("retrieve", node_retrieve, next="generate")
        .add_node("generate", node_generate, next="check")
        .add_node("check", guarded_check, fallback=node_reject)  # 合规闸门带兜底
        .add_node("deliver", node_deliver)
        .branch("check", {"pass": "deliver"})
    )
    wf.breaker = breaker  # 暴露熔断器，供服务端/前端读取状态
    return wf


# =====================================================================
# 场景 3：企业办公 Agent —— 跨系统 Saga（并行汇聚 + 逆序补偿）
# =====================================================================
def build_office_agent():
    def node_create(ctx):
        tools.tool_create_approval(ctx["approval_id"], ctx["applicant"])

    def node_undo_create(ctx):
        tools.tool_undo_approval(ctx["approval_id"])

    def node_prepare(ctx):
        tools.tool_prepare_attachment(ctx["approval_id"])

    def node_calendar(ctx):
        tools.tool_add_calendar(ctx["approval_id"], ctx["event_id"])

    def node_undo_calendar(ctx):
        tools.tool_undo_calendar(ctx["event_id"])

    def node_notice(ctx):
        tools.tool_send_notice(ctx["approval_id"], ctx["notice_id"],
                               fail=ctx.get("fail_notice", False))

    def node_undo_notice(ctx):
        tools.tool_undo_notice(ctx["notice_id"])

    return (
        Workflow("企业办公Agent", start="create_approval")
        # 两个无依赖节点：并行执行
        .add_node("create_approval", node_create, compensate=node_undo_create)
        .add_node("prepare_attachment", node_prepare)
        # 汇聚节点：等上面两个都完成后才执行
        .add_node("add_calendar", node_calendar, compensate=node_undo_calendar,
                  depends_on=["create_approval", "prepare_attachment"])
        .add_node("send_notice", node_notice, compensate=node_undo_notice,
                  depends_on=["add_calendar"])
    )


# =====================================================================
# 通用演示：节点级重试（指数退避）+ 超时降级
# =====================================================================
def build_resilience_demo():
    call_count = {"n": 0}

    def node_flaky(ctx):
        call_count["n"] += 1
        if call_count["n"] < 3:  # 前两次失败，第三次成功
            raise RuntimeError("上游网关抖动")
        ctx["ok"] = True
        print(f"      [重试] 第 {call_count['n']} 次调用成功")

    def node_slow(ctx):
        time.sleep(1.0)  # 超过 0.3s 超时阈值

    def node_slow_fallback(ctx):
        ctx["slow_result"] = "降级：跳过慢查询，使用缓存结果"
        print("      [超时兜底] 慢查询超时，降级到缓存")

    return (
        Workflow("重试与超时演示", start="flaky")
        .add_node("flaky", node_flaky, retries=2, backoff=0.3, next="slow")
        .add_node("slow", node_slow, timeout=0.3, fallback=node_slow_fallback)
    )


# =====================================================================
# 核心：检索增强问答 Agent（RAG + Agent 闭环）
# 流程：意图识别 -> RAG 混合检索（场景隔离） -> 证据生成 -> 合规闸门(熔断) -> 交付
# 这是「知识底座(RAG) + 决策引擎(Agent)」统一后的企业级问答主线。
# =====================================================================
def build_qa_agent():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    def node_intent(ctx):
        # 意图识别：按 query 关键词路由到对应业务场景（真实实现可用 LLM 分类）。
        # 注意用「子串」匹配，关键词必须能命中真实问法（如"营业收入"含"收入"而非"营收"）。
        q = ctx.get("query", "")
        scene = ctx.get("scene") or ""
        if scene in ("", "all", None):
            if any(k in q for k in ["退款", "订单", "购买", "商品", "售后", "优惠",
                                    "空调", "耳机", "T恤", "面料", "尺码", "降噪", "续航", "能效"]):
                scene = "ecommerce"
            elif any(k in q for k in ["收入", "净利润", "毛利", "估值", "研报", "投资",
                                      "增速", "财报", "同比", "渗透率", "GDP", "CPI", "装机量", "仓位"]):
                scene = "finance"
            else:
                scene = "office"
        ctx["scene"] = scene
        print(f"      [意图] 路由到场景: {ctx['scene']}")
        return ctx["scene"]

    def node_retrieve(ctx):
        res = tools.tool_search_reports(ctx["query"], scene=ctx["scene"])
        ctx["evidence"] = res["evidence"]
        ctx["sources"] = [h.get("title", "") for h in res.get("hits", [])]
        if not res.get("hits"):
            ctx["no_hit"] = True

    def node_generate(ctx):
        if ctx.get("no_hit"):
            ctx["answer"] = "未在知识库中检索到相关证据，建议补充语料或换个问法。"
            return
        # 基于检索证据生成答案（真实实现：把 evidence 拼入 LLM prompt）
        ev = ctx["evidence"]
        ctx["answer"] = f"根据知识库证据：{ev}"
        ctx["answer"] += "。" if not ev.endswith("。") else ""

    def node_check(ctx):
        # 仅金融场景强制合规校验；其他场景直通
        if ctx["scene"] != "finance":
            return "pass"
        violations = tools.tool_compliance_check(ctx["answer"])["violations"]
        if violations:
            raise ValueError(f"命中合规红线：{violations[0]}")
        return "pass"

    def node_deliver(ctx):
        tools.tool_deliver(ctx["answer"])

    def node_reject(ctx):
        ctx["answer"] = "（合规免责声明）本内容仅供参考，不构成投资建议，市场有风险，投资需谨慎。"
        print("      [兜底] 输出被拦截，返回标准免责声明")

    def guarded_check(ctx):
        return breaker.call(node_check, ctx)

    wf = (
        Workflow("检索增强问答Agent", start="intent")
        .add_node("intent", node_intent, next="retrieve")
        .add_node("retrieve", node_retrieve, next="generate")
        .add_node("generate", node_generate, next="check")
        .add_node("check", guarded_check, fallback=node_reject)
        .add_node("deliver", node_deliver)
        .branch("check", {"pass": "deliver"})
    )
    wf.breaker = breaker
    return wf
