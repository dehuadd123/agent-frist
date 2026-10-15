"""真实 LLM Function Calling 闭环（OpenAI 兼容协议，零依赖）。

与「确定性 DAG Agent」不同，这里让 LLM 自主决定调用哪些工具、以什么参数调用，
多轮循环直到产出最终答案：

   用户问题
     └─> LLM(messages, tools) ── 返回 tool_calls?
             ├─ 否：返回最终 answer（结束）
             └─ 是：逐条执行工具 -> 把 tool 结果回填 messages -> 再调 LLM（循环）

关键工程点：
- 工具声明复用 tools.REGISTRY.to_openai_tools()，执行复用同一套 Tool.call()，保证「声明/执行」一致。
- 写工具（side_effect=write）默认受 request_id 幂等保护，且可被 HITL 审核队列拦截。
- 限制最大轮数与单轮工具数，防止死循环/失控成本。
- 无 API Key 时由调用方降级到确定性路由（见 server.py），本模块只负责真实 LLM 路径。
"""
import json

MAX_ROUNDS = 6          # 最多 LLM 推理轮次（含工具调用）
MAX_TOOLS_PER_ROUND = 4  # 单轮最多并行工具数


def normalize_tool_calls(raw):
    """把不同 LLM 返回的 tool_calls 归一化为 [{id, name, arguments(dict)}]。"""
    out = []
    for tc in raw or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args_raw = fn.get("arguments") or "{}"
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            try:
                args = json.loads(args_raw)
            except (TypeError, ValueError):
                args = {}
        out.append({"id": tc.get("id") or "", "name": name, "arguments": args})
    return out


def run_agent_loop(chat_fn, messages, tools, tool_executor, max_rounds=MAX_ROUNDS):
    """执行多轮 function calling 循环。

    参数：
      chat_fn(messages, tools) -> {"content": str, "tool_calls": [...], "error": str}
      messages: 初始 messages（含 system/user）
      tools: OpenAI 格式 tools 声明列表
      tool_executor(name, arguments_dict) -> dict  (执行结果，会被 json 序列化回填)

    返回 {answer, rounds, tool_trace:[{round, name, arguments, result, ok}]}
    """
    msgs = [dict(m) for m in messages]
    tool_trace = []

    for rnd in range(1, max_rounds + 1):
        resp = chat_fn(msgs, tools) if tools else chat_fn(msgs)
        if resp.get("error"):
            return {"answer": "", "error": resp["error"], "rounds": rnd, "tool_trace": tool_trace}

        tool_calls = normalize_tool_calls(resp.get("tool_calls"))

        if not tool_calls:
            # 无工具调用 -> 最终回答
            return {"answer": resp.get("content") or "", "rounds": rnd, "tool_trace": tool_trace}

        # 把 assistant 的 tool_calls 消息原样回填（保持协议完整性）
        msgs.append({
            "role": "assistant",
            "content": resp.get("content") or None,
            "tool_calls": resp.get("tool_calls") or [],
        })

        for tc in tool_calls[:MAX_TOOLS_PER_ROUND]:
            try:
                result = tool_executor(tc["name"], tc["arguments"])
                ok = True
            except Exception as e:
                result = {"error": str(e)}
                ok = False
            msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
            tool_trace.append({"round": rnd, "name": tc["name"],
                               "arguments": tc["arguments"], "result": result, "ok": ok})

    # 达到最大轮次仍未收敛
    return {"answer": resp.get("content") or "", "rounds": max_rounds,
            "tool_trace": tool_trace, "truncated": True}


def build_tool_executor(registry, guard=None):
    """由 ToolRegistry 构造 tool_executor(name, args) -> dict。

    guard(name, args) 可选：返回 (allow, reason)。用于写工具前的 HITL/权限二次校验。
    """
    def _exec(name, arguments):
        tool = registry.get(name)
        if guard is not None:
            allow, reason = guard(name, arguments)
            if not allow:
                return {"blocked": True, "reason": reason}
        return tool.call(**arguments)
    return _exec


def build_system_prompt(scene_hint=""):
    """构造 function calling 的 system prompt，约束 LLM 只调用已声明工具、优先证据。"""
    return (
        "你是企业级智能助手，可调用系统提供的工具获取真实数据后作答。\n"
        "规则：1) 只能调用工具列表中声明的工具，禁止编造不存在的数据；\n"
        "2) 需要订单/商品/研报/制度等具体信息时，必须先调用对应工具；\n"
        "3) 金融/投资类问题只陈述事实，不给出买卖建议；\n"
        "4) 中文作答，条理清晰，引用工具返回的关键数据。"
        + ("\n当前业务场景：%s" % scene_hint if scene_hint else "")
    )
