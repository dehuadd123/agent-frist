"""企业级 DAG 工作流编排器（零依赖）。

能力清单：
- DAG：`depends_on` 静态依赖 + 拓扑排序 + 环检测 + 并行执行（无依赖同层并发）+ 汇聚
- 条件分支：`branch({分支值: 目标})`，运行时按节点返回值路由，未选中分支自动跳过（skip 传播）
- 异常兜底：节点级重试（指数退避）、超时、fallback 降级
- Saga 补偿：节点 `compensate` + 逆序回滚
- 熔断器：CLOSED / OPEN / HALF_OPEN
- 幂等：业务键 + 状态机(PENDING/RUNNING/SUCCESS/FAILED) + 并发锁

节点函数签名：fn(ctx) -> branch_value（用于条件分支路由；无分支返回 None 即可）。

向后兼容旧 API：Workflow(name, start=...)、add_node(name, fn, next=...)、branch(...)、run(...)。
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常（快速失败）。"""


class SagaFailed(Exception):
    """工作流因节点失败而中断，需要触发 Saga 补偿。"""


class IdempotencyStore:
    """幂等存储：业务键 -> 状态机(PENDING/RUNNING/SUCCESS/FAILED)。

    用线程锁保证「检查-占位」原子性，杜绝并发下的重复执行。
    """

    PENDING, RUNNING, SUCCESS, FAILED = "PENDING", "RUNNING", "SUCCESS", "FAILED"

    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def try_acquire(self, key):
        """原子占位。返回 (cached_result, should_execute)。

        - 首次进入：置 RUNNING，返回 (None, True)  —— 调用方应执行
        - 已 SUCCESS：返回 (result, False)          —— 命中缓存，直接返回
        - 执行中 RUNNING：返回 (None, False)         —— 并发冲突，调用方应等待/拒绝
        - 已 FAILED：允许重试，重置 RUNNING，返回 (None, True)
        """
        with self._lock:
            rec = self._store.get(key)
            if rec is None:
                self._store[key] = {"status": self.RUNNING, "result": None}
                return None, True
            if rec["status"] == self.SUCCESS:
                return rec["result"], False
            if rec["status"] == self.RUNNING:
                return None, False
            # FAILED：允许重试
            self._store[key] = {"status": self.RUNNING, "result": None}
            return None, True

    def commit(self, key, result):
        with self._lock:
            self._store[key] = {"status": self.SUCCESS, "result": result}

    def mark_failed(self, key):
        with self._lock:
            self._store[key] = {"status": self.FAILED, "result": None}

    def get(self, key):
        rec = self._store.get(key)
        return rec["result"] if rec and rec["status"] == self.SUCCESS else None


class CircuitBreaker:
    """熔断器：连续失败达阈值打开，冷却后进半开试探，成功则恢复闭合。

    参数：
    - failure_threshold：连续失败多少次后打开（默认 3）
    - recovery_timeout：打开后冷却多久进入半开（秒，默认 5）
    - half_open_probe：半开时允许多少个试探请求（默认 1）
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold=3, recovery_timeout=5.0, half_open_probe=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_probe = half_open_probe
        self.failures = 0
        self.state = self.CLOSED
        self.last_failure_time = 0.0
        self._probe_count = 0

    def call(self, fn, *args, **kwargs):
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.HALF_OPEN
                self._probe_count = 0
            else:
                raise CircuitOpenError("熔断器打开，快速失败拒绝请求")
        if self.state == self.HALF_OPEN and self._probe_count >= self.half_open_probe:
            raise CircuitOpenError("熔断器半开，试探额度已用尽")
        if self.state == self.HALF_OPEN:
            self._probe_count += 1
        try:
            result = fn(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failures = 0
            return result
        except Exception:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = self.OPEN
            raise


class Workflow:
    """DAG 工作流：拓扑排序 + 并行执行 + 条件分支 + 重试退避超时 + Saga 补偿 + 幂等。"""

    def __init__(self, name, start=None, idempotency=None, max_workers=8):
        self.name = name
        self.start = start
        self.nodes = {}       # name -> node(dict)
        self.branches = {}    # name -> {branch_value: target}
        self.idempotency = idempotency or IdempotencyStore()
        self.max_workers = max_workers

    # ---------- 声明式构建 ----------
    def add_node(self, name, fn, compensate=None, fallback=None, next=None,
                 depends_on=None, retries=0, backoff=1.0, timeout=None):
        """注册节点。

        - depends_on：静态前驱列表，全部完成后本节点才执行（并行/汇聚的关键）
        - next：无条件后继（单个，线性链）
        - retries / backoff / timeout：重试次数 / 退避基数秒 / 超时秒
        - fallback：失败降级函数；compensate：Saga 反向补偿函数
        """
        self.nodes[name] = {
            "fn": fn, "compensate": compensate, "fallback": fallback,
            "next": next, "depends_on": depends_on or [],
            "retries": retries, "backoff": backoff, "timeout": timeout,
        }
        return self

    def branch(self, name, mapping):
        """条件分支：节点返回值作为路由键，决定下一个节点。"""
        self.branches[name] = mapping
        return self

    # ---------- 图构建 ----------
    def _build_graph(self):
        preds = {n: set() for n in self.nodes}
        succs = {n: set() for n in self.nodes}
        for n, node in self.nodes.items():
            for d in node["depends_on"]:
                preds[n].add(d)
                succs[d].add(n)
            if node.get("next"):
                preds[node["next"]].add(n)
                succs[n].add(node["next"])
        for src, mapping in self.branches.items():
            for t in mapping.values():
                preds[t].add(src)
            # 分支目标不加入 succs：运行时由 branch 路由决定激活哪个
        return preds, succs

    # ---------- 节点执行（重试/退避/超时） ----------
    def _execute_node(self, name, ctx):
        node = self.nodes[name]
        attempts = node["retries"] + 1
        last_err = None
        for attempt in range(attempts):
            try:
                if node.get("timeout"):
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(node["fn"], ctx)
                        try:
                            return fut.result(timeout=node["timeout"])
                        except TimeoutError:
                            raise TimeoutError(
                                f"节点执行超过 {node['timeout']}s 超时") from None
                return node["fn"](ctx)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    delay = node["backoff"] * (2 ** attempt)
                    print(f"    ↻ [{name}] 第{attempt + 1}次失败({e})，{delay:.2f}s 后重试")
                    time.sleep(delay)
        raise last_err

    # ---------- 路由与跳过 ----------
    def _maybe_ready(self, n, preds, done, skipped, scheduled, ready):
        if not preds[n] and n not in done and n not in skipped and n not in scheduled:
            ready.append(n)

    def _skip(self, n, preds, succs, orig_preds, done, skipped, scheduled, ready):
        """跳过节点并向后传播：被分支排除的节点不执行，其纯 skip 后继一并跳过。"""
        if n in skipped or n in done:
            return
        skipped.add(n)
        for s in succs.get(n, []):
            preds[s].discard(n)
            if s in done or s in skipped or s in scheduled:
                continue
            if not preds[s]:
                if all(p in skipped for p in orig_preds[s]):
                    self._skip(s, preds, succs, orig_preds, done, skipped, scheduled, ready)
                else:
                    ready.append(s)

    def _release(self, n, route, preds, succs, orig_preds, done, skipped, scheduled, ready):
        """节点完成后释放后继（含分支路由与 skip）。"""
        if n in self.branches:
            selected = self.branches[n].get(route)
            for t in self.branches[n].values():
                if t == selected:
                    preds[t].discard(n)
                    self._maybe_ready(t, preds, done, skipped, scheduled, ready)
                else:
                    self._skip(t, preds, succs, orig_preds, done, skipped, scheduled, ready)
        else:
            for s in succs.get(n, []):
                preds[s].discard(n)
                self._maybe_ready(s, preds, done, skipped, scheduled, ready)

    # ---------- 主执行 ----------
    def run(self, ctx, idempotency_key=None):
        # 1) 幂等占位
        if idempotency_key:
            cached, should_run = self.idempotency.try_acquire(idempotency_key)
            if not should_run:
                if cached is not None:
                    print(f"    [幂等] 命中缓存，跳过执行（key={idempotency_key}）")
                    return cached
                time.sleep(0.3)  # 并发中：短暂等待后重查
                cached = self.idempotency.get(idempotency_key)
                if cached is not None:
                    return cached
                raise RuntimeError(f"幂等键 {idempotency_key} 正在执行中（并发冲突）")

        # 2) 图构建
        preds, succs = self._build_graph()
        orig_preds = {n: set(preds[n]) for n in preds}
        done, skipped, scheduled = set(), set(), set()
        executed = []  # 成功执行顺序（用于逆序补偿）
        ready = [n for n in self.nodes if not preds[n]]

        try:
            # 3) 主循环：并行执行就绪节点
            while ready or (len(done) + len(skipped)) < len(self.nodes):
                if not ready:
                    remaining = [n for n in self.nodes if n not in done and n not in skipped]
                    raise RuntimeError(f"工作流存在环或无法继续，未完成节点: {remaining}")
                batch = ready[:]
                ready = []
                scheduled.update(batch)

                results, errors = {}, {}
                if len(batch) == 1:
                    # 单节点：串行执行，保证输出顺序稳定
                    n = batch[0]
                    try:
                        results[n] = self._execute_node(n, ctx)
                    except Exception as e:
                        errors[n] = e
                else:
                    with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as ex:
                        futs = {ex.submit(self._execute_node, n, ctx): n for n in batch}
                        for f in as_completed(futs):
                            n = futs[f]
                            try:
                                results[n] = f.result()
                            except Exception as e:
                                errors[n] = e

                # 4) 处理结果 + 失败兜底
                for n in batch:
                    if n in errors:
                        self._handle_failure(n, errors[n], ctx, preds, succs, orig_preds,
                                             done, skipped, scheduled, ready, executed)
                    else:
                        print(f"    → [{n}] 完成")
                        done.add(n)
                        executed.append(n)
                        self._release(n, results.get(n), preds, succs, orig_preds,
                                      done, skipped, scheduled, ready)
        except SagaFailed as sf:
            self._compensate(executed, ctx)
            if idempotency_key:
                self.idempotency.mark_failed(idempotency_key)
            raise sf.__cause__ if sf.__cause__ else sf

        # 5) 幂等提交
        if idempotency_key:
            self.idempotency.commit(idempotency_key, ctx)
        return ctx

    def _handle_failure(self, n, err, ctx, preds, succs, orig_preds,
                        done, skipped, scheduled, ready, executed):
        node = self.nodes[n]
        if node.get("fallback"):
            print(f"    ↻ [{n}] 失败({err})，执行兜底 fallback")
            node["fallback"](ctx)
            done.add(n)
            executed.append(n)
            self._release(n, None, preds, succs, orig_preds, done, skipped, scheduled, ready)
        else:
            raise SagaFailed(f"节点 {n} 失败: {err}") from err

    def _compensate(self, executed, ctx):
        for n in reversed(executed):
            comp = self.nodes[n].get("compensate")
            if comp:
                print(f"    ⟲ 补偿 [{n}]")
                comp(ctx)
