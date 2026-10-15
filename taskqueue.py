"""异步任务队列（零依赖，标准库 threading + queue 实现）。

企业级场景里，文档索引、Agent 执行、批量评测等重操作应异步化：
- submit(fn, *args, **kwargs) 立即返回 task_id，不阻塞 HTTP 请求
- 后台 worker 线程顺序消费任务，状态机 pending -> running -> done/failed
- 结果与异常留存在内存（可选落盘），供前端轮询 GET /api/tasks/{id}
- 支持取消（对尚未开始的任务）、按状态过滤列表

线程安全：dict 操作加锁；worker 用 daemon 线程随进程退出。
"""
import json
import os
import queue
import threading
import time
import traceback
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK_FILE = os.path.join(_HERE, "data", "tasks.json")

STATUS_CN = {"pending": "排队中", "running": "执行中", "done": "已完成", "failed": "失败"}


class TaskQueue:
    def __init__(self, workers=2, path=None):
        self._path = path or _TASK_FILE
        self._q = queue.Queue()
        self._tasks = {}          # task_id -> task dict
        self._lock = threading.Lock()
        self._workers = []
        for _ in range(max(1, workers)):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    # ---------- 状态管理 ----------
    def _set(self, task_id, **fields):
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.update(fields)

    def _snapshot(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            return dict(t) if t else None

    def _save(self):
        try:
            with self._lock:
                data = [{k: v for k, v in t.items() if not callable(v)}
                        for t in self._tasks.values()]
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._path)
        except (OSError, TypeError):
            pass

    # ---------- 提交 ----------
    def submit(self, name, fn, *args, **kwargs):
        task_id = "TSK-" + uuid.uuid4().hex[:8]
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id, "name": name, "status": "pending",
                "created_at": time.time(), "started_at": None, "finished_at": None,
                "result": None, "error": None, "_fn": fn, "_args": args, "_kwargs": kwargs,
            }
        self._q.put(task_id)
        return self._snapshot(task_id)

    # ---------- worker ----------
    def _worker(self):
        while True:
            task_id = self._q.get()
            try:
                t = self._snapshot(task_id)
                if not t:
                    continue
                self._set(task_id, status="running", started_at=time.time())
                fn, args, kwargs = t["_fn"], t["_args"], t["_kwargs"]
                result = fn(*args, **kwargs)
                self._set(task_id, status="done", result=result,
                          finished_at=time.time())
            except Exception as e:
                self._set(task_id, status="failed", error=str(e),
                          traceback=traceback.format_exc(limit=5),
                          finished_at=time.time())
            finally:
                self._save()
                self._q.task_done()

    # ---------- 查询 ----------
    def get(self, task_id):
        t = self._snapshot(task_id)
        if not t:
            return None
        out = {k: v for k, v in t.items() if not k.startswith("_")}
        out["status_cn"] = STATUS_CN.get(out.get("status"), out.get("status"))
        return out

    def list(self, status=None):
        with self._lock:
            items = [dict(t) for t in self._tasks.values()]
        if status:
            items = [t for t in items if t["status"] == status]
        items.sort(key=lambda t: t["created_at"], reverse=True)
        return [{"task_id": t["task_id"], "name": t["name"], "status": t["status"],
                 "status_cn": STATUS_CN.get(t["status"], t["status"]),
                 "created_at": t["created_at"], "started_at": t.get("started_at"),
                 "finished_at": t.get("finished_at"), "error": t.get("error")}
                for t in items]

    def cancel(self, task_id):
        """取消尚未开始的任务。"""
        t = self._snapshot(task_id)
        if not t:
            return None
        if t["status"] == "pending":
            self._set(task_id, status="failed", error="已取消")
            return self.get(task_id)
        return self.get(task_id)

    def stats(self):
        with self._lock:
            items = list(self._tasks.values())
        return {
            "total": len(items),
            "pending": sum(1 for t in items if t["status"] == "pending"),
            "running": sum(1 for t in items if t["status"] == "running"),
            "done": sum(1 for t in items if t["status"] == "done"),
            "failed": sum(1 for t in items if t["status"] == "failed"),
        }


# 全局单例（后台 worker 随服务启动常驻）
QUEUE = TaskQueue(workers=2)
