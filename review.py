"""人工审核（HITL，Human-in-the-Loop）审核队列（零依赖）。

企业级关键动作不能全自动：退款大额/风控、金融合规拦截、写操作复核等，需要人工确认后才放行。
本模块提供一套「待审队列 → 通过/驳回 → 落盘留痕」的完整闭环：

- create()：产生一条待审任务（pending），自动落盘
- list()/get()：查看待审/已审
- approve()/reject()：人工裁决，记录裁决人与备注，触发 on_verdict 回调（可接后续放行动作）
- 持久化：data/review_tasks.json，重启后队列不丢
"""
import json
import os
import threading
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_REVIEW_FILE = os.path.join(_HERE, "data", "review_tasks.json")

STATUS_CN = {"pending": "待审核", "approved": "已通过", "rejected": "已驳回"}


class ReviewQueue:
    def __init__(self, path=None):
        self._path = path or _REVIEW_FILE
        self._tasks = {}          # review_id -> task dict
        self._lock = threading.Lock()
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for t in json.load(f):
                    self._tasks[t["review_id"]] = t
        except Exception:
            pass

    def _save(self):
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(list(self._tasks.values()), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass

    # ---------- 业务 ----------
    def create(self, kind, title, payload, actor="system", on_verdict=None):
        """创建一条待审任务。payload 存放待审上下文（订单/合规详情等）。"""
        review_id = "RV-" + uuid.uuid4().hex[:10]
        task = {
            "review_id": review_id, "kind": kind, "title": title,
            "payload": payload, "actor": actor, "status": "pending",
            "verdict_by": "", "comment": "", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": "",
        }
        with self._lock:
            self._tasks[review_id] = task
            self._save()
        if on_verdict:
            task["_on_verdict"] = on_verdict  # 回调不落盘
        return task

    def list(self, status=None):
        with self._lock:
            items = list(self._tasks.values())
        if status:
            items = [t for t in items if t["status"] == status]
        items.sort(key=lambda t: t["created_at"], reverse=True)
        return [{k: v for k, v in t.items() if not k.startswith("_")} for t in items]

    def get(self, review_id):
        with self._lock:
            t = self._tasks.get(review_id)
            if not t:
                return None
            return {k: v for k, v in t.items() if not k.startswith("_")}

    def verdict(self, review_id, action, actor, comment=""):
        """裁决：action in {approve, reject}。返回更新后的任务或 None。"""
        with self._lock:
            t = self._tasks.get(review_id)
            if not t:
                return None
            if t["status"] != "pending":
                return {k: v for k, v in t.items() if not k.startswith("_")}
            t["status"] = "approved" if action == "approve" else "rejected"
            t["verdict_by"] = actor
            t["comment"] = comment
            t["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            cb = t.pop("_on_verdict", None)
            self._save()
        if cb:
            try:
                cb(t, action)
            except Exception:
                pass
        return {k: v for k, v in t.items() if not k.startswith("_")}

    def stats(self):
        with self._lock:
            items = list(self._tasks.values())
        return {
            "total": len(items),
            "pending": sum(1 for t in items if t["status"] == "pending"),
            "approved": sum(1 for t in items if t["status"] == "approved"),
            "rejected": sum(1 for t in items if t["status"] == "rejected"),
        }


# 全局单例（跨请求保持审核队列状态）
QUEUE = ReviewQueue()
