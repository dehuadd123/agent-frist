"""文档级 ACL 权限隔离（零依赖）。

企业知识库中，不同文档对不同的角色/用户可见性不同：
- public：所有人可见（内置语料、公开制度）
- owner：仅上传者本人 + admin 可见
- 角色白名单：显式指定可见角色集合

与「路由级 RBAC」的区别：RBAC 控制「能不能调这个接口」，ACL 控制「能不能看到这篇文档的切片」。
两者叠加：接口权限足够 + 文档权限足够，才返回该文档的检索结果。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

# 角色等级（与 auth.ROLE_LEVEL 对齐，避免循环依赖，这里独立定义一份）
ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}

# 文档元数据里 ACL 字段的默认约定
PUBLIC_MARK = "public"


def normalize_acl(acl):
    """把用户传的 acl 规整为统一结构：{"public": bool, "owner": str, "roles": [..]}。"""
    acl = acl or {}
    if isinstance(acl, list):
        acl = {"roles": list(acl)}
    public = bool(acl.get("public", False))
    owner = acl.get("owner") or ""
    roles = list(acl.get("roles") or [])
    return {"public": public, "owner": owner, "roles": roles}


def make_acl(owner="", public=False, roles=None):
    """构造文档 ACL：上传文档默认 owner 私有（owner/admin 可见），可显式 public 或加角色。"""
    return {"public": public, "owner": owner or "", "roles": list(roles or [])}


def can_access(acl, actor="", role="viewer"):
    """判断 (actor, role) 是否可访问携带该 ACL 的文档。

    - public -> 任何已登录用户可读
    - owner 命中 -> 可读
    - 角色白名单命中 -> 可读
    - admin 永远可读（超级管理员）
    - 未登录（actor 为空）-> 仅 public
    """
    acl = normalize_acl(acl)
    if acl["public"]:
        return True
    if role == "admin":
        return True
    if actor and acl["owner"] and actor == acl["owner"]:
        return True
    if role in acl["roles"]:
        return True
    return False


def build_doc_filter(actor="", role="viewer"):
    """返回一个 filter_fn(rec) 闭包，供 VectorStore.search 做文档级 ACL 过滤。

    rec["meta"] 需含 acl 字段（由 make_acl 构造后随 chunk 写入）。
    """
    def _filter(rec):
        meta = rec.get("meta") or {}
        acl = meta.get("acl")
        if not acl:
            return True  # 无 ACL 标注的旧数据视为 public（向后兼容）
        return can_access(acl, actor, role)
    return _filter
