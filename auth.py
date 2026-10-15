"""企业级安全模块：JWT 登录 + RBAC 权限控制 + API 限流（纯标准库，零依赖）。

组件：
  1. JWT（HS256）签发/校验      —— hmac + hashlib + base64，无第三方依赖
  2. 密码哈希（PBKDF2-SHA256）  —— 加盐 120000 次迭代，抵御彩虹表/暴力破解
  3. 用户存储与 RBAC 三级角色    —— viewer(只读) / operator(运营) / admin(管理员)
  4. 滑动窗口限流器             —— 按 IP + 接口分级，防刷/防爆破/控成本

安全要点：
  - JWT 密钥与过期时间可由环境变量 JWT_SECRET / JWT_EXPIRE 覆盖
  - 密码永不落明文，只存 PBKDF2 哈希
  - 用户持久化到 data/users.json（首次启动自动写入默认三账号）
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(_HERE, "data", "users.json")

# ============================================================
# 1. JWT（HS256）
# ============================================================
JWT_SECRET = os.environ.get("JWT_SECRET", "llm-platform-dev-secret-change-me-in-prod")
JWT_EXPIRE = int(os.environ.get("JWT_EXPIRE", "7200"))  # 默认 2 小时


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(data: str) -> str:
    digest = hmac.new(JWT_SECRET.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(digest)


def issue_token(payload: dict, expires: int = JWT_EXPIRE) -> str:
    """签发 JWT：header.payload.signature（HS256）。"""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    body = dict(payload)
    body.update({"iat": now, "exp": now + expires})
    h = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    b = _b64url(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return h + "." + b + "." + _sign(h + "." + b)


def verify_token(token: str) -> dict or None:
    """校验 JWT：签名 + 过期时间，通过返回 payload，否则 None。"""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, b, sig = parts
        if not hmac.compare_digest(_sign(h + "." + b), sig):
            return None
        payload = json.loads(_b64url_decode(b))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ============================================================
# 2. 密码哈希（PBKDF2-SHA256）
# ============================================================
_PBKDF_ITER = 120000


def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _PBKDF_ITER)
    return "pbkdf2$%d$%s$%s" % (_PBKDF_ITER, salt, dk.hex())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, hexd = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), hexd)
    except Exception:
        return False


# ============================================================
# 3. 用户存储 + RBAC
# ============================================================
ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
ROLE_NAMES = {"viewer": "只读访客", "operator": "运营人员", "admin": "管理员"}

_DEFAULT_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin", "name": "系统管理员"},
    {"username": "operator", "password": "operator123", "role": "operator", "name": "运营人员"},
    {"username": "viewer", "password": "viewer123", "role": "viewer", "name": "只读访客"},
]

_users_lock = threading.Lock()


def _save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _ensure_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, encoding="utf-8") as f:
                users = json.load(f)
            if users:
                return users
        except Exception:
            pass
    users = []
    for u in _DEFAULT_USERS:
        users.append({
            "username": u["username"], "name": u["name"], "role": u["role"],
            "pass_hash": hash_password(u["password"]),
        })
    _save_users(users)
    return users


USERS = _ensure_users()


def authenticate(username: str, password: str):
    """登录校验：返回用户 dict（不含哈希）或 None。"""
    with _users_lock:
        u = next((x for x in USERS if x["username"] == username), None)
    if not u or not verify_password(password, u["pass_hash"]):
        return None
    return {"username": u["username"], "name": u["name"], "role": u["role"]}


def get_user(username: str):
    with _users_lock:
        u = next((x for x in USERS if x["username"] == username), None)
    return {"username": u["username"], "name": u["name"], "role": u["role"]} if u else None


def list_users():
    with _users_lock:
        return [{"username": u["username"], "name": u["name"], "role": u["role"]} for u in USERS]


def create_user(username: str, password: str, role: str, name: str = ""):
    if not username or len(username) < 2:
        return False, "用户名至少 2 个字符"
    if len(password) < 6:
        return False, "密码至少 6 位"
    if role not in ROLE_LEVEL:
        return False, "非法角色（仅 viewer/operator/admin）"
    with _users_lock:
        if any(u["username"] == username for u in USERS):
            return False, "用户名已存在"
        USERS.append({"username": username, "name": name or username, "role": role,
                      "pass_hash": hash_password(password)})
        _save_users(USERS)
    return True, "ok"


# ============================================================
# 4. 滑动窗口限流
# ============================================================
class RateLimiter:
    """内存滑动窗口限流器：key -> 时间戳列表，窗口 60s。"""

    def __init__(self):
        self._windows = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: int = 60) -> tuple:
        now = time.time()
        with self._lock:
            ts = self._windows.get(key)
            if ts is None:
                ts = self._windows[key] = []
            while ts and now - ts[0] >= window:
                ts.pop(0)
            if len(ts) >= limit:
                return False, len(ts)
            ts.append(now)
            return True, len(ts)


# 接口分级限流：req/min/IP
RATE_RULES = {
    "auth": 5,       # 登录接口，防暴力破解
    "write": 10,     # 写操作（改订单/上传/执行 Agent），防误操作刷量
    "llm": 20,       # LLM 接口，控成本
    "default": 100,  # 普通读接口
}
LIMITER = RateLimiter()


def rate_key(path: str) -> str:
    if path == "/api/auth/login":
        return "auth"
    if path.startswith("/api/llm/"):
        return "llm"
    if (path.startswith("/api/order/update") or path.startswith("/api/rag/upload")
            or path.startswith("/api/rag/import_sample") or path.startswith("/api/agent/run")):
        return "write"
    return "default"


def check_rate(ip: str, path: str):
    """返回 (允许?, 每分钟上限)。"""
    key = rate_key(path)
    limit = RATE_RULES[key]
    ok, _cnt = LIMITER.allow("%s:%s" % (ip, key), limit)
    return ok, limit


# ============================================================
# 5. RBAC 路由权限映射
# ============================================================
# 返回该路径所需最低角色等级：0=公开 1=viewer 2=operator 3=admin
_WRITE_PREFIXES = (
    "/api/order/update", "/api/rag/upload", "/api/rag/import_sample",
    "/api/agent/run", "/api/llm/", "/api/ml/tuning", "/api/ml/abtest",
    "/api/ml/offline_eval",
)


def route_level(method: str, path: str) -> int:
    # 前端页面与健康检查、登录接口公开
    if path in ("/", "/index.html", "/console", "/enterprise"):
        return 0
    if path in ("/api/health", "/api/auth/login"):
        return 0
    if path == "/api/auth/me":
        return 1  # 任何已登录用户可查自己
    if path.startswith("/api/auth/"):
        return 3  # 用户管理仅 admin
    if any(path.startswith(w) for w in _WRITE_PREFIXES):
        return 2
    if path == "/api/audit":
        return 2  # 审计日志仅运营及以上
    return 1  # 其余只读接口，viewer 即可
