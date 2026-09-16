"""5 种切片策略（零依赖）。每种策略函数签名：func(text) -> list[str]。"""
import re

_SENT_SPLIT = re.compile(r"(?<=[。！？；])\s*")
_SEC_SPLIT = re.compile(r"\n(?=## )")
_TABLE_LINE = re.compile(r"^\|.*\|$")
_TABLE_SEP = re.compile(r"^\|[\s:|-]+\|$")


def chunk_fixed(text, size=100, overlap=20):
    """1) 固定长度切片：按字符数等长切分，滑动重叠。"""
    chunks = []
    step = max(1, size - overlap)
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


def chunk_recursive(text, size=150):
    """2) 递归切片：段落 -> 句子 -> 字符 逐级切分。"""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text.strip()) if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= size:
            chunks.append(para)
            continue
        sents = [s for s in _SENT_SPLIT.split(para) if s]
        cur = ""
        for s in sents:
            if len(cur) + len(s) <= size:
                cur += s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
                while len(cur) > size:
                    chunks.append(cur[:size])
                    cur = cur[size:]
        if cur:
            chunks.append(cur)
    return chunks


def chunk_semantic(text, threshold=0.25):
    """3) 语义切片：句子间 TF-IDF 余弦相似度低于阈值处切分。"""
    from common.vector import TfidfVectorizer, cosine

    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        return [text]
    vecs = TfidfVectorizer().fit_transform(sents)
    chunks, cur = [], sents[0]
    for i in range(1, len(sents)):
        if cosine(vecs[i - 1], vecs[i]) < threshold:
            chunks.append(cur)
            cur = sents[i]
        else:
            cur += sents[i]
    chunks.append(cur)
    return chunks


def chunk_structure(text):
    """4) 结构感知切片：按 Markdown 标题分节；表格数据行带表头上下文单独成块；段落各成一块。"""
    sections = [s.strip() for s in _SEC_SPLIT.split(text.strip()) if s.strip()]
    if not sections:
        sections = [text.strip()]
    chunks = []
    for sec in sections:
        lines = sec.split("\n")
        title = ""
        if lines and lines[0].startswith("#"):
            title = lines[0].strip("# ").strip()
            body = lines[1:]
        else:
            body = lines
        prefix = ("[%s] " % title) if title else ""
        rows, para_buf = [], []

        def flush_para():
            if para_buf:
                chunks.append(prefix + " ".join(para_buf))
                para_buf.clear()

        def emit_table():
            if not rows:
                return
            header = rows[0]
            data_rows = [r for r in rows[1:] if not _TABLE_SEP.match(r)]
            if not data_rows:
                chunks.append(prefix + header)
            else:
                for r in data_rows:
                    chunks.append(prefix + header + " " + r)

        for ln in body:
            if _TABLE_LINE.match(ln.strip()):
                rows.append(ln.strip())
            else:
                if rows:
                    emit_table()
                    rows = []
                if ln.strip():
                    para_buf.append(ln.strip())
                else:
                    flush_para()
        emit_table()
        flush_para()
    return chunks


def chunk_parent_child(text, child_size=60):
    """5) 父子切片：返回子块列表（检索用）；父块由相邻子块聚合（供生成上下文）。"""
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    children, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= child_size:
            cur += s
        else:
            if cur:
                children.append(cur)
            cur = s
            while len(cur) > child_size:
                children.append(cur[:child_size])
                cur = cur[child_size:]
    if cur:
        children.append(cur)
    return children


# 策略注册表：name -> (func, 说明)
STRATEGIES = {
    "fixed": (chunk_fixed, "固定长度切片(size=100,overlap=20)"),
    "recursive": (chunk_recursive, "递归切片(size=150)"),
    "semantic": (chunk_semantic, "语义切片(相似度阈值0.25)"),
    "structure": (chunk_structure, "结构感知切片(标题/表格)"),
    "parent_child": (chunk_parent_child, "父子切片(子块60)"),
}
