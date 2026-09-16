"""多格式文档解析（纯标准库，零依赖）。

覆盖三类业务文档的自动解析：
  电商导购  -> 图文混排（Markdown/HTML 含图片占位，图片降级为多模态占位）
  金融投研  -> PDF 表格（FlateDecode 文本流提取 + 表格还原）
  企业办公  -> 多格式（docx / xlsx / pptx / csv / json 等 OOXML 与文本格式）

核心函数：
  SUPPORTED_EXTS  -> {扩展名: 格式说明}
  detect_format(filename) -> 返回 (ext, format_name)
  parse_bytes(filename, data: bytes) -> {ok, ext, format, text, note, tables?}
  parse_text(filename, text: str, ext=None) -> {ok, ext, format, text, note}

设计说明：真实生产环境图片走 OCR/多模态，PDF 扫描件走 OCR；此处在线沙箱
零依赖，图片类降级为占位标记，文本型 PDF 用 zlib 解压文本流提取。
"""
import io
import re
import csv
import json
import zipfile
import zlib
from html.parser import HTMLParser

# 扩展名 -> 格式说明
SUPPORTED_EXTS = {
    ".txt": "纯文本",
    ".md": "Markdown（图文混排/表格）",
    ".markdown": "Markdown（图文混排/表格）",
    ".csv": "CSV 表格",
    ".tsv": "TSV 表格",
    ".json": "JSON 结构化数据",
    ".html": "HTML 网页",
    ".htm": "HTML 网页",
    ".docx": "Word 文档（OOXML）",
    ".xlsx": "Excel 表格（OOXML）",
    ".pptx": "PowerPoint 演示（OOXML）",
    ".pdf": "PDF（文本流提取）",
    ".png": "图片（需多模态/OCR）",
    ".jpg": "图片（需多模态/OCR）",
    ".jpeg": "图片（需多模态/OCR）",
}


def detect_format(filename):
    """根据文件名后缀返回 (ext, format_name)。"""
    name = (filename or "").lower()
    dot = name.rfind(".")
    ext = name[dot:] if dot >= 0 else ""
    fmt = SUPPORTED_EXTS.get(ext, "未知格式（按纯文本尝试）")
    return ext, fmt


# ---------------------------------------------------------------- 文本类 ----
def _extract_txt(data):
    return data.decode("utf-8", errors="replace")


def _extract_csv(data, delim=","):
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return _rows_to_markdown(rows)


def _extract_json(data):
    obj = json.loads(data.decode("utf-8", errors="replace"))
    parts = []

    def walk(o, depth=0):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)
                else:
                    parts.append("%s%s: %s" % ("  " * depth, k, v))
        elif isinstance(o, list):
            for v in o:
                walk(v, depth)
        else:
            parts.append(str(o))

    walk(obj)
    return "\n".join(parts)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table"):
            self.parts.append("\n")
        if tag == "img":
            for k, v in attrs:
                if k == "alt" and v:
                    self.parts.append("[图片: %s]" % v)
                elif k == "src" and v:
                    self.parts.append("[图片: %s]" % v.rsplit("/", 1)[-1])

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        if tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def _extract_html(data):
    p = _TextExtractor()
    p.feed(data.decode("utf-8", errors="replace"))
    text = "".join(p.parts)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# -------------------------------------------------------------- OOXML 类 ----
_XML_NS = re.compile(r"</?[a-zA-Z0-9]+:")


def _strip_ns(xml):
    """去掉 OOXML 命名空间前缀，便于正则匹配（保留闭合斜杠）。"""
    return _XML_NS.sub(lambda m: ("</" if m.group(0).startswith("</") else "<"), xml)


def _extract_docx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = _strip_ns(z.read("word/document.xml").decode("utf-8", "replace"))
    paras = re.findall(r"<p[ >].*?</p>", xml, re.S)
    lines = []
    for p in paras:
        texts = re.findall(r"<t[^>]*>(.*?)</t>", p, re.S)
        line = "".join(_unescape(t) for t in texts).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _extract_xlsx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            sxml = _strip_ns(z.read("xl/sharedStrings.xml").decode("utf-8", "replace"))
            shared = [_unescape(t) for t in re.findall(r"<t[^>]*>(.*?)</t>", sxml, re.S)]
        sheets = sorted(
            [n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)],
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        out_tables = []
        for sheet in sheets:
            xml = _strip_ns(z.read(sheet).decode("utf-8", "replace"))
            rows = re.findall(r"<row[ >].*?</row>", xml, re.S)
            table = []
            for r in rows:
                cells = re.findall(r"<c[ >].*?(?:/>|</c>)", r, re.S)
                vals = []
                for c in cells:
                    t = re.search(r't="(\w+)"', c)
                    v = re.search(r"<v>(.*?)</v>", c)
                    if t and t.group(1) == "s" and v:
                        idx = int(float(v.group(1)))
                        vals.append(shared[idx] if idx < len(shared) else "")
                    elif v:
                        vals.append(_unescape(v.group(1)))
                    else:
                        vals.append("")
                if any(x.strip() for x in vals):
                    table.append(vals)
            if table:
                out_tables.append(_rows_to_markdown(table))
    return "\n\n".join(out_tables)


def _extract_pptx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        slides = sorted(
            [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        parts = []
        for i, s in enumerate(slides, 1):
            xml = _strip_ns(z.read(s).decode("utf-8", "replace"))
            texts = [_unescape(t).strip() for t in re.findall(r"<t[^>]*>(.*?)</t>", xml, re.S)]
            texts = [t for t in texts if t]
            if texts:
                parts.append("## 第 %d 页\n%s" % (i, "\n".join(texts)))
    return "\n\n".join(parts)


def _unescape(s):
    return (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&apos;", "'"))


def _rows_to_markdown(rows):
    """二维列表 -> markdown 表格。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [list(r) + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in norm[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------- PDF 类 ----
def _decompress_stream(raw):
    """解压 PDF 流：去掉 stream 与 endstream 之间的单个 EOL 后尝试 zlib 解压。"""
    candidates = [raw]
    for cut in (b"\n", b"\r\n", b"\r"):
        if raw.endswith(cut):
            candidates.append(raw[: -len(cut)])
    for cand in candidates:
        try:
            return zlib.decompress(cand)
        except Exception:
            continue
    # 未压缩（明文）流
    return raw.rstrip(b"\r\n")


def _extract_pdf(data):
    """从 PDF 提取文本流：解 FlateDecode，抓 (..)Tj 与 [..]TJ 文本。"""
    texts = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        raw = m.group(1)
        dec = _decompress_stream(raw)
        # 单段文本 (xxx) Tj
        for tm in re.finditer(rb"\((?:[^()\\]|\\.)*\)\s*Tj", dec):
            s = tm.group(0)
            inner = re.sub(rb"^\(|\)\s*Tj$", b"", s)
            texts.append(_pdf_unescape(inner))
        # 多段文本 [ (a) (b) ] TJ
        for tm in re.finditer(rb"\[(?:[^\[\]\\]|\\.)*\]\s*TJ", dec):
            parts = re.findall(rb"\((?:[^()\\]|\\.)*\)", tm.group(0))
            texts.append("".join(_pdf_unescape(re.sub(rb"^\(|\)$", b"", p)) for p in parts))
    return "\n".join(t for t in texts if t.strip())


def _pdf_unescape(b):
    b = b.replace(rb"\(", b"(").replace(rb"\)", b")").replace(rb"\\", b"\\")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")


# -------------------------------------------------------------- 图片类 ----
def _placeholder_image(filename):
    return "[图片: %s]\n（在线沙箱零依赖、无 OCR/多模态；真实环境此处走图文 OCR 或多模态视觉模型提取）" % filename


# ----------------------------------------------------------------- 入口 ----
def parse_bytes(filename, data):
    """解析二进制文件内容，返回统一结构。"""
    ext, fmt = detect_format(filename)
    name = filename or "未命名"
    if ext in (".txt", ".md", ".markdown"):
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_txt(data), "note": ""}
    if ext == ".csv":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_csv(data, ","), "note": ""}
    if ext == ".tsv":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_csv(data, "\t"), "note": ""}
    if ext == ".json":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_json(data), "note": ""}
    if ext in (".html", ".htm"):
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_html(data), "note": ""}
    if ext == ".docx":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_docx(data), "note": ""}
    if ext == ".xlsx":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_xlsx(data), "note": ""}
    if ext == ".pptx":
        return {"ok": True, "ext": ext, "format": fmt, "text": _extract_pptx(data), "note": ""}
    if ext == ".pdf":
        text = _extract_pdf(data)
        if text.strip():
            return {"ok": True, "ext": ext, "format": fmt, "text": text,
                    "note": "已从 PDF 文本流提取内容（图片型/扫描件 PDF 需 OCR）"}
        return {"ok": True, "ext": ext, "format": fmt,
                "text": "[PDF: %s] 该 PDF 未检测到文本层（图片型/扫描件），在线环境无 OCR，已按占位纳入" % name,
                "note": "图片型 PDF 无文本层，真实环境需 OCR 提取"}
    if ext in (".png", ".jpg", ".jpeg"):
        return {"ok": True, "ext": ext, "format": fmt, "text": _placeholder_image(name),
                "note": "图片需多模态/OCR，在线环境已降级为占位标记"}
    # 未知格式：按纯文本兜底
    return {"ok": True, "ext": ext, "format": fmt, "text": _extract_txt(data), "note": "未知格式，按纯文本解析"}


def parse_text(filename, text, ext=None):
    """直接解析文本内容（用于内置模拟文档，标注原始格式）。"""
    ext = ext or detect_format(filename)[0]
    fmt = SUPPORTED_EXTS.get(ext, "文本")
    note = {
        ".md": "Markdown 图文混排（图片占位标记保留）",
        ".pdf": "PDF 表格（模拟文本层提取结果）",
        ".xlsx": "Excel 表格（已解析为 Markdown 表格）",
        ".pptx": "PowerPoint 演示（按页提取）",
        ".docx": "Word 文档（段落文本提取）",
    }.get(ext, "")
    return {"ok": True, "ext": ext, "format": fmt, "text": text, "note": note}


if __name__ == "__main__":
    # 自检：生成一份最小文本型 PDF 与一份 docx，验证解析路径
    def _make_mini_pdf():
        import zlib as _z
        content = b"BT /F1 12 Tf 72 720 Td (\xe8\x90\xa5\xe4\xb8\x9a\xe6\x94\xb6\xe5\x85\xa5 128.6 \xe4\xba\xbf) Tj ET"
        stream = _z.compress(content)
        objs = [
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj",
            b"4 0 obj << /Length %d >> stream\n%s\nendstream endobj" % (len(stream), stream),
        ]
        pdf = b"%PDF-1.4\n"
        offs = []
        for i, o in enumerate(objs, 1):
            offs.append(len(pdf))
            pdf += o + b"\n"
        xref = len(pdf)
        pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
        for off in offs:
            pdf += b"%010d 00000 n \n" % off
        pdf += b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
        return pdf

    print("== 自检：文本型 PDF 表格解析 ==")
    r = parse_bytes("季报.pdf", _make_mini_pdf())
    print("  ok=%s ext=%s note=%r" % (r["ok"], r["ext"], r["note"]))
    print("  text=", r["text"][:60])

    print("== 自检：docx 解析 ==")
    import zipfile as _z
    buf = io.BytesIO()
    with _z.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   '<?xml version="1.0"?><w:document xmlns:w="x"><w:body>'
                   '<w:p><w:r><w:t>报销标准：一线城市 500 元/晚</w:t></w:r></w:p></w:body></w:document>')
    r2 = parse_bytes("制度.docx", buf.getvalue())
    print("  ok=%s ext=%s text=%r" % (r2["ok"], r2["ext"], r2["text"][:40]))
