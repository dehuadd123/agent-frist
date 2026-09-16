"""中英混合分词（零依赖）：英文/数字按词，中文按字符 unigram + bigram。"""
import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list:
    """返回 token 列表。中文采用 unigram + bigram 覆盖，兼顾单字与相邻词组。"""
    if not text:
        return []
    text = text.lower()
    tokens = []
    # 英文/数字连续串作为一个词
    for m in re.finditer(r"[a-z0-9]+", text):
        tokens.append(m.group())
    # 中文字符序列：unigram + bigram
    zh = _CJK_RE.findall(text)
    tokens.extend(zh)
    for i in range(len(zh) - 1):
        tokens.append(zh[i] + zh[i + 1])
    return tokens
