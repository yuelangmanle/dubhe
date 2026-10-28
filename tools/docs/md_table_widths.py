#!/usr/bin/env python3
"""为 pandoc 的 docx 输出计算合理的表格列宽。

问题：Markdown 管道表的列宽由分隔行（`| --- | --- |`）中短横线的数量决定。
若全部写成 `---`，pandoc 会把所有列设为等宽，长文本列会被压成极窄的一条。

做法：扫描表格块，按每列内容的显示宽度（CJK 记 2）重写分隔行，
使短横线数量与内容宽度成比例。仅影响排版，不改变任何正文内容。

用法：python3 md_table_widths.py <in.md> <out.md>
"""

import re
import sys

DELIM_RE = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
TOTAL_DASHES = 72      # 分隔行总短横线数（相对量，pandoc 只取比例）
MIN_DASHES = 6
MAX_RATIO = 5.0        # 单列最宽不超过平均列宽的 5 倍
MIN_WIDTH = 14         # 单列最小显示宽度（约 7 个汉字），防止表头被压断行


def display_width(text: str) -> int:
    w = 0
    for ch in text:
        w += 2 if ord(ch) > 0x2E7F else 1
    return w


def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # 链接
    text = re.sub(r"[`*_~]", "", text)                      # 强调/代码
    return text.strip()


def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # 不处理单元格内的转义竖线，文档中不存在该用法
    return [c.strip() for c in line.split("|")]


def rewrite_table(rows, delim_idx):
    header = split_row(rows[0])
    ncol = len(header)
    if ncol == 0:
        return

    widths = [0] * ncol
    for line in rows:
        cells = split_row(line)
        if len(cells) != ncol:
            continue
        for i, cell in enumerate(cells):
            # 取最长的一行（单元格内可能含 <br>）
            longest = max((display_width(strip_inline(p)) for p in cell.split("<br>")), default=0)
            widths[i] = max(widths[i], longest)

    widths = [max(w, MIN_WIDTH) for w in widths]
    avg = sum(widths) / ncol
    widths = [min(w, avg * MAX_RATIO) for w in widths]
    total = sum(widths)

    dashes = [max(MIN_DASHES, round(w / total * TOTAL_DASHES)) for w in widths]
    rows[delim_idx] = "| " + " | ".join("-" * d for d in dashes) + " |"


def main(src: str, dst: str) -> None:
    lines = open(src, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines) - 1:
        if "|" in lines[i] and DELIM_RE.match(lines[i + 1]):
            start = i
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            block = lines[start:j]
            rewrite_table(block, 1)
            lines[start:j] = block
            i = j
        else:
            i += 1
    open(dst, "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
