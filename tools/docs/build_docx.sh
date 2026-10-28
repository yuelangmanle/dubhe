#!/usr/bin/env bash
# Markdown → docx（排版产物）
#
# 用法：
#   ./tools/docs/build_docx.sh docs/00-总项目开发书.md
#   ./tools/docs/build_docx.sh docs/00-总项目开发书.md out/总开发书.docx
#
# 说明：
#   - Markdown 是唯一真相来源，本脚本只生成「给人看的排版版」。
#   - 不要反向编辑生成的 docx；需要改内容请改 .md 后重新生成。
#   - 依赖：pandoc（brew install pandoc）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="$ROOT/tools/docs/reference.docx"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "错误：未找到 pandoc。请先安装：brew install pandoc" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "用法：$0 <input.md> [output.docx]" >&2
  exit 1
fi

IN="$1"
BASE="$(basename "${IN%.md}")"
OUT="${2:-$ROOT/docs/build/$BASE.docx}"

mkdir -p "$(dirname "$OUT")"

# 首次运行生成参考样式（中文字体友好）
if [[ ! -f "$REF" ]]; then
  echo "生成参考样式：$REF"
  pandoc --print-default-data-file reference.docx > "$REF"
fi

pandoc "$IN" \
  --from=gfm \
  --to=docx \
  --reference-doc="$REF" \
  --toc \
  --toc-depth=2 \
  --resource-path="$ROOT" \
  -o "$OUT"

echo "已生成：$OUT"
