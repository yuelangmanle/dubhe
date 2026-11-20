#!/usr/bin/env python3
"""Extract DOCX text units or rebuild the original package with their translations."""

import base64
import copy
from dataclasses import dataclass
import io
import json
import re
import shutil
import sys
from urllib.parse import unquote
from xml.parsers import expat
from xml.sax.saxutils import escape
import zipfile
import zlib


MAX_DOCUMENT = 24 * 1024 * 1024
MAX_REQUEST = 48 * 1024 * 1024
MAX_UNPACKED = 128 * 1024 * 1024
MAX_XML = 12 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_ELEMENTS = 250_000
MAX_PARAGRAPHS = 50_000

CONTENT_TYPES = "[Content_Types].xml"
TYPE_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/content-types "
WORD_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml." + kind + "+xml": root
    for kind, root in {
        "document.main": "document", "header": "hdr", "footer": "ftr", "footnotes": "footnotes", "endnotes": "endnotes",
    }.items()
}
WORD_NAMESPACES = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
}
BLOCK_CONTAINERS = {
    "body", "tbl", "tr", "tc", "sdt", "sdtContent", "customXml", "ins", "moveTo", "footnote", "endnote",
}
INLINE_CONTAINERS = {"hyperlink", "sdt", "sdtContent", "smartTag", "customXml", "ins", "moveTo", "dir", "bdo"}
FORMATTING = {"pPr", "rPr", "sdtPr", "sdtEndPr", "smartTagPr", "customXmlPr", "proofErr"}
XML_SPACE = "http://www.w3.org/XML/1998/namespace space"
TAG_END = re.compile(rb"""(?:[^>"']|"[^"]*"|'[^']*')*>""")
ATTRIBUTE = re.compile(rb"""\s+([^\s=<>/]+)\s*=\s*(["'])(.*?)\2""", re.DOTALL)
INVALID_XML_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


class DocumentError(ValueError):
    pass


@dataclass
class TextSpan:
    start: int
    space: str
    end: int = 0
    value: str = ""


def reject_doctype(*_):
    raise DocumentError("Word 文档不能包含 DTD。")


class WordPart:
    """Locate translatable text without rewriting its surrounding XML."""

    def __init__(self, raw, root_name):
        self.original = raw
        self.utf16 = None
        if raw.startswith((b"\xff\xfe", b"<\x00")):
            self.utf16 = "utf-16-le"
        elif raw.startswith((b"\xfe\xff", b"\x00<")):
            self.utf16 = "utf-16-be"
        self.raw = raw.decode(self.utf16).encode("utf-8") if self.utf16 else raw
        self.encoding = "utf-8"
        self.root_name = root_name
        self.namespace = ""
        self.path = []
        self.elements = 0
        self.ignored_depth = 0
        self.paragraph_depth = 0
        self.field_depth = 0
        self.text_node = None
        self.text_chunks = []
        self.text_has_markup = False
        self.group = []
        self.paragraphs = []
        self.parser = expat.ParserCreate("UTF-8" if self.utf16 else None, " ")
        self.parser.buffer_text = True
        self.parser.StartElementHandler = self.start
        self.parser.EndElementHandler = self.end
        self.parser.CharacterDataHandler = self.characters
        self.parser.XmlDeclHandler = self.xml_declaration
        self.parser.CommentHandler = self.inline_markup
        self.parser.ProcessingInstructionHandler = self.inline_markup
        self.parser.StartDoctypeDeclHandler = reject_doctype
        self.parser.Parse(self.raw, True)
        self.parser = None

    def xml_declaration(self, version, encoding, standalone):
        if encoding and not self.utf16:
            self.encoding = encoding

    def finish_group(self):
        if self.group:
            value = "".join(span.value for span in self.group)
            if any(character.isalpha() for character in value):
                self.paragraphs.append((value, self.group))
                if len(self.paragraphs) > MAX_PARAGRAPHS:
                    raise DocumentError("Word 文档段落过多。")
            self.group = []

    def start(self, name, attributes):
        depth = len(self.path) + 1
        self.elements += 1
        if self.elements > MAX_ELEMENTS or depth > 256:
            raise DocumentError("Word 文档结构过于复杂。")
        if depth == 1:
            namespace, _, local = name.partition(" ")
            if namespace not in WORD_NAMESPACES or local != self.root_name:
                raise DocumentError("Word 文档部件格式无效。")
            self.namespace = namespace + " "
        local = name[len(self.namespace):] if name.startswith(self.namespace) else ""
        self.path.append(local)
        if self.ignored_depth or depth == 1:
            return
        if self.text_node is not None:
            raise DocumentError("Word 文本节点格式无效。")

        if not self.paragraph_depth:
            if local == "p":
                self.paragraph_depth = depth
            elif local not in BLOCK_CONTAINERS:
                self.ignored_depth = depth
            elif local in {"footnote", "endnote"} and attributes.get(self.namespace + "type", "normal") != "normal":
                self.ignored_depth = depth
            return

        if local in FORMATTING:
            self.ignored_depth = depth
            return
        if local in INLINE_CONTAINERS:
            self.finish_group()
            return
        if local == "r":
            return
        if local == "t" and self.path[-2] == "r" and not self.field_depth:
            self.text_node = TextSpan(self.parser.CurrentByteIndex, attributes.get(XML_SPACE, ""))
            self.text_chunks = []
            self.text_has_markup = False
            return

        self.finish_group()
        if local == "fldChar":
            kind = attributes.get(self.namespace + "fldCharType")
            if kind == "begin":
                self.field_depth += 1
            elif kind == "end" and self.field_depth:
                self.field_depth -= 1
        self.ignored_depth = depth

    def characters(self, value):
        if self.text_node is not None:
            self.text_chunks.append(value)

    def inline_markup(self, *_):
        if self.text_node is not None:
            self.text_has_markup = True

    def end(self, name):
        depth = len(self.path)
        local = self.path.pop()
        if self.ignored_depth:
            if self.ignored_depth == depth:
                self.ignored_depth = 0
            return
        if self.text_node is not None:
            if self.text_has_markup:
                self.finish_group()
            elif self.text_chunks:
                self.text_node.end = self.parser.CurrentByteIndex
                self.text_node.value = "".join(self.text_chunks)
                self.group.append(self.text_node)
            self.text_node = None
            self.text_chunks = []
        if self.paragraph_depth == depth:
            self.finish_group()
            self.paragraph_depth = 0
        elif self.paragraph_depth and local in INLINE_CONTAINERS:
            self.finish_group()

    def replace_span(self, span, value):
        tag_end = TAG_END.match(self.raw, span.start).end()
        opening = self.raw[span.start:tag_end]
        if value and (value[0].isspace() or value[-1].isspace()) and span.space != "preserve":
            if span.space:
                for attribute in ATTRIBUTE.finditer(opening):
                    if attribute[1] == b"xml:space":
                        start, end = attribute.span(3)
                        opening = opening[:start] + b"preserve" + opening[end:]
                        break
            else:
                opening = opening[:-1] + b' xml:space="preserve">'
        return opening + escape(value, {"\r": "&#13;"}).encode(self.encoding, "xmlcharrefreplace")

    def rebuild(self, translations):
        if sum(map(len, translations)) > MAX_XML:
            raise DocumentError("翻译后的 Word 文档正文过大。")
        chunks = []
        cursor = 0
        for (source, spans), translated in zip(self.paragraphs, translations):
            leading = source[:len(source) - len(source.lstrip())]
            trailing = source[len(source.rstrip()):]
            translated = leading + translated.strip() + trailing
            if translated == source:
                continue
            # Keep the original runs and apportion the translation by their lengths.
            # A shorter or reordered translation has no exact character-to-run map.
            offset = 0
            start = 0
            for span in spans:
                offset += len(span.value)
                end = round(offset * len(translated) / len(source))
                replacement = translated[start:end]
                start = end
                if replacement != span.value:
                    chunks.append(self.raw[cursor:span.start])
                    chunks.append(self.replace_span(span, replacement))
                    cursor = span.end
        if not chunks:
            return self.original
        chunks.append(self.raw[cursor:])
        updated = b"".join(chunks)
        if self.utf16:
            updated = updated.decode("utf-8").encode(self.utf16)
        if len(updated) > MAX_XML:
            raise DocumentError("翻译后的 Word 文档正文过大。")
        return updated


def package_part_types(archive, names):
    if archive.getinfo(CONTENT_TYPES).file_size > 1024 * 1024:
        raise DocumentError("Word 文档部件清单过大。")
    parts = {}
    depth = 0

    def start(name, attributes):
        nonlocal depth
        depth += 1
        if depth == 1 and name != TYPE_NAMESPACE + "Types":
            raise DocumentError("Word 文档部件清单格式无效。")
        if depth != 2 or name != TYPE_NAMESPACE + "Override":
            return
        root = WORD_CONTENT_TYPES.get(attributes.get("ContentType"))
        if not root:
            return
        part = attributes.get("PartName", "")
        if not part.startswith("/"):
            raise DocumentError("Word 文档部件路径无效。")
        path = part[1:]
        if path not in names:
            path = unquote(path)
        if path not in names or path in parts:
            raise DocumentError("Word 文档部件缺失或重复。")
        parts[path] = root

    def end(_):
        nonlocal depth
        depth -= 1

    parser = expat.ParserCreate(namespace_separator=" ")
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject_doctype
    parser.Parse(archive.read(CONTENT_TYPES), True)
    mains = [name for name, root in parts.items() if root == "document"]
    if len(mains) != 1:
        raise DocumentError("Word 文档缺少唯一的正文部件。")
    main = mains[0]
    return [(main, parts.pop(main)), *sorted(parts.items())]


def word_parts(archive):
    members = archive.infolist()
    if len(members) > MAX_MEMBERS or sum(member.file_size for member in members) > MAX_UNPACKED:
        raise DocumentError("Word 文档解压后过大。")
    if any(member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
           for member in members):
        raise DocumentError("不支持加密或使用特殊压缩格式的 Word 文档。")
    names = set(archive.namelist())
    if len(names) != len(members) or CONTENT_TYPES not in names:
        raise DocumentError("文件不是有效的 .docx 文档。")

    for name, root in package_part_types(archive, names):
        if archive.getinfo(name).file_size > MAX_XML:
            raise DocumentError("Word 文档正文过大。")
        yield name, WordPart(archive.read(name), root)


def process(request):
    if not isinstance(request, dict) or not isinstance(request.get("document"), str):
        raise DocumentError("缺少 Word 文档。")
    encoded = request["document"]
    if len(encoded) > 4 * ((MAX_DOCUMENT + 2) // 3):
        raise DocumentError("Word 文档最大支持 24 MB。")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_DOCUMENT:
        raise DocumentError("Word 文档最大支持 24 MB。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if "translations" not in request:
            paragraphs = []
            for _, part in word_parts(archive):
                paragraphs.extend(source for source, _ in part.paragraphs)
                if len(paragraphs) > MAX_PARAGRAPHS:
                    raise DocumentError("Word 文档段落过多。")
            return {"paragraphs": paragraphs}

        translations = request["translations"]
        if not isinstance(translations, list) or len(translations) > MAX_PARAGRAPHS:
            raise DocumentError("译文段落与原文不匹配。")
        if any(not isinstance(value, str) or not value.strip() or INVALID_XML_CHARACTER.search(value)
               for value in translations):
            raise DocumentError("译文包含空段落或无效字符。")

        updated = {}
        offset = 0
        for name, part in word_parts(archive):
            end = offset + len(part.paragraphs)
            if end > len(translations):
                raise DocumentError("译文段落与原文不匹配。")
            changed = part.rebuild(translations[offset:end])
            if changed is not part.original:
                updated[name] = changed
            offset = end
        if offset != len(translations):
            raise DocumentError("译文段落与原文不匹配。")
        if not updated:
            return {"document": encoded}

        unpacked = sum(len(updated[member.filename]) if member.filename in updated else member.file_size
                       for member in archive.infolist())
        if unpacked > MAX_UNPACKED:
            raise DocumentError("翻译后的 Word 文档解压后过大。")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as rebuilt:
            rebuilt.comment = archive.comment
            for member in archive.infolist():
                target = copy.copy(member)
                if member.filename in updated:
                    rebuilt.writestr(target, updated[member.filename])
                else:
                    with archive.open(member) as source, rebuilt.open(target, "w") as output:
                        shutil.copyfileobj(source, output, length=64 * 1024)
        if buffer.tell() > MAX_DOCUMENT:
            raise DocumentError("翻译后的 Word 文档超过 24 MB。")
        return {"document": base64.b64encode(buffer.getbuffer()).decode("ascii")}


def main():
    try:
        request = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(request) > MAX_REQUEST:
            raise DocumentError("Word 文档请求过大。")
        result = process(json.loads(request))
    except DocumentError as error:
        result = {"error": str(error)}
    except (ValueError, LookupError, RecursionError, expat.ExpatError,
            zipfile.BadZipFile, NotImplementedError, zlib.error):
        result = {"error": "无法读取 Word 文档，请确认文件是未加密且完整的 .docx。"}
    sys.stdout.buffer.write((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
