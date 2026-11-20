"""Exercise the DOCX adapter through the same JSON protocol as the server."""

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile


ADAPTER = Path(__file__).resolve().parents[1] / "tools" / "translation_docx.py"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOCUMENT = "word/document.xml"


def fixture_parts():
    return {
        "[Content_Types].xml": f'''<Types xmlns="{CT}">
            <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
            <Default Extension="png" ContentType="image/png"/>
            <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
            <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
        </Types>'''.encode(),
        "_rels/.rels": f'''<Relationships xmlns="{RELS}">
            <Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/>
        </Relationships>'''.encode(),
        "word/_rels/document.xml.rels": f'''<Relationships xmlns="{RELS}">
            <Relationship Id="rIdHeader" Type="{R}/header" Target="header1.xml"/>
            <Relationship Id="rIdStyles" Type="{R}/styles" Target="styles.xml"/>
            <Relationship Id="rIdImage" Type="{R}/image" Target="media/image1.png"/>
            <Relationship Id="rIdLink" Type="{R}/hyperlink" Target="https://example.org/?a=1&amp;b=2" TargetMode="External"/>
        </Relationships>'''.encode(),
        "word/styles.xml": f'''<w:styles xmlns:w="{W}">
            <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
        </w:styles>'''.encode(),
        "word/header1.xml": f'''<w:hdr xmlns:w="{W}">
            <w:p><w:r><w:t xml:space="preserve">   Report header  </w:t></w:r></w:p>
        </w:hdr>'''.encode(),
        "word/media/image1.png": base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lJcAAAAASUVORK5CYII="
        ),
        DOCUMENT: f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:m="{M}" xmlns:r="{R}" xmlns:wp="{WP}">
    <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
            <w:r><w:rPr><w:b/></w:rPr><w:t>River</w:t></w:r>
            <w:r><w:t xml:space="preserve"> report</w:t></w:r>
        </w:p>
        <w:tbl><w:tblPr/><w:tblGrid><w:gridCol w:w="2880"/></w:tblGrid>
            <w:tr><w:tc><w:tcPr><w:tcW w:w="2880" w:type="dxa"/></w:tcPr>
                <w:p><w:r><w:t>Flow at {{station}} &amp; 20 m³/s</w:t></w:r></w:p>
            </w:tc></w:tr>
        </w:tbl>
        <w:p><w:r><w:t xml:space="preserve">Before </w:t></w:r>
            <m:oMath><m:r><m:t>x=y</m:t></m:r></m:oMath>
            <w:r><w:t xml:space="preserve"> after</w:t><w:tab/><w:t>Diagram</w:t></w:r>
            <w:r><w:drawing><wp:inline><wp:extent cx="12700" cy="12700"/>
                <wp:docPr id="1" name="Image"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                    <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                        <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr/>
                            <pic:blipFill><a:blip r:embed="rIdImage"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr/>
                        </pic:pic>
                    </a:graphicData>
                </a:graphic>
            </wp:inline></w:drawing></w:r>
            <w:hyperlink r:id="rIdLink"><w:r><w:t>Web page</w:t></w:r></w:hyperlink>
        </w:p>
        <w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> REF _Ref1 </w:instrText></w:r>
            <w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Reference result</w:t></w:r>
            <w:r><w:fldChar w:fldCharType="end"/></w:r><w:r><w:t>Source</w:t></w:r>
            <w:del><w:r><w:delText>Deleted sentence</w:delText></w:r></w:del>
        </w:p>
        <w:sectPr><w:headerReference w:type="default" r:id="rIdHeader"/></w:sectPr>
    </w:body>
</w:document>'''.encode(),
    }


def package(parts):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.comment = b"Fixture archive"
        for name, raw in parts.items():
            archive.writestr(name, raw)
    return base64.b64encode(buffer.getvalue()).decode()


def adapter(request):
    process = subprocess.run(
        [sys.executable, "-I", str(ADAPTER)], input=json.dumps(request), encoding="utf-8",
        capture_output=True, timeout=20,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


class TranslationDocxTests(unittest.TestCase):
    def test_round_trip_translates_text_in_place_and_preserves_package(self):
        parts = fixture_parts()
        document = package(parts)
        extracted = adapter({"document": document})
        self.assertEqual(extracted, {"paragraphs": [
            "River report", "Flow at {station} & 20 m³/s", "Before ", " after", "Diagram",
            "Web page", "Source", "   Report header  ",
        ]})

        result = adapter({"document": document, "translations": [
            "河流报告", "{station} 的流量 & 20 m³/s", "前文", "后文", "示意图", "网页", "来源", "报告页眉",
        ]})
        with ZipFile(io.BytesIO(base64.b64decode(result["document"]))) as archive:
            self.assertEqual(archive.namelist(), list(parts))
            self.assertEqual(archive.comment, b"Fixture archive")
            self.assertIsNone(archive.testzip())
            for name, raw in parts.items():
                if name not in {DOCUMENT, "word/header1.xml"}:
                    self.assertEqual(archive.read(name), raw, name)

            output_xml = archive.read(DOCUMENT)
            for preserved in (
                b'<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>',
                b'<m:oMath><m:r><m:t>x=y</m:t></m:r></m:oMath>',
                b'<a:blip r:embed="rIdImage"/>', b'<w:tab/>',
                b'<w:instrText> REF _Ref1 </w:instrText>', b'<w:t>Reference result</w:t>',
                b'<w:delText>Deleted sentence</w:delText>',
            ):
                self.assertIn(preserved, output_xml)
            root = ElementTree.fromstring(output_xml)
            ns = {"w": W}
            heading = root.find("w:body/w:p", ns)
            self.assertIsNotNone(heading.find("w:r/w:rPr/w:b", ns))
            runs = heading.findall("w:r/w:t", ns)
            self.assertEqual([run.text for run in runs], ["河流", "报告"])
            self.assertEqual(root.find(".//w:tc/w:p/w:r/w:t", ns).text, "{station} 的流量 & 20 m³/s")
            self.assertEqual(root.find(".//w:hyperlink/w:r/w:t", ns).text, "网页")

        self.assertEqual(adapter(result)["paragraphs"], [
            "河流报告", "{station} 的流量 & 20 m³/s", "前文 ", " 后文", "示意图", "网页", "来源", "   报告页眉  ",
        ])

    def test_rejects_invalid_archives_and_translation_lists(self):
        self.assertIn("error", adapter({"document": "not base64"}))
        self.assertIn("error", adapter({"document": package({"other.txt": b"not Word"})}))
        parts = fixture_parts()
        document = package(parts)
        self.assertIn("error", adapter({"document": document, "translations": ["only one"]}))
        self.assertIn("error", adapter({"document": document, "translations": ["illegal\x01"] * 8}))
        parts[DOCUMENT] = b'<!DOCTYPE x [<!ENTITY y "expanded">]>' + parts[DOCUMENT].split(b"?>", 1)[1]
        self.assertIn("error", adapter({"document": package(parts)}))
        parts[DOCUMENT] = b" " * (32 * 1024 * 1024)
        self.assertIn("error", adapter({"document": package(parts)}))

    def test_uses_declared_part_names_and_preserves_xml_encoding(self):
        parts = fixture_parts()
        renamed = "revisions/report-main.xml"
        parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace(DOCUMENT.encode(), renamed.encode())
        parts["_rels/.rels"] = parts["_rels/.rels"].replace(DOCUMENT.encode(), renamed.encode())
        xml = parts.pop(DOCUMENT).decode().replace('encoding="UTF-8"', 'encoding="UTF-16"')
        parts[renamed] = b"\xfe\xff" + xml.encode("utf-16-be")
        parts["word/header-decorations.xml"] = b"<decorations/>"

        document = package(parts)
        originals = adapter({"document": document})["paragraphs"]
        self.assertEqual(originals[0], "River report")
        self.assertEqual(adapter({"document": document, "translations": originals}), {"document": document})
        originals[0] = "河流报告"
        updated = adapter({"document": document, "translations": originals})
        with ZipFile(io.BytesIO(base64.b64decode(updated["document"]))) as archive:
            self.assertEqual(archive.read("word/header-decorations.xml"), b"<decorations/>")
            rebuilt = archive.read(renamed)
            self.assertTrue(rebuilt.startswith(b"\xfe\xff"))
            texts = ElementTree.fromstring(rebuilt).findall(f".//{{{W}}}t")
            self.assertEqual("".join(node.text for node in texts[:2]), "河流报告")


if __name__ == "__main__":
    unittest.main()
