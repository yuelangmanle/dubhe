#!/usr/bin/env python3
"""Generate Dubhe QA fixtures and measure installed models through the local API."""

import argparse
import base64
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "quality_images.swift"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def pdf_text(x, y, value, size=15):
    safe = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F {size} Tf {x} {y} Td ({safe}) Tj ET"


def write_pdf(path, pages):
    """Use PDF's built-in Helvetica so this synthetic input has no font dependency."""
    pages = list(pages)
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(pages)))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for index, content in enumerate(pages):
        stream = content.encode("ascii")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>".encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        pdf.extend(f"{offset:010} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(pdf)


def fixtures(folder):
    folder.mkdir(parents=True, exist_ok=True)
    screenshot = "\n".join([
        "0.93 0.96 0.94 rg 40 215 532 360 re f", "0 g",
        pdf_text(72, 520, "River station", 30),
        pdf_text(72, 450, "Sensor: NX-204", 24),
        pdf_text(72, 410, "Water level: 18.25 m", 24),
        pdf_text(72, 370, "Measured at 08:45", 24),
    ])
    two_column = "\n".join([
        pdf_text(48, 735, "Field report: reservoir survey", 24),
        pdf_text(48, 670, "North station", 17), pdf_text(320, 670, "South station", 17),
        pdf_text(48, 632, "Soil samples were collected."), pdf_text(320, 632, "Water samples were collected."),
        pdf_text(48, 608, "Recorded level: 18.25 m."), pdf_text(320, 608, "Recorded level: 24.75 m."),
        "0.8 G 300 682 m 300 585 l S",
    ])
    table = "\n".join([
        "1 w 62 345 480 190 re S",
        "62 495 m 542 495 l 62 445 m 542 445 l 62 395 m 542 395 l S",
        "222 345 m 222 495 l 382 345 m 382 495 l S",
        pdf_text(200, 510, "Station measurements", 18),
        pdf_text(86, 463, "Site"), pdf_text(249, 463, "Depth (m)"), pdf_text(403, 463, "Sample count"),
        pdf_text(86, 413, "North"), pdf_text(249, 413, "18.25"), pdf_text(403, 413, "12"),
        pdf_text(86, 363, "South"), pdf_text(249, 363, "24.75"), pdf_text(403, 363, "17"),
    ])
    formula = "\n".join([
        pdf_text(160, 390, "x", 40), pdf_text(182, 416, "2", 21),
        pdf_text(211, 390, "+", 40),
        pdf_text(253, 390, "y", 40), pdf_text(276, 416, "2", 21),
        pdf_text(303, 390, "= 25", 40),
    ])
    # Intentionally irregular pen strokes with a printed line for comparison.
    handwriting = "\n".join([
        pdf_text(74, 540, "Sample number:", 22),
        "4.5 w 1 J 1 j 118 477 m 143 475 l 130 454 l 120 426 l S",
        "187 465 m 190 485 229 486 211 455 c 204 443 193 434 186 425 c 215 427 l S",
        "256 458 m 272 477 l 269 432 l 267 425 l S",
    ])
    for name, page in (("screenshot", screenshot), ("two-column", two_column),
                       ("table", table), ("formula", formula), ("pen", handwriting)):
        write_pdf(folder / f"{name}.pdf", [page])
    write_pdf(folder / "resume-20.pdf", [
        "\n".join([
            pdf_text(48, 745, "Reservoir field notebook - synthetic QA", 20),
            pdf_text(48, 711, f"DUBHE-PAGE-{page:02}", 25),
            pdf_text(48, 670, "Survey method", 17),
            pdf_text(48, 635, "A technician recorded the depth."),
            pdf_text(48, 610, f"Sample {page + 100}: {page + 10}.25 m."),
            pdf_text(48, 572, "No personal information is present."),
        ])
        for page in range(1, 21)
    ])
    specification = {
        "ocr": [
            {"id": "screenshot", "pdf": "screenshot.pdf", "mode": "text",
             "terms": [["NX-204"], ["18.25"], ["08:45"]]},
            {"id": "bilingual-screenshot", "mode": "text", "terms": [["河流观测"], ["A-17"], ["2026-05-03"]]},
            {"id": "two-column", "pdf": "two-column.pdf", "mode": "document",
             "terms": [["Soil samples"], ["18.25"], ["Water samples"], ["24.75"]],
             "order": ["North station", "18.25", "South station", "24.75"]},
            {"id": "merged-header-table", "pdf": "table.pdf", "mode": "table",
             "terms": [["18.25"], ["24.75"], ["12"], ["17"]], "merged_header": True},
            {"id": "formula", "pdf": "formula.pdf", "mode": "formula",
             "terms": [["x^2"], ["y^2"], ["=25"]]},
            {"id": "pen-strokes", "pdf": "pen.pdf", "mode": "text", "terms": [["721"]]},
        ],
        "translation": [
            {"id": "en-zh", "source": "en", "target": "zh",
             "text": "The water level at North Station was 18.25 metres. Keep sample ID NX-204.",
             "terms": [["水位", "水面"], ["18.25"], ["NX-204"]]},
            {"id": "zh-en", "source": "zh", "target": "en",
             "text": "北站采集了 12 份样本。编号 NX-204 的记录需要复核。",
             "terms": [["sample"], ["12"], ["NX-204"], ["review", "recheck", "verif", "check"]]},
        ],
    }
    (folder / "cases.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\\(?:mathbf|mathrm|mathit|mathsf|mathcal|boldsymbol)\s*\{([a-z0-9]+)\}", r"\1", text)
    text = re.sub(r"\^\{([0-9]+)\}", r"^\1", text)
    return " ".join(text.split())


def contains(term, text):
    term = normalized(term)
    if term.isdecimal() or re.fullmatch(r"[0-9]+\.[0-9]+", term):
        return re.search(rf"(?<![\d.]){re.escape(term)}(?!\d|\.\d)", text) is not None
    if "^" in term or "=" in term:
        return term in text.replace(" ", "")
    return term in text


def checks(case, text):
    value = normalized(text)
    missing = [" / ".join(group) for group in case["terms"] if not any(contains(term, value) for term in group)]
    order = case.get("order", [])
    begin = 0
    for term in order:
        begin = value.find(normalized(term), begin)
        if begin < 0:
            missing.append(" → ".join(order))
            break
        begin += len(normalized(term))
    # Dubhe emits an HTML table from OTSL with explicit spans.
    if case.get("merged_header") and not re.search(r'colspan\s*=\s*["\']?3(?:["\'\s>])', value):
        missing.append("表头跨三列")
    return missing


def request(port, path, body=None, **kwargs):
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", None if body is None else json.dumps(body).encode(), headers)
    return OPENER.open(req, timeout=kwargs.get("timeout", 250))


def json_request(port, path, body=None):
    with request(port, path, body) as response:
        return json.load(response)


def render(case, folder, scratch, native):
    output = scratch / (case["id"] + ".png")
    if native:
        command = [str(native), "raster", str(folder / case["pdf"]), str(output)] if "pdf" in case else [
            str(native), "bilingual", str(output)]
    else:
        pdftoppm = shutil.which("pdftoppm")
        if "pdf" not in case or not pdftoppm:
            return None
        command = [pdftoppm, "-r", "150", "-singlefile", "-png", str(folder / case["pdf"]), str(output.with_suffix(""))]
    subprocess.run(command, check=True)
    return output


def measure(case, invoke, provider, baseline=False):
    started = time.perf_counter()
    try:
        output = invoke()
        if not isinstance(output, str) or not output.strip():
            raise ValueError("没有可检查的文字输出。")
        missing = checks(case, output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return {"case": case["id"], "provider": provider, "status": "failed", "baseline": baseline,
                "elapsed_ms": round((time.perf_counter() - started) * 1000), "error": str(error)}
    return {"case": case["id"], "provider": provider, "status": "failed" if missing else "passed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "missing_checks": missing, "output": output, "baseline": baseline}


def evaluate(args):
    spec = json.loads((args.folder / "cases.json").read_text(encoding="utf-8"))
    results = []
    if args.suite in ("all", "ocr"):
        options = json_request(args.port, "/api/scan")
        installed = options.get("installed") and options.get("runtime_available")
        with tempfile.TemporaryDirectory(prefix="dubhe-quality-") as temp:
            scratch = Path(temp)
            native = None
            if sys.platform == "darwin":
                native = scratch / "quality-images"
                subprocess.run(["swiftc", str(RENDERER), "-o", str(native)], check=True)
            for case in spec["ocr"]:
                image = render(case, args.folder, scratch, native) if installed or (args.vision and native) else None
                if not installed or image is None:
                    results.append({"case": case["id"], "provider": "Xiaomi-OCR-0", "status": "skipped",
                                    "reason": "模型或运行时不可用" if not installed else "缺少此图像的渲染工具"})
                else:
                    data = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
                    results.append(measure(case, lambda: json_request(
                        args.port, "/api/scan", {"mode": case["mode"], "image": data})["text"], "Xiaomi-OCR-0"))
                # Vision returns text only. Structured tables and formula parsing are separate capabilities.
                if args.vision and native and image and case["mode"] == "text":
                    results.append(measure(case, lambda: subprocess.check_output(
                        [str(native), "vision", str(image)], text=True), "Apple Vision", baseline=True))
    if args.suite in ("all", "translation"):
        options = json_request(args.port, "/api/translate")
        for model in options["models"]:
            for case in spec["translation"]:
                if not model["installed"] or not options["runtime_available"]:
                    results.append({"case": case["id"], "provider": model["id"], "status": "skipped",
                                    "reason": "模型或运行时不可用"})
                    continue
                body = {key: case[key] for key in ("source", "target", "text")}
                results.append(measure(case, lambda: json_request(
                    args.port, "/api/translate", {**body, "model": model["id"]})["text"], model["id"]))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    measured = [item for item in results if item["status"] != "skipped" and not item.get("baseline")]
    return int(not measured or any(item["status"] != "passed" for item in measured))


def read_completion(response, started, clock=time.perf_counter):
    first_visible = None
    usage = {}
    done = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event or event.get("dubhe", {}).get("type") == "error":
            raise ValueError(f"对话请求失败：{event.get('error') or event['dubhe'].get('message')}")
        usage = event.get("usage") or usage
        choices = event.get("choices", [])
        content = choices[0].get("delta", {}).get("content") if choices else None
        if first_visible is None and isinstance(content, str) and content.strip():
            first_visible = round((clock() - started) * 1000)
    if not done or first_visible is None:
        raise ValueError("对话未完整结束，或没有可见正文。")
    return {"first_visible_ms": first_visible, "total_ms": round((clock() - started) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


def chat(args):
    state = json_request(args.port, "/api/state")
    if not state.get("loaded"):
        raise ValueError("请先加载一个对话模型。")
    phase = "first_server_request" if state.get("requests") == 0 else "already_served"
    short = "用一句中文描述月球。"
    long = "\n".join(f"Station {n:02}: survey team recorded water level {10 + n}.25 metres."
                     for n in range(1, 41))
    cases = [(phase, short), *[("warm_short", short)] * args.repeat,
             *[("long_input", long + "\n用一句中文总结这些记录。")] * args.repeat]
    rows = []
    for index, (name, prompt) in enumerate(cases, start=1):
        # Alter the first content tokens to limit reuse of the previous request's prompt.
        body = {"model": state["model_id"], "stream": True, "max_tokens": 256, "temperature": 0,
                "thinking": False,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": f"QA-{index:03}\n{prompt}"}]}
        started = time.perf_counter()
        try:
            with request(args.port, "/v1/chat/completions", body) as response:
                rows.append({"phase": name, **read_completion(response, started)})
        except (OSError, ValueError) as error:
            rows.append({"phase": name, "error": str(error),
                         "total_ms": round((time.perf_counter() - started) * 1000)})
    medians = {}
    for name in ("warm_short", "long_input"):
        samples = [row["first_visible_ms"] for row in rows if row["phase"] == name and "first_visible_ms" in row]
        medians[name] = statistics.median(samples) if samples else None
    print(json.dumps({"version": state.get("version"), "model": state["model_id"],
                      "hardware": state.get("hardware"), "requests_before": state.get("requests"),
                      "median_first_visible_ms": medians, "results": rows}, ensure_ascii=False, indent=2))
    return int(any("error" in row for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("fixtures", help="生成可重复的 PDF 和中英文本用例")
    create.add_argument("folder", type=Path)
    run = commands.add_parser("evaluate", help="用已安装模型测 OCR 与翻译")
    run.add_argument("folder", type=Path)
    run.add_argument("--port", type=int, default=7777)
    run.add_argument("--suite", choices=("all", "ocr", "translation"), default="all")
    run.add_argument("--vision", action="store_true", help="在 macOS 同时测系统 OCR 的文字基线")
    bench = commands.add_parser("chat", help="测首个请求、热启动和长文的首字时间")
    bench.add_argument("--port", type=int, default=7777)
    bench.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "fixtures":
            fixtures(args.folder)
        elif args.command == "evaluate":
            return evaluate(args)
        else:
            if args.repeat < 1:
                parser.error("--repeat 必须大于 0")
            return chat(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"质量检查失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
