#!/usr/bin/env python3
"""Service regression tests using temporary models, data, and loopback ports."""

import base64
import concurrent.futures
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from test_model_backend import FAKE_SERVER


ROOT = Path(__file__).resolve().parents[1]
BINARY = (ROOT / "build/dubhed").resolve()
MODEL = os.getenv("DUBHE_TEST_MODEL")
AUDIO_MODEL = os.getenv("DUBHE_TEST_AUDIO_MODEL")
AUDIO_TOWER = os.getenv("DUBHE_TEST_AUDIO_MMPROJ")
AUDIO_WAV = os.getenv("DUBHE_TEST_AUDIO_WAV")
AUDIO_TEXT = os.getenv("DUBHE_TEST_AUDIO_TEXT", "this is a local test")
BATCH_MODEL = os.getenv("DUBHE_TEST_BATCH_MODEL")
QWEN35_MODEL = os.getenv("DUBHE_TEST_QWEN35_MODEL")
SCAN_IMAGE = ("data:image/png;base64,"
              "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+l0V8AAAAASUVORK5CYII=")


def request(base, path, body=None, *, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


@contextmanager
def server(root, *args, env=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            [str(BINARY), "--port", str(port), "--web", str(ROOT / "web"),
             "--data-dir", str(root / "data"), "--model-dir", str(root / "models"),
             "--tmp-dir", str(root), "--no-shell", "--no-write", "--no-net", *args],
            cwd=root, stdout=log, stderr=log, env=env,
        )
        try:
            until = time.monotonic() + 20
            while time.monotonic() < until and proc.poll() is None:
                try:
                    if request(base, "/healthz")[0] == 200:
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace"))
            yield base
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def conversation(key):
    return {"id": key, "title": key, "messages": [], "at": 1}


class ServiceTest(unittest.TestCase):
    @unittest.skipUnless(QWEN35_MODEL, "DUBHE_TEST_QWEN35_MODEL is unset")
    def test_qwen35_search_uses_one_short_tool_prompt(self):
        body = {
            "messages": [
                {"role": "system", "content": "你是聊天助手。"},
                {"role": "system", "content": "用一句话回答。"},
                {"role": "user", "content": "你是谁？"},
            ],
            "temperature": 0, "max_tokens": 1,
        }
        with tempfile.TemporaryDirectory() as tmp, server(
            Path(tmp), "-m", QWEN35_MODEL, "--no-mmproj", "-c", "4096",
        ) as base:
            def prompt_tokens(extra):
                status, result = request(base, "/v1/chat/completions", {**body, **extra}, timeout=90)
                self.assertEqual(status, 200, result)
                return result["usage"]["prompt_tokens"]

            plain = prompt_tokens({})
            search = prompt_tokens({"dubhe_search": True})
            general = prompt_tokens({"dubhe_tools": True})
            self.assertLess(50, search - plain, "search must reach the model")
            self.assertLess(search - plain, 240, "search must not repeat tool descriptions")
            self.assertLess(search, general, "search must not send unrelated tools")

    @unittest.skipUnless(QWEN35_MODEL, "DUBHE_TEST_QWEN35_MODEL is unset")
    def test_qwen35_translates_with_native_cpu_and_metal(self):
        body = {
            "messages": [
                {"role": "system", "content": "你是专业翻译引擎。直接输出译文。"},
                {"role": "user", "content": "翻译成简体中文：\n\nThe dam did not overtop, and the spillway gates remained closed."},
            ],
            "temperature": 0, "max_tokens": 48, "no_think": True,
        }
        backends = ("cpu", "metal") if sys.platform == "darwin" else ("cpu",)
        for backend in backends:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, server(
                Path(tmp), "-m", QWEN35_MODEL, "--no-mmproj", "-c", "1024",
                *(["--no-gpu"] if backend == "cpu" else []),
            ) as base:
                code, result = request(base, "/v1/chat/completions", body, timeout=60)
                self.assertEqual(code, 200)
                text = result["choices"][0]["message"]["content"]
                self.assertRegex(text, r"坝.*闸.*关")

    def test_translation_catalog_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp, server(Path(tmp)) as base:
            code, options = request(base, "/api/translate")
            self.assertEqual(code, 200)
            self.assertEqual(options["default_model"], "hy-mt2-18b-q4")
            expected = {"hy-mt2-18b-q4", "index-translate-2b-q4"}
            self.assertEqual({item["id"] for item in options["models"]}, expected)
            self.assertFalse(any(item["installed"] for item in options["models"]))
            _, store = request(base, "/api/store")
            self.assertEqual({item["id"] for item in store["items"] if item.get("translation")}, expected)
            job = {"source": "en", "target": "zh", "text": "A sentence."}
            self.assertEqual(request(base, "/api/translate", job)[0], 404)
            self.assertEqual(request(base, "/api/translate", {**job, "model": "unknown"})[0], 400)
            self.assertEqual(request(base, "/api/translate", {**job, "target": "unknown"})[0], 400)

    def test_scan_catalog_and_safe_input(self):
        # A valid 1 x 1 PNG keeps the regression independent of image libraries.
        models = {"xiaomi-ocr-0-q4": "Xiaomi-OCR-0.Q4_K_M.gguf",
                  "xiaomi-ocr-0-vision": "Xiaomi-OCR-0.mmproj-bf16.gguf"}
        with tempfile.TemporaryDirectory() as tmp, server(Path(tmp)) as base:
            code, options = request(base, "/api/scan")
            self.assertEqual(code, 200)
            self.assertFalse(options["installed"])
            self.assertEqual({entry["id"] for entry in options["files"]}, models.keys())
            _, store = request(base, "/api/store")
            self.assertEqual({entry["id"] for entry in store["items"] if entry.get("scan")}, models.keys())
            job = {"mode": "text", "image": SCAN_IMAGE}
            self.assertEqual(request(base, "/api/scan", job)[0], 404)
            png = base64.b64decode(SCAN_IMAGE.split(",", 1)[1])
            oversized = "data:image/png;base64," + base64.b64encode(
                png[:16] + struct.pack(">I", 65536) + png[20:]).decode()
            for invalid in ({"mode": "other"}, {"mode": "extract", "fields": [" "]},
                            {"mode": "ask"}, {"image": "/etc/hosts"},
                            {"image": "https://example.org/photo.png"},
                            {"image": "data:image/png;base64,aGVsbG8="},
                            {"image": SCAN_IMAGE[:-1] + "!"}, {"image": oversized}):
                with self.subTest(invalid=invalid):
                    self.assertEqual(request(base, "/api/scan", {**job, **invalid})[0], 400)
            root = Path(tmp) / "models"
            root.mkdir(exist_ok=True)
            for filename in models.values():
                model = root / filename
                model.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
                code, result = request(base, "/api/load", {"path": str(model)})
                self.assertEqual(code, 400)
                self.assertIn("扫描", result["error"])
            self.assertEqual(request(base, "/api/models")[1]["items"], [])

    @unittest.skipUnless(sys.platform == "darwin", "OCR worker requires the macOS runtime")
    def test_scan_worker_shapes_outputs_and_cancels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "llama-server"
            fake.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SERVER))
            fake.chmod(0o700)
            wrapper = root / "worker-python"
            wrapper.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                import runpy, subprocess, sys, time
                terminate = subprocess.Popen.terminate
                def delayed_terminate(process):
                    time.sleep(0.06)
                    terminate(process)
                subprocess.Popen.terminate = delayed_terminate
                sys.argv = sys.argv[2:]
                runpy.run_path(sys.argv[0], run_name="__main__")
            """))
            wrapper.chmod(0o700)
            child = root / "child.pid"
            started = root / "scan-started"
            shutdown = root / "child-shutdown"
            env = dict(os.environ, DUBHE_LLAMA_SERVER=str(fake), DUBHE_TEST_CHILD_PID=str(child),
                       DUBHE_TEST_SCAN_STARTED=str(started), DUBHE_TEST_CHILD_SHUTDOWN=str(shutdown),
                       DUBHE_BACKEND_PYTHON=str(wrapper))
            models = root / "models"
            models.mkdir()
            weights = models / "Xiaomi-OCR-0.Q4_K_M.gguf"
            projector = models / "Xiaomi-OCR-0.mmproj-bf16.gguf"
            for path, size in ((weights, 529297664), (projector, 207346624)):
                with path.open("wb") as file:
                    file.write(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
                    file.truncate(size)

            with server(root, env=env) as base:
                options = request(base, "/api/scan")[1]
                self.assertTrue(options["installed"] and options["runtime_available"])
                def recognize(mode, **fields):
                    code, result = request(base, "/api/scan", {"mode": mode, "image": SCAN_IMAGE, **fields})
                    self.assertEqual(code, 200, result)
                    return result

                self.assertEqual(recognize("text"), {"kind": "text", "text": "Water level 10.2 m"})
                self.assertEqual(recognize("formula"), {"kind": "latex", "text": "x^2+y^2"})
                self.assertEqual(recognize("ask", question="What is the water level?")["kind"], "text")
                data = recognize("extract", fields=["日期", "地点"])
                self.assertEqual(data["kind"], "json")
                self.assertEqual(json.loads(data["text"]), {"日期": "11/20", "地点": ""})
                document = recognize("document")["text"]
                self.assertTrue(document.startswith("# Report"))
                self.assertTrue(document.rstrip().endswith("Notes"))
                self.assertIn('<td colspan="2">Header</td>', document)
                table = recognize("table")["text"]
                self.assertIn('colspan="2" rowspan="2"', table)
                self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", table)

                host, port = base.removeprefix("http://").split(":")
                session = http.client.HTTPConnection(host, int(port), timeout=5)
                try:
                    session.request("POST", "/api/scan", json.dumps({
                        "mode": "ask", "image": SCAN_IMAGE, "question": "slow-test",
                    }), {"Content-Type": "application/json"})
                    deadline = time.monotonic() + 5
                    while not started.exists() and time.monotonic() < deadline:
                        time.sleep(0.025)
                    self.assertTrue(started.exists(), "slow scan did not reach the worker")
                    self.assertEqual(request(base, "/api/models/delete", {"path": str(projector)})[0], 409)
                    pid = int(child.read_text())
                finally:
                    session.close()
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.025)
                else:
                    self.fail("worker still alive after the scan client closed")
                self.assertEqual(shutdown.read_text().splitlines(), ["SIGTERM"])
                self.assertEqual(recognize("text")["text"], "Water level 10.2 m")
                self.assertEqual(request(base, "/api/models/delete", {"path": str(projector)})[0], 200)
                self.assertFalse(request(base, "/api/scan")[1]["installed"])

    def test_remove_imported_model_only_removes_the_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picked = root / "picked.gguf"
            picked.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
            with server(root) as base:
                self.assertEqual(request(base, "/api/models/import", {"path": str(picked)})[0], 200)
                self.assertEqual(request(base, "/api/models/delete", {"path": str(picked)})[0], 400)
                entry = root / "models/picked.gguf"
                self.assertEqual(request(base, "/api/models/delete", {"path": str(entry)})[0], 200)
                self.assertFalse(entry.is_symlink())
                self.assertTrue(picked.exists())
                self.assertEqual(request(base, "/api/models")[1]["items"], [])

    @unittest.skipUnless(MODEL, "DUBHE_TEST_MODEL is unset")
    def test_current_model_can_be_unloaded_then_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            entry = model_dir / "active.gguf"
            entry.symlink_to(Path(MODEL).resolve())
            with server(root, "-m", str(entry), "--no-mmproj", "-c", "512") as base:
                self.assertEqual(request(base, "/api/models/delete", {"path": str(entry)})[0], 409)
                self.assertEqual(request(base, "/api/unload", {})[0], 200)
                self.assertFalse(request(base, "/api/state")[1]["loaded"])
                self.assertEqual(request(base, "/api/models/delete", {"path": str(entry)})[0], 200)
                self.assertTrue(Path(MODEL).exists())

    def test_first_import_creates_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picked = root / "picked.gguf"
            picked.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
            with server(root) as base:
                status, result = request(base, "/api/models/import", {"path": str(picked)})
                self.assertEqual((status, result.get("ok")), (200, True))
                self.assertEqual((root / "models/picked.gguf").resolve(), picked.resolve())
                status, result = request(base, "/api/models/import", {"path": str(picked)})
                self.assertEqual((status, result.get("ok")), (200, True))

    def test_history_is_port_independent_and_applies_incremental_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with server(root) as first, server(root) as second:
                self.assertEqual(request(first, "/api/history"), (200, {"items": []}))
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    for status, result in pool.map(
                        lambda n: request(first if n < 3 else second, "/api/history",
                                          {"upsert": [conversation(f"c{n}")]}),
                        range(6),
                    ):
                        self.assertEqual((status, result.get("ok")), (200, True))
                self.assertEqual(
                    request(second, "/api/history", {"remove": ["c0"]})[0], 200)
                self.assertEqual(
                    request(first, "/api/history", {"import": [conversation("c0")]})[0], 200)
            with server(root) as fresh:
                status, result = request(fresh, "/api/history")
                self.assertEqual(status, 200)
                self.assertEqual({c["id"] for c in result["items"]}, {f"c{n}" for n in range(1, 6)})

    @unittest.skipUnless(MODEL, "DUBHE_TEST_MODEL is unset")
    def test_generation_reports_limit_and_rejects_stale_model(self):
        with tempfile.TemporaryDirectory() as tmp, server(
            Path(tmp), "-m", str(Path(MODEL).resolve()), "--no-mmproj", "--no-gpu", "-c", "512"
        ) as base:
            query = {"messages": [{"role": "user", "content": "Name a season in one word."}],
                     "temperature": 0, "max_tokens": 1}
            status, completion = request(base, "/v1/chat/completions", query)
            self.assertEqual(status, 200)
            self.assertEqual(completion["choices"][0]["finish_reason"], "length")
            status, error = request(base, "/v1/chat/completions", {
                **query, "dubhe_model_path": str(Path(tmp) / "another.gguf")
            })
            self.assertEqual(status, 409)
            self.assertTrue(error["error"]["message"])

    @unittest.skipUnless(AUDIO_MODEL and AUDIO_TOWER and AUDIO_WAV, "audio fixtures are unset")
    def test_audio_only_uses_transcription_prompt(self):
        with tempfile.TemporaryDirectory() as tmp, server(
            Path(tmp), "-m", AUDIO_MODEL, "--mmproj", AUDIO_TOWER, "-c", "2048"
        ) as base:
            data = base64.b64encode(Path(AUDIO_WAV).read_bytes()).decode()
            status, completion = request(base, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": [{
                    "type": "input_audio", "input_audio": {"data": data, "format": "wav"}
                }]}],
                "temperature": 0, "max_tokens": 64,
            })
            self.assertEqual(status, 200)
            self.assertIn(AUDIO_TEXT.casefold(),
                          completion["choices"][0]["message"]["content"].casefold())

    @unittest.skipUnless(MODEL, "DUBHE_TEST_MODEL is unset")
    def test_structured_response_is_constrained_during_generation(self):
        with tempfile.TemporaryDirectory() as tmp, server(
            Path(tmp), "-m", str(Path(MODEL).resolve()), "--no-mmproj", "--no-gpu", "-c", "1024"
        ) as base:
            status, completion = request(base, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Write one sentence about the sky."}],
                "temperature": 0, "max_tokens": 100,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "sample", "strict": True, "schema": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "label": {"enum": ["中文"]},
                            "exact": {"const": 1234567.891234},
                            "values": {"type": "array", "items": {"type": "integer", "const": 3},
                                       "minItems": 2, "maxItems": 2},
                        },
                        "required": ["label", "exact", "values"],
                    },
                }},
            })
            self.assertEqual(status, 200)
            answer = completion["choices"][0]
            self.assertEqual(answer["finish_reason"], "stop")
            self.assertEqual(json.loads(answer["message"]["content"]),
                             {"label": "中文", "exact": 1234567.891234, "values": [3, 3]})

    @unittest.skipUnless(BATCH_MODEL, "a Metal-capable DUBHE_TEST_BATCH_MODEL is required")
    def test_http_clients_use_separate_batched_sequences(self):
        with tempfile.TemporaryDirectory() as tmp, server(
            Path(tmp), "-m", str(Path(BATCH_MODEL).resolve()), "--no-mmproj", "-c", "1024",
            "--parallel", "2"
        ) as base:
            self.assertEqual(request(base, "/api/state")[1].get("parallel"), 2)
            barrier = threading.Barrier(2)

            def ask(question, expected):
                barrier.wait(timeout=5)
                code, result = request(base, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0, "max_tokens": 24,
                })
                self.assertEqual(code, 200)
                self.assertEqual(result.get("dubhe", {}).get("batch_size"), 2)
                self.assertIn(expected, result["choices"][0]["message"]["content"].lower())

            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                first = executor.submit(ask, "What is the capital of France? Answer briefly.", "paris")
                second = executor.submit(ask, "What is the capital of Japan? Answer briefly.", "tokyo")
                first.result()
                second.result()


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        BINARY = Path(sys.argv.pop(1)).resolve()
    unittest.main()
